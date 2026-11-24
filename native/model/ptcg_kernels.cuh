// Primitive CUDA ops for the native PTCG model (fp32 build only).
// Every op has a hand-written backward. GEMMs run TF32 by default
// (PTCG_TF32=0 disables) to match torch.set_float32_matmul_precision('high')
// in the production pufferl 3.0 stack.
#ifndef PTCG_KERNELS_CUH
#define PTCG_KERNELS_CUH

#include <cublas_v2.h>
#include <math.h>

#ifdef PRECISION_FLOAT
// precision_t == float in this build; the model requires it.
#else
#error "ptcg native model requires the --float build (obs carry integer ids)"
#endif

#define PTB 256
static inline int ptg(long n) { return (int)((n + PTB - 1) / PTB); }

// ------------------------------------------------------------------ gemm
static cublasComputeType_t ptcg_compute_type(void) {
    static int cached = -1;
    if (cached < 0) {
        const char* e = getenv("PTCG_TF32");
        cached = (e && e[0] == '0') ? 0 : 1;
    }
    return cached ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}

static cublasHandle_t ptcg_cublas(void) {
    static thread_local cublasHandle_t h = nullptr;
    if (!h) {
        cublasCreate(&h);
        void* ws = nullptr;
        cudaMalloc(&ws, 32u << 20);
        cublasSetWorkspace(h, ws, 32u << 20);
    }
    return h;
}

// Row-major C(M,N) = op_a(A) @ op_b(B) [+ beta*C]
static inline void pt_gemm(cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, const float* B, float* C,
        cudaStream_t stream, float alpha = 1.f, float beta = 0.f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasHandle_t h = ptcg_cublas();
    cublasSetStream(h, stream);
    cublasGemmEx(h, op_b, op_a, N, M, K, &alpha,
        B, CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta,
        C, CUDA_R_32F, N, ptcg_compute_type(), CUBLAS_GEMM_DEFAULT);
}

// Batched row-major: C[b](M,N) = op_a(A[b]) @ op_b(B[b])
static inline void pt_gemm_batched(cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, const float* A, long strideA,
        const float* B, long strideB, float* C, long strideC, int batch,
        cudaStream_t stream, float alpha = 1.f, float beta = 0.f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    cublasHandle_t h = ptcg_cublas();
    cublasSetStream(h, stream);
    cublasGemmStridedBatchedEx(h, op_b, op_a, N, M, K, &alpha,
        B, CUDA_R_32F, ldb, strideB, A, CUDA_R_32F, lda, strideA, &beta,
        C, CUDA_R_32F, N, strideC, batch, ptcg_compute_type(),
        CUBLAS_GEMM_DEFAULT);
}

// Linear: Y(N,O) = X(N,I) @ W(O,I)^T + b(O)
__global__ void pt_bias_add(float* y, const float* b, long n, int o) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] += b[i % o];
}

static inline void pt_linear(const float* x, const float* w, const float* b,
        float* y, int N, int I, int O, cudaStream_t s) {
    pt_gemm(CUBLAS_OP_N, CUBLAS_OP_T, N, O, I, x, w, y, s);
    if (b) pt_bias_add<<<ptg((long)N * O), PTB, 0, s>>>(y, b, (long)N * O, O);
}

// Linear backward: dX = dY @ W ; dW += dY^T @ X ; db += colsum(dY)
// Row-sliced colsum: each block owns (row-slice, column-tile); threads sum
// their column over the slice (coalesced) and atomicAdd the partial into db.
// A one-thread-per-column loop would leave 128 threads covering 165K rows at
// token_proj bwd. Atomic merge order makes db fp-reorder-nondeterministic,
// same as the embedding scatters elsewhere in this backward.
#define PT_BGRAD_ROWS 256
__global__ void pt_bias_grad(float* db, const float* dy, long N, int O) {
    int o = blockIdx.y * blockDim.x + threadIdx.x;
    if (o >= O) return;
    long n0 = (long)blockIdx.x * PT_BGRAD_ROWS;
    long n1 = n0 + PT_BGRAD_ROWS < N ? n0 + PT_BGRAD_ROWS : N;
    float s = 0.f;
    for (long n = n0; n < n1; n++) s += dy[n * O + o];
    atomicAdd(&db[o], s);
}

static inline void pt_linear_bwd(const float* x, const float* w, const float* dy,
        float* dx, float* dw, float* db, int N, int I, int O, cudaStream_t s,
        float beta_dw = 1.f) {
    if (dw) pt_gemm(CUBLAS_OP_T, CUBLAS_OP_N, O, I, N, dy, x, dw, s, 1.f, beta_dw);
    if (db) {
        dim3 g((unsigned)(((long)N + PT_BGRAD_ROWS - 1) / PT_BGRAD_ROWS),
               (unsigned)((O + PTB - 1) / PTB));
        pt_bias_grad<<<g, PTB, 0, s>>>(db, dy, (long)N, O);
    }
    if (dx) pt_gemm(CUBLAS_OP_N, CUBLAS_OP_N, N, I, O, dy, w, dx, s);
}

// Same, but ACCUMULATE dX (beta=1). MoE runs one masked head per expert and
// sums their input gradients; the plain version overwrites dX.
static inline void pt_linear_bwd_acc(const float* x, const float* w,
        const float* dy, float* dx, float* dw, float* db, int N, int I, int O,
        cudaStream_t s, float beta_dw = 1.f) {
    if (dw) pt_gemm(CUBLAS_OP_T, CUBLAS_OP_N, O, I, N, dy, x, dw, s, 1.f, beta_dw);
    if (db) {
        dim3 g((unsigned)(((long)N + PT_BGRAD_ROWS - 1) / PT_BGRAD_ROWS),
               (unsigned)((O + PTB - 1) / PTB));
        pt_bias_grad<<<g, PTB, 0, s>>>(db, dy, (long)N, O);
    }
    if (dx) pt_gemm(CUBLAS_OP_N, CUBLAS_OP_N, N, I, O, dy, w, dx, s, 1.f, 1.f);
}

// ------------------------------------------------------------------ gelu
__global__ void pt_gelu_fwd(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    }
}

// dY -> dX given saved input x
__global__ void pt_gelu_bwd(float* dx, const float* dy, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        float cdf = 0.5f * (1.f + erff(v * 0.70710678118654752f));
        float pdf = 0.3989422804014327f * expf(-0.5f * v * v);
        dx[i] = dy[i] * (cdf + v * pdf);
    }
}

// ------------------------------------------------------------- layernorm
// One block per row; D <= 1024. Saves mean and rstd. eps = 1e-5 (torch).
__global__ void pt_ln_fwd(float* y, float* mean, float* rstd, const float* x,
        const float* w, const float* b, int N, int D) {
    int row = blockIdx.x;
    if (row >= N) return;
    const float* xr = x + (long)row * D;
    float* yr = y + (long)row * D;
    __shared__ float sh[32];
    float s = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) s += xr[i];
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x < 32) {
        s = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o);
        if (threadIdx.x == 0) sh[0] = s / D;
    }
    __syncthreads();
    float mu = sh[0];
    float v = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float d = xr[i] - mu;
        v += d * d;
    }
    __syncthreads();
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = threadIdx.x < (blockDim.x + 31) / 32 ? sh[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        if (threadIdx.x == 0) sh[1] = rsqrtf(v / D + 1e-5f);
    }
    __syncthreads();
    float rs = sh[1];
    if (threadIdx.x == 0) { mean[row] = mu; rstd[row] = rs; }
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = (xr[i] - mu) * rs * w[i] + b[i];
}

// dX (written), dW/dB (atomicAdd across rows).
// PT_LN_ROWS rows per block, walked serially; per-row math and reduction
// order are identical to the historical one-row-per-block kernel. The dw/db
// contributions of the block's rows are accumulated in registers and merged
// with one atomicAdd per (thread, element) at the end: the cross-row merge
// order into dw/db is fp-reorder-nondeterministic in the original too (both
// are atomicAdd accumulations scheduled at block granularity), so this only
// reduces the atomic storm ~PT_LN_ROWS-fold; dx — the gradient that actually
// flows down the network — is bitwise identical.
#define PT_LN_ROWS 32
#define PT_LN_MAX_ELEMS 8      // ceil(PT_CARD_IN 634 / 128); guarded below
static inline int pt_ln_bwd_grid(long rows) {
    long g = (rows + PT_LN_ROWS - 1) / PT_LN_ROWS;
    return (int)(g < 1 ? 1 : g);
}
__global__ void pt_ln_bwd(float* dx, float* dw, float* db, const float* dy,
        const float* x, const float* w, const float* mean, const float* rstd,
        int N, int D) {
    long row0 = (long)blockIdx.x * PT_LN_ROWS;
    float wacc[PT_LN_MAX_ELEMS], bacc[PT_LN_MAX_ELEMS];
    int nmine = 0;
    if (D <= blockDim.x * PT_LN_MAX_ELEMS) {
        nmine = (D + blockDim.x - 1) / blockDim.x;
        for (int k = 0; k < nmine; k++) { wacc[k] = 0.f; bacc[k] = 0.f; }
    }
    __shared__ float sh[64];
    for (long row = row0; row < N && row < row0 + PT_LN_ROWS; row++) {
        const float* xr = x + row * D;
        const float* dyr = dy + row * D;
        float* dxr = dx + row * D;
        float mu = mean[row], rs = rstd[row];
        float s1 = 0.f, s2 = 0.f;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float xh = (xr[i] - mu) * rs;
            float g = dyr[i] * w[i];
            s1 += g;
            s2 += g * xh;
        }
        for (int o = 16; o > 0; o >>= 1) {
            s1 += __shfl_down_sync(0xffffffff, s1, o);
            s2 += __shfl_down_sync(0xffffffff, s2, o);
        }
        if ((threadIdx.x & 31) == 0) {
            sh[threadIdx.x >> 5] = s1;
            sh[32 + (threadIdx.x >> 5)] = s2;
        }
        __syncthreads();
        if (threadIdx.x < 32) {
            int nw = (blockDim.x + 31) / 32;
            s1 = threadIdx.x < nw ? sh[threadIdx.x] : 0.f;
            s2 = threadIdx.x < nw ? sh[32 + threadIdx.x] : 0.f;
            for (int o = 16; o > 0; o >>= 1) {
                s1 += __shfl_down_sync(0xffffffff, s1, o);
                s2 += __shfl_down_sync(0xffffffff, s2, o);
            }
            if (threadIdx.x == 0) { sh[0] = s1 / D; sh[32] = s2 / D; }
        }
        __syncthreads();
        float m1 = sh[0], m2 = sh[32];
        int k = 0;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            float xh = (xr[i] - mu) * rs;
            float g = dyr[i] * w[i];
            dxr[i] = (g - m1 - xh * m2) * rs;
            if (nmine > 0) {
                wacc[k] += dyr[i] * xh;
                bacc[k] += dyr[i];
                k++;
            } else {
                atomicAdd(&dw[i], dyr[i] * xh);
                atomicAdd(&db[i], dyr[i]);
            }
        }
    }
    for (int k = 0; k < nmine; k++) {
        int i = threadIdx.x + k * blockDim.x;
        if (i < D) {
            atomicAdd(&dw[i], wacc[k]);
            atomicAdd(&db[i], bacc[k]);
        }
    }
}

// ------------------------------------------------- masked softmax (attn)
// scores (B,H,L,L): out-of-mask KEYS get -inf before softmax. Saves probs
// in place. keep (B, L) float 0/1 over key positions.
//
// Grid mapping: PT_SM_GPT (128-thread) groups per block, row pairs strided
// across the grid. The per-row arithmetic and reduction order are IDENTICAL
// to the historical one-row-per-block kernel (same 128-thread i-loop over
// [tid, L, 128], same 16..1 shuffle tree, same expf / multiply sequence);
// only the row->block mapping changed, and the masked-scale intermediate is
// recomputed in registers instead of round-tripping through global memory
// (identical bits, one fewer full pass over the probs tensor).
#define PT_SM_GPT 2                     // rows (128-thread groups) per block
static inline int pt_softmax_grid(long rows) {
    long g = (rows + PT_SM_GPT - 1) / PT_SM_GPT;
    return (int)(g < 1 ? 1 : g);
}

__global__ void pt_attn_softmax_fwd(float* sc, const float* keep,
        int B, int H, int L, float scale) {
    // 128 threads per row, PT_SM_GPT rows per block; row pairs strided over
    // the grid. Shared layout per group: sh[gb+0..3] warp partials,
    // sh[gb+4] reduced scalar (gb = g*8; identical protocol to the historical
    // one-row-per-block kernel, which used sh[0..3] + sh[0]).
    int g = threadIdx.x >> 7;
    int t = threadIdx.x & 127;
    long total = (long)B * H * L;
    long pairs = (total + PT_SM_GPT - 1) / PT_SM_GPT;
    __shared__ float sh[8 * PT_SM_GPT];
    const int gb = g * 8;
    for (long p = blockIdx.x; p < pairs; p += gridDim.x) {
        long row = p * PT_SM_GPT + g;
        bool active = row < total;
        const int b = (int)(row / ((long)H * L));
        float* r = sc + row * L;
        const float* k = keep + (long)b * L;
        // The masked-scaled scores for i = t / t+128 live in registers across
        // all three passes (max / sum / normalize; L <= 256 in this model) —
        // same values the 3-pass version round-tripped through global memory,
        // same per-thread accumulation sequence (t, t+128, then t+256...).
        float v0 = -1e30f, v1 = -1e30f;
        bool h0 = false, h1 = false;
        float mx = -1e30f;
        if (active) {
            h0 = t < L;
            h1 = t + 128 < L;
            if (h0) { v0 = k[t] != 0.f ? r[t] * scale : -1e30f; mx = fmaxf(mx, v0); }
            if (h1) { v1 = k[t + 128] != 0.f ? r[t + 128] * scale : -1e30f;
                      mx = fmaxf(mx, v1); }
            for (int i = t + 256; i < L; i += 128)
                mx = fmaxf(mx, k[i] != 0.f ? r[i] * scale : -1e30f);
        }
        for (int o = 16; o > 0; o >>= 1)
            mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
        if (active && (t & 31) == 0) sh[gb + (t >> 5)] = mx;
        __syncthreads();
        if (t < 32) {
            mx = t < 4 ? sh[gb + t] : -1e30f;
            for (int o = 16; o > 0; o >>= 1)
                mx = fmaxf(mx, __shfl_down_sync(0xffffffff, mx, o));
            if (t == 0) sh[gb + 4] = mx;
        }
        __syncthreads();
        mx = sh[gb + 4];
        float sum = 0.f;
        if (active) {
            if (h0) sum += expf(v0 - mx);
            if (h1) sum += expf(v1 - mx);
            for (int i = t + 256; i < L; i += 128)
                sum += expf((k[i] != 0.f ? r[i] * scale : -1e30f) - mx);
        }
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, o);
        if (active && (t & 31) == 0) sh[gb + (t >> 5)] = sum;
        __syncthreads();
        if (t < 32) {
            sum = t < 4 ? sh[gb + t] : 0.f;
            for (int o = 16; o > 0; o >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, o);
            if (t == 0) sh[gb + 4] = sum;
        }
        __syncthreads();
        if (active) {
            float inv = 1.f / sh[gb + 4];
            if (h0) r[t] = expf(v0 - mx) * inv;
            if (h1) r[t + 128] = expf(v1 - mx) * inv;
            for (int i = t + 256; i < L; i += 128)
                r[i] = expf((k[i] != 0.f ? r[i] * scale : -1e30f) - mx) * inv;
        }
        __syncthreads();
    }
}

// softmax bwd in place: dP -> dS = P*(dP - sum(dP*P)); then unscale.
// Same grid mapping / shared protocol / per-row reduction order as fwd.
// The dot pass caches the (dp, p) register pairs for i = t and i = t+128
// (covers L <= 256; the model's L is LSEQ=161 / LCRIT=218) so the write pass
// does not re-read them — same bits, one fewer full read pass.
__global__ void pt_attn_softmax_bwd(float* dp, const float* p,
        int B, int H, int L, float scale) {
    int g = threadIdx.x >> 7;
    int t = threadIdx.x & 127;
    long total = (long)B * H * L;
    long pairs = (total + PT_SM_GPT - 1) / PT_SM_GPT;
    __shared__ float sh[8 * PT_SM_GPT];
    const int gb = g * 8;
    for (long pi = blockIdx.x; pi < pairs; pi += gridDim.x) {
        long row = pi * PT_SM_GPT + g;
        bool active = row < total;
        float* dpr = dp + row * L;
        const float* pr = p + row * L;
        float dot = 0.f;
        float d0 = 0.f, p0 = 0.f, d1 = 0.f, p1 = 0.f;
        bool h0 = false, h1 = false;
        if (active) {
            h0 = t < L;
            h1 = t + 128 < L;
            if (h0) { d0 = dpr[t]; p0 = pr[t]; dot += d0 * p0; }
            if (h1) { d1 = dpr[t + 128]; p1 = pr[t + 128]; dot += d1 * p1; }
            for (int i = t + 256; i < L; i += 128) dot += dpr[i] * pr[i];
        }
        for (int o = 16; o > 0; o >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, o);
        if (active && (t & 31) == 0) sh[gb + (t >> 5)] = dot;
        __syncthreads();
        if (t < 32) {
            dot = t < 4 ? sh[gb + t] : 0.f;
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_down_sync(0xffffffff, dot, o);
            if (t == 0) sh[gb + 4] = dot;
        }
        __syncthreads();
        if (active) {
            dot = sh[gb + 4];
            if (h0) dpr[t] = p0 * (d0 - dot) * scale;
            if (h1) dpr[t + 128] = p1 * (d1 - dot) * scale;
            for (int i = t + 256; i < L; i += 128)
                dpr[i] = pr[i] * (dpr[i] - dot) * scale;
        }
        __syncthreads();
    }
}

// (B,L,3D) qkv rows -> Q,K,V (B,H,L,Dh) each
__global__ void pt_qkv_split(float* q, float* k, float* v, const float* qkv,
        int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    int D = H * Dh;
    const float* src = qkv + ((long)b * L + l) * 3 * D;
    long dst = (((long)b * H + h) * L + l) * Dh + d;
    q[dst] = src[h * Dh + d];
    k[dst] = src[D + h * Dh + d];
    v[dst] = src[2 * D + h * Dh + d];
}

// inverse: dQ,dK,dV (B,H,L,Dh) -> d_qkv (B,L,3D)
__global__ void pt_qkv_merge(float* dqkv, const float* dq, const float* dk,
        const float* dv, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    int D = H * Dh;
    float* dst = dqkv + ((long)b * L + l) * 3 * D;
    long src = (((long)b * H + h) * L + l) * Dh + d;
    dst[h * Dh + d] = dq[src];
    dst[D + h * Dh + d] = dk[src];
    dst[2 * D + h * Dh + d] = dv[src];
}

// (B,H,L,Dh) -> (B,L,D) concat heads
__global__ void pt_heads_merge(float* out, const float* x, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    out[((long)b * L + l) * H * Dh + h * Dh + d] = x[i];
}

__global__ void pt_heads_split(float* out, const float* x, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int b = (int)(i / ((long)Dh * L * H));
    out[i] = x[((long)b * L + l) * H * Dh + h * Dh + d];
}

// ---------------------------------------------------------- misc small ops
// ------------------------------------- fused linear-tail elementwise chains
// All of these produce bitwise-identical results to the separate
// bias_add / gelu / split / residual kernels they replace: the per-element
// expression and its evaluation order are unchanged, only the pass structure.
// qkv raw gemm rows + bias -> Q,K,V (the fwd bias_add + pt_qkv_split pair)
__global__ void pt_qkv_split_bias(float* q, float* k, float* v,
        const float* qkv_raw, const float* b, int B, int L, int H, int Dh) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    long total = (long)B * L * H * Dh;
    if (i >= total) return;
    int d = (int)(i % Dh);
    int l = (int)((i / Dh) % L);
    int h = (int)((i / ((long)Dh * L)) % H);
    int bi = (int)(i / ((long)Dh * L * H));
    int D = H * Dh;
    long src = ((long)bi * L + l) * 3 * D;
    long dst = (((long)bi * H + h) * L + l) * Dh + d;
    q[dst] = qkv_raw[src + h * Dh + d] + b[h * Dh + d];
    k[dst] = qkv_raw[src + D + h * Dh + d] + b[D + h * Dh + d];
    v[dst] = qkv_raw[src + 2 * D + h * Dh + d] + b[2 * D + h * Dh + d];
}

// y = gelu(x + b)  (the fwd bias_add + pt_gelu_fwd pair on a linear output)
__global__ void pt_bias_gelu(float* y, const float* x, const float* b, long n,
        int o) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + b[i % o];
        y[i] = 0.5f * v * (1.f + erff(v * 0.70710678118654752f));
    }
}

// dY -> dX for y = gelu(x + b) given the saved RAW pre-bias gemm output
__global__ void pt_bias_gelu_bwd(float* dx, const float* dy, const float* x,
        const float* b, long n, int o) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] + b[i % o];
        float cdf = 0.5f * (1.f + erff(v * 0.70710678118654752f));
        float pdf = 0.3989422804014327f * expf(-0.5f * v * v);
        dx[i] = dy[i] * (cdf + v * pdf);
    }
}

// y = x + (g + b)  (the fwd bias_add + residual pt_add pair, same add order)
__global__ void pt_bias_res(float* y, const float* g, const float* b,
        const float* x, long n, int o) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + (g[i] + b[i % o]);
}

__global__ void pt_add(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void pt_zero(float* y, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = 0.f;
}

__global__ void pt_copy(float* y, const float* x, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i];
}

__global__ void pt_scale(float* y, float a, long n) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i < n) y[i] *= a;
}

// concat along last dim: copy src (N, D_src) into dst (N, D_dst) at col off
__global__ void pt_copy_block(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[(long)n * d_dst + off + c] = src[i];
}

// inverse: slice dst (N, d_src) out of src (N, d_dst) at col off
__global__ void pt_slice_block(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[i] = src[(long)n * d_dst + off + c];
}

// like pt_slice_block but ADDS into dst (grad accumulation)
__global__ void pt_slice_block_add(float* dst, const float* src, int N,
        int d_src, int d_dst, int off) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * d_src) return;
    int n = (int)(i / d_src), c = (int)(i % d_src);
    dst[i] += src[(long)n * d_dst + off + c];
}

// Embedding gather: ids read from a strided float column of the obs.
// out (N, E) = table[clamp(int(obs[n*stride + col]), 0, V-1)]
__global__ void pt_gather_obs(float* out, const float* table, const float* obs,
        long stride, long col, long col_stride, int N, int V, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    int id = (int)obs[n * stride + col + (n % 1) * 0 + col_stride * 0];
    (void)col_stride;
    id = id < 0 ? 0 : (id >= V ? V - 1 : id);
    out[i] = table[(long)id * E + e];
}

// scatter-add backward for a gather with per-row ids
__global__ void pt_scatter_obs(float* dtable, const float* dout, const float* obs,
        long stride, long col, int N, int V, int E) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * E) return;
    int n = (int)(i / E), e = (int)(i % E);
    int id = (int)obs[n * stride + col];
    id = id < 0 ? 0 : (id >= V ? V - 1 : id);
    atomicAdd(&dtable[(long)id * E + e], dout[i]);
}

#endif // PTCG_KERNELS_CUH
