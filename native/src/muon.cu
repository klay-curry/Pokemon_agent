#include <nccl.h>

__global__ void muon_norm_reduce(float* __restrict__ out, const float* __restrict__ partials, int num_blocks) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = (tid < num_blocks) ? partials[tid] : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out = sdata[0];
    }
}

__global__ void muon_norm_partials(float* __restrict__ partials, const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(src[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partials[blockIdx.x] = sdata[0];
    }
}

__global__ void muon_norm_apply(precision_t* __restrict__ dst, const float* __restrict__ norm_ptr, float eps, int n) {
    float inv_norm = 1.0f / fmaxf(sqrtf(*norm_ptr), eps);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * inv_norm);
    }
}

// Nesterov with f32 momentum accumulator and precision_t gradients
// LOCAL PATCH (deviation from vendored @ c5d3c63): gate pointer -- when the
// trainer's KL guard writes 0 there, the whole step (momentum included) is a
// no-op. Must be a branch, not a multiply: the gate exists precisely for
// minibatches whose gradients may be garbage.
__global__ void muon_nesterov(float* __restrict__ mb, precision_t* __restrict__ gc,
        const float* __restrict__ gate, float mu, int n) {
    if (gate != nullptr && *gate == 0.0f) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float m = mu * mb[idx] + to_float(gc[idx]);
        mb[idx] = m;
        gc[idx] = from_float(to_float(gc[idx]) + mu * m);
    }
}

// LOCAL PATCH (deviation from vendored @ c5d3c63): EMA momentum for the 1-D
// parameter path -- root-cause fix for the divergence mode described in
// docs/training.md.
//
// Upstream applies the classic SUM momentum above to every parameter, then
// orthogonalizes only ndim>=2 tensors. A 1-D param therefore takes the raw
// momentum step: ~1/(1-mu) = 20x the mean gradient at steady state, step size
// proportional to gradient magnitude rather than lr-bounded, and with wd=0
// nothing damps it -- LN gains and the logit readout enter a sharpening
// feedback loop (bigger gain -> sharper logits -> bigger gradient) and blow
// up. The torch control runs use heavyball's nesterov_ema
// (m = mu*m + (1-mu)*g; update = g + m) and are measurably stable: LN gains
// pinned at 1.02 where this path drifted 1.46 -> 9.22 -> NaN. Apply that
// exact rule to the params Muon does not orthogonalize; the >=2-D path stays
// untouched upstream math.
__global__ void muon_nesterov_ema(float* __restrict__ mb, precision_t* __restrict__ gc,
        const float* __restrict__ gate, float mu, int n) {
    if (gate != nullptr && *gate == 0.0f) return;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gc[idx]);
        float m = mu * mb[idx] + (1.0f - mu) * g;
        mb[idx] = m;
        gc[idx] = from_float(g + m);
    }
}

// Fused weight update: wb = wb * (1 - lr*wd) - lr * scale * update
// LOCAL PATCH: gate pointer -- a vetoed step must return without touching wb.
// A multiplicative gate would still poison weights on non-finite updates
// (0 * nan = nan), which is exactly when the guard fires.
__global__ void muon_weight_update(float* __restrict__ wb, const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, const float* __restrict__ gate,
        float wd, float scale, int n) {
    if (gate != nullptr && *gate == 0.0f) return;
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        wb[idx] = wb[idx] * wd_scale - lr * scale * to_float(update[idx]);
    }
}

__global__ void muon_clip_norm(precision_t* __restrict__ dst,
        const float* __restrict__ sum_sq_ptr, float max_norm, float eps, int n) {
    float clip_coef = fminf(max_norm / (sqrtf(*sum_sq_ptr) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = from_float(to_float(dst[idx]) * clip_coef);
    }
}

// V21 optimizer split (docs/training.md): AdamW path for the tagged params
// (input embeddings, loss output heads, all 1-D). Standard decoupled AdamW
// with bias correction, mirroring torch.optim.AdamW: w <- w*(1-lr*wd) -
// lr * mhat / (sqrt(vhat) + eps). The step counter lives on device as a long
// (float spacing breaks bias correction past 2^24 steps) and only advances
// when the gate allows the step, so vetoed minibatches leave m/v/t untouched
// -- the same branch-not-multiply contract as the Muon kernels.
__global__ void adamw_prestep(long* __restrict__ t_io, float* __restrict__ inv_io,
        const float* __restrict__ gate, double b1, double b2) {
    if (gate != nullptr && *gate == 0.0f) return;
    long t = t_io[0] + 1;
    t_io[0] = t;
    double td = (double)t;
    inv_io[0] = (float)(1.0 / (1.0 - pow(b1, td)));
    inv_io[1] = (float)(1.0 / (1.0 - pow(b2, td)));
}

__global__ void adamw_step(float* __restrict__ wb, const precision_t* __restrict__ gc,
        float* __restrict__ m, float* __restrict__ v,
        const float* __restrict__ lr_ptr, const float* __restrict__ inv_io,
        const float* __restrict__ gate, float b1, float b2, float wd, float eps,
        int n) {
    if (gate != nullptr && *gate == 0.0f) return;
    float lr = *lr_ptr;
    float inv1 = inv_io[0], inv2 = inv_io[1];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = to_float(gc[idx]);
        float m_new = b1 * m[idx] + (1.0f - b1) * g;
        float v_new = b2 * v[idx] + (1.0f - b2) * g * g;
        m[idx] = m_new;
        v[idx] = v_new;
        float update = (m_new * inv1) / (sqrtf(v_new * inv2) + eps);
        wb[idx] = wb[idx] * (1.0f - lr * wd) - lr * update;
    }
}

static constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

struct Muon {
    double momentum, weight_decay, eps;
    // LOCAL PATCH: decoupled weight decay applied ONLY to 1-D params (the
    // un-orthogonalized path); weight_decay above stays upstream's global
    // knob, still passed 0.0 at the call site so >=2-D behaviour is
    // unchanged. gate_ptr is the trainer's KL-guard veto (1.0 = apply step,
    // 0.0 = skip); nullptr disables.
    double wd_1d;
    const float* gate_ptr;
    float lr_val_init;
    float* lr_ptr;
    float* lr_derived_ptr;
    float* norm_ptr;
    float* grad_norm_ptr;
    FloatTensor lr_puf, lr_derived_puf, ns_norm_puf, grad_norm_puf;
    FloatTensor mb_puf;
    PrecisionTensor gram, gram_buf, x_buf;
    FloatTensor norm_partials;
    long max_M, max_N;
    Allocator* param_alloc;  // params allocator — shapes used by muon_step
    ncclComm_t nccl_comm;
    int world_size;
    // V21 optimizer split: when optimizer_split is 0 the walk below is the
    // legacy ndim-based path bit-for-bit. When 1, params tagged use_adamw in
    // the params allocator take the AdamW path with their own m/v state and
    // lr; everything else is unchanged. State buffers mirror the params
    // ARENA layout (registered only by muon_enable_adamw, so the default
    // path allocates nothing new). t counts steps on device (long); inv
    // holds the bias corrections for the current step.
    int optimizer_split;
    float adamw_beta1, adamw_beta2, adamw_wd, adamw_eps, adamw_lr_val_init;
    float* adamw_lr_ptr;
    LongTensor adamw_t_puf;
    FloatTensor adamw_inv_puf, adamw_m_puf, adamw_v_puf, adamw_lr_puf;
};

// Called between muon_init and alloc_create on the SAME allocator (the state
// tensors join that arena). Nothing here runs unless the trainer turns the
// split on.
void muon_enable_adamw(Muon* m, Allocator* alloc, double lr, double beta1,
        double beta2, double wd) {
    m->optimizer_split = 1;
    m->adamw_beta1 = (float)beta1;
    m->adamw_beta2 = (float)beta2;
    m->adamw_wd = (float)wd;
    m->adamw_eps = 1e-8f;
    m->adamw_lr_val_init = (float)lr;
    m->adamw_lr_ptr = nullptr;
    m->adamw_t_puf = {.shape = {2}};
    m->adamw_inv_puf = {.shape = {2}};
    m->adamw_lr_puf = {.shape = {1}};
    long n = m->param_alloc->total_bytes / (long)sizeof(float);
    m->adamw_m_puf = {.shape = {n}};
    m->adamw_v_puf = {.shape = {n}};
    alloc_register(alloc, &m->adamw_t_puf);
    alloc_register(alloc, &m->adamw_inv_puf);
    alloc_register(alloc, &m->adamw_lr_puf);
    alloc_register(alloc, &m->adamw_m_puf);
    alloc_register(alloc, &m->adamw_v_puf);
    // C-side accounting of the tag mask, so the split's scope is visible at
    // startup without trusting the python mirror (they are independent
    // tables and must agree; a mismatch here is the loud failure mode).
    long n_adamw = 0, n_total = 0, e_adamw = 0, e_total = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        long ne = numel(e.shape);
        n_total++;
        e_total += ne;
        if (e.use_adamw) { n_adamw++; e_adamw += ne; }
    }
    fprintf(stderr, "adamw split: %ld/%ld params, %ld/%ld elements on AdamW "
        "(lr %g b1 %g b2 %g wd %g)\n", n_adamw, n_total, e_adamw, e_total,
        lr, beta1, beta2, wd);
}

void muon_init(Muon* m, Allocator* param_alloc, double lr_val,
        double momentum, double eps, double weight_decay, double wd_1d,
        Allocator* alloc) {
    m->momentum = momentum;
    m->weight_decay = weight_decay;
    m->wd_1d = wd_1d;
    m->gate_ptr = nullptr;
    m->eps = eps;
    m->lr_val_init = (float)lr_val;
    m->lr_ptr = nullptr;
    m->lr_derived_ptr = nullptr;
    m->param_alloc = param_alloc;
    m->nccl_comm = nullptr;
    m->world_size = 1;
    m->optimizer_split = 0;
    m->adamw_beta1 = 0.9f; m->adamw_beta2 = 0.95f;
    m->adamw_wd = 0.01f; m->adamw_eps = 1e-8f;
    m->adamw_lr_val_init = 0.0f;
    m->adamw_lr_ptr = nullptr;
    m->max_M = 0; m->max_N = 0;
    // Momentum mirrors the params ARENA layout (incl. 16B alignment gaps),
    // not the packed element count — muon indexes grads/momentum by arena
    // offset. Gap floats stay zero (arena memset; backward never writes them).
    long n = param_alloc->total_bytes / (long)sizeof(float);
    m->lr_puf =         {.shape = {1}};
    m->lr_derived_puf = {.shape = {2}};
    m->mb_puf =         {.shape = {n}};
    m->norm_partials =  {.shape = {256}};
    m->grad_norm_puf =  {.shape = {1}};
    alloc_register(alloc, &m->lr_puf);
    alloc_register(alloc, &m->lr_derived_puf);
    alloc_register(alloc, &m->mb_puf);
    alloc_register(alloc, &m->norm_partials);
    alloc_register(alloc, &m->grad_norm_puf);
    long max_M = 0, max_N = 0;
    for (int _i = 0; _i < param_alloc->num_regs; _i++) {
        AllocEntry& e = param_alloc->regs[_i];
        if (ndim(e.shape) >= 2) {
            long R = e.shape[0], C = numel(e.shape) / R;
            max_M = max(max_M, min(R, C));
            max_N = max(max_N, max(R, C));
        }
    }
    if (max_M > 0) {
        m->max_M = max_M; m->max_N = max_N;
        m->gram =        {.shape = {max_M, max_M}};
        m->gram_buf =    {.shape = {max_M, max_M}};
        m->x_buf =       {.shape = {max_M, max_N}};
        m->ns_norm_puf = {.shape = {1}};
        alloc_register(alloc, &m->gram);
        alloc_register(alloc, &m->gram_buf);
        alloc_register(alloc, &m->x_buf);
        alloc_register(alloc, &m->ns_norm_puf);
    }
}

void muon_post_create(Muon* m) {
    m->lr_ptr = m->lr_puf.data;
    m->lr_derived_ptr = m->lr_derived_puf.data;
    m->grad_norm_ptr = m->grad_norm_puf.data;
    if (m->ns_norm_puf.data) m->norm_ptr = m->ns_norm_puf.data;
    cudaMemcpy(m->lr_ptr, &m->lr_val_init, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(m->lr_derived_ptr, 0, 2 * sizeof(float));
    cudaMemset(m->mb_puf.data, 0, numel(m->mb_puf.shape) * sizeof(float));
    if (m->optimizer_split) {
        m->adamw_lr_ptr = m->adamw_lr_puf.data;
        cudaMemcpy(m->adamw_lr_ptr, &m->adamw_lr_val_init, sizeof(float),
                   cudaMemcpyHostToDevice);
        // m/v/t start at zero: a warm-start from a pre-split checkpoint has
        // no AdamW state, and a resumed PTCGSTA2 file overwrites these.
        cudaMemset(m->adamw_t_puf.data, 0, numel(m->adamw_t_puf.shape) * sizeof(long));
        cudaMemset(m->adamw_inv_puf.data, 0, numel(m->adamw_inv_puf.shape) * sizeof(float));
        cudaMemset(m->adamw_m_puf.data, 0, numel(m->adamw_m_puf.shape) * sizeof(float));
        cudaMemset(m->adamw_v_puf.data, 0, numel(m->adamw_v_puf.shape) * sizeof(float));
    }
}

void muon_step(Muon* m, FloatTensor weights, PrecisionTensor grads, float max_grad_norm, cudaStream_t stream = 0) {
    // Head-only mode (PTCG_TRAIN_MODE=headonly): frozen params contribute no
    // gradient and must not dilute the global clip. Zero their arena slots
    // before the allreduce/clip; the param walk below skips them entirely
    // (no momentum, no weight decay), so the frozen weights are bit-identical
    // to the warm-start checkpoint.
    {
        long zoff = 0;
        for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
            AllocEntry& e = m->param_alloc->regs[_i];
            zoff = (zoff + 3) & ~3L;
            long ne = numel(e.shape);
            if (!e.trainable)
                cudaMemsetAsync(grads.data + zoff, 0,
                    ne * sizeof(precision_t), stream);
            zoff += ne;
        }
    }

    // Multi-GPU support: simple all-reduce over a contiguous grad buffer
    if (m->nccl_comm != nullptr && m->world_size > 1) {
        ncclAllReduce(grads.data, grads.data, numel(grads.shape),
            NCCL_PRECISION, ncclAvg, m->nccl_comm, stream);
    }

    // Clip gradients by norm
    int clip_blocks = min((int)grid_size(numel(grads.shape)), 256);
    muon_norm_partials<<<clip_blocks, 256, 0, stream>>>(
        m->norm_partials.data, grads.data, numel(grads.shape));
    muon_norm_reduce<<<1, 256, 0, stream>>>(m->grad_norm_ptr, m->norm_partials.data, clip_blocks);
    muon_clip_norm<<<grid_size(numel(grads.shape)), BLOCK_SIZE, 0, stream>>>(
        grads.data, m->grad_norm_ptr, max_grad_norm, 1e-6f, numel(grads.shape));

    // V21 split: advance the AdamW step counter once per optimizer step.
    // Runs BEFORE the param walk so every adamw_step below reads this step's
    // bias corrections; the gate branch inside keeps a vetoed step from
    // touching t, matching the momentum kernels' no-op contract.
    if (m->optimizer_split) {
        adamw_prestep<<<1, 1, 0, stream>>>(m->adamw_t_puf.data,
            m->adamw_inv_puf.data, m->gate_ptr,
            (double)m->adamw_beta1, (double)m->adamw_beta2);
    }

    // LOCAL PATCH: the momentum kernel moved from one arena-wide launch into
    // the per-param walk below so the 1-D path can use EMA accumulation while
    // the >=2-D path keeps the classic rule. Both kernels are elementwise on
    // disjoint regions, so the >=2-D math is bit-identical to the upstream
    // arena-wide launch; alignment-gap floats (no-op'd through the kernel
    // upstream) are simply skipped and stay zero.

    long offset = 0;
    for (int _i = 0; _i < m->param_alloc->num_regs; _i++) {
        AllocEntry& e = m->param_alloc->regs[_i];
        // mirror alloc_create's 16B alignment (4 floats) — params/grads/
        // momentum all live in identically-laid-out aligned arenas
        static_assert(sizeof(precision_t) == sizeof(float),
            "aligned arena walk shares one element offset across the fp32 "
            "master and precision_t grads — only valid when they match");
        offset = (offset + 3) & ~3L;
        long ne = numel(e.shape);
        if (!e.trainable) {          // head-only: frozen param, no update
            offset += ne;
            continue;
        }
        precision_t* gc_ptr = grads.data + offset;
        float* wb_ptr = weights.data + offset;
        float* mb_ptr = m->mb_puf.data + offset;
        const precision_t* update_ptr = gc_ptr;
        float scale = 1.0f;
        float wd = (float)m->weight_decay;

        // Orthogonalize the update
        if (m->optimizer_split && e.use_adamw) {
            // V21 split: tagged params (embeddings, loss heads, 1-D) take
            // decoupled AdamW. The grads were already arena-wide allreduced
            // and clipped above, so DDP and the KL guard keep their exact
            // semantics; m/v state is rank-local like Muon momentum. The
            // kernel applies the weight update itself, so the shared
            // muon_weight_update below must NOT run for this param.
            adamw_step<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                wb_ptr, gc_ptr, m->adamw_m_puf.data + offset,
                m->adamw_v_puf.data + offset, m->adamw_lr_ptr,
                m->adamw_inv_puf.data, m->gate_ptr, m->adamw_beta1,
                m->adamw_beta2, m->adamw_wd, m->adamw_eps, (int)ne);
            offset += ne;
            continue;
        } else if (ndim(e.shape) >= 2) {
            muon_nesterov<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                mb_ptr, gc_ptr, m->gate_ptr, (float)m->momentum, (int)ne);
            long R = e.shape[0], C = ne / R;
            long M = min(R, C), N = max(R, C);
            bool tall = R > C;
            PrecisionTensor x = {.data = gc_ptr, .shape = {R, C}};
            PrecisionTensor x_buf = {.data = m->x_buf.data, .shape = {R, C}};
            PrecisionTensor gram = {.data = m->gram.data, .shape = {M, M}};
            PrecisionTensor gram_buf = {.data = m->gram_buf.data, .shape = {M, M}};

            int nblk = min((int)grid_size(numel(x.shape)), 256);
            muon_norm_partials<<<nblk, 256, 0, stream>>>(
                m->norm_partials.data, x.data, numel(x.shape));
            muon_norm_reduce<<<1, 256, 0, stream>>>(m->norm_ptr, m->norm_partials.data, nblk);
            muon_norm_apply<<<grid_size(numel(x.shape)), BLOCK_SIZE, 0, stream>>>(
                x.data, m->norm_ptr, 1e-7f, numel(x.shape));

            cublasOperation_t gram_op_a = tall ? CUBLAS_OP_T : CUBLAS_OP_N;
            cublasOperation_t gram_op_b = tall ? CUBLAS_OP_N : CUBLAS_OP_T;
            for (int i = 0; i < 5; ++i) {
                PrecisionTensor& src = (i % 2 == 0) ? x : x_buf;
                PrecisionTensor& dst = (i % 2 == 0) ? x_buf : x;
                cublasGemmExDense(gram_op_a, gram_op_b, (int)M, (int)M, (int)N,
                    src.data, src.data, gram.data, stream);
                puf_copy(&gram_buf, &gram, stream);
                puf_addmm_nn(&gram, &gram, &gram_buf, ns_coeffs[i][2], ns_coeffs[i][1], stream);
                puf_copy(&dst, &src, stream);
                cublasGemmExDense(CUBLAS_OP_N, CUBLAS_OP_N, (int)R, (int)C, (int)M,
                    tall ? src.data : gram_buf.data, tall ? gram_buf.data : src.data, dst.data,
                    stream, 1.0f, ns_coeffs[i][0]);
            }

            update_ptr = x_buf.data;
            scale = sqrtf(fmaxf(1.0f, (float)R / (float)C));
        } else {
            // LOCAL PATCH: 1-D params — EMA momentum (see muon_nesterov_ema)
            // and their own decoupled weight decay. The classic-sum update at
            // wd 0 is the diagnosed divergence path.
            muon_nesterov_ema<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
                mb_ptr, gc_ptr, m->gate_ptr, (float)m->momentum, (int)ne);
            wd = (float)m->wd_1d;
        }

        muon_weight_update<<<grid_size(ne), BLOCK_SIZE, 0, stream>>>(
            wb_ptr, update_ptr, m->lr_ptr, m->gate_ptr, wd, scale, (int)ne);
        offset += ne;
    }
}
