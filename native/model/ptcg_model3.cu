// Part 3: transformer blocks, drivers, encoder/decoder/network vtables.

struct PtcgActFull;   // fwd decl

// one pre-LN block forward. x_in/x_out (N,L,D) distinct buffers.
// scratch: a->s_ln (N,L,D), a->s_qkv (N,L,3D), s_q/s_k/s_v/s_ctx (N,H,L,Dh),
// s_ffn (N,L,FFN)
struct BlkScratch {
    float *ln, *qkv, *q, *k, *v, *ctx, *ffn, *dffn, *dq, *dk, *dv, *dp;
};

static void blk_fwd(BlkW* bw, BlkAct* ba, BlkScratch* sc, const float* x_in,
        float* x_out, const float* keep, int N, int L, cudaStream_t s,
        PtProfSite* prof = nullptr) {
    long NL = (long)N * L;
    float scale = 1.0f / sqrtf((float)PT_DH);
    #define PTPB(i) do { if (prof) cudaEventRecord(prof->e[i], s); } while (0)
    PTPB(0);
    // The train-path ln1/qkv outputs are written straight into their saved
    // tensors (rollout keeps the scratch path) so the backward can consume
    // them with zero copy traffic — the values are the same gemm/ln results
    // the backward used to replay bitwise.
    float* ln1out = ba->ln1_save.data ? ba->ln1_save.data : sc->ln;
    float* qkvout = ba->qkv_save.data ? ba->qkv_save.data : sc->qkv;
    pt_ln_fwd<<<(int)NL, 128, 0, s>>>(ln1out, ba->ln1_mean.data, ba->ln1_rstd.data,
        x_in, bw->ln1w, bw->ln1b, (int)NL, PT_D);
    // gemm writes the raw product; bias is folded into the split (bitwise
    // same per-element adds as the separate bias_add + pt_qkv_split passes)
    pt_linear(ln1out, bw->qkvw, nullptr, qkvout, (int)NL, PT_D, 3 * PT_D, s);
    pt_qkv_split_bias<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->q, sc->k, sc->v,
        qkvout, bw->qkvb, N, L, PT_H, PT_DH);
    PTPB(1);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T, L, L, PT_DH,
        sc->q, (long)L * PT_DH, sc->k, (long)L * PT_DH,
        ba->probs.data, (long)L * L, N * PT_H, s);
    PTPB(2);
    pt_attn_softmax_fwd<<<pt_softmax_grid((long)N * PT_H * L), PT_SM_GPT * 128, 0, s>>>(
        ba->probs.data, keep, N, PT_H, L, scale);
    PTPB(3);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, L, PT_DH, L,
        ba->probs.data, (long)L * L, sc->v, (long)L * PT_DH,
        sc->ctx, (long)L * PT_DH, N * PT_H, s);
    PTPB(4);
    pt_heads_merge<<<ptg(NL * PT_D), PTB, 0, s>>>(ba->merged.data, sc->ctx,
        N, L, PT_H, PT_DH);
    pt_linear(ba->merged.data, bw->projw, nullptr, ba->x_mid.data,
        (int)NL, PT_D, PT_D, s);
    pt_bias_res<<<ptg(NL * PT_D), PTB, 0, s>>>(ba->x_mid.data, ba->x_mid.data,
        bw->projb, x_in, NL * PT_D, PT_D);
    PTPB(5);
    float* ln2out = ba->ln2_save.data ? ba->ln2_save.data : sc->ln;
    pt_ln_fwd<<<(int)NL, 128, 0, s>>>(ln2out, ba->ln2_mean.data, ba->ln2_rstd.data,
        ba->x_mid.data, bw->ln2w, bw->ln2b, (int)NL, PT_D);
    // ffn1 gemm raw + fused bias->gelu straight into the saved activation the
    // backward consumes (ffn1 keeps the raw pre-activation, as before)
    float* ffnout = ba->ffn_save.data ? ba->ffn_save.data : sc->ffn;
    pt_linear(ln2out, bw->f1w, nullptr, ba->ffn1.data, (int)NL, PT_D, PT_FFN, s);
    pt_bias_gelu<<<ptg(NL * PT_FFN), PTB, 0, s>>>(ffnout, ba->ffn1.data,
        bw->f1b, NL * PT_FFN, PT_FFN);
    PTPB(6);
    pt_linear(ffnout, bw->f2w, nullptr, x_out, (int)NL, PT_FFN, PT_D, s);
    pt_bias_res<<<ptg(NL * PT_D), PTB, 0, s>>>(x_out, x_out, bw->f2b,
        ba->x_mid.data, NL * PT_D, PT_D);
    PTPB(7);
    if (prof) pt_prof_finish(prof, s);
    #undef PTPB
}

// block backward. dxA holds d(x_out) on entry and d(x_in) on exit.
// dxB is same-size scratch.
static void blk_bwd(BlkW* bw, BlkG* bg, BlkAct* ba, BlkScratch* sc,
        const float* x_in, float* dxA, float* dxB, int N, int L, cudaStream_t s,
        PtProfSite* prof = nullptr) {
    long NL = (long)N * L;
    float scale = 1.0f / sqrtf((float)PT_DH);
    #define PTPB(i) do { if (prof) cudaEventRecord(prof->e[i], s); } while (0)
    PTPB(0);
    // ffn chain input comes from the saved gelu output (bitwise identical to
    // the gelu(ffn1)+bias replay it replaces)
    PTPB(1);
    pt_linear_bwd(ba->ffn_save.data, bw->f2w, dxA, sc->dffn, bg->f2w, bg->f2b,
        (int)NL, PT_FFN, PT_D, s);
    PTPB(2);
    pt_bias_gelu_bwd<<<ptg(NL * PT_FFN), PTB, 0, s>>>(sc->dffn, sc->dffn,
        ba->ffn1.data, bw->f1b, NL * PT_FFN, PT_FFN);
    pt_linear_bwd(ba->ln2_save.data, bw->f1w, sc->dffn, dxB, bg->f1w, bg->f1b,
        (int)NL, PT_D, PT_FFN, s);
    PTPB(3);
    pt_ln_bwd<<<pt_ln_bwd_grid(NL), 128, 0, s>>>(dxB, bg->ln2w, bg->ln2b, dxB,
        ba->x_mid.data, bw->ln2w, ba->ln2_mean.data, ba->ln2_rstd.data,
        (int)NL, PT_D);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(dxA, dxB, NL * PT_D);  // dxA = d_x_mid
    // attn chain
    pt_linear_bwd(ba->merged.data, bw->projw, dxA, sc->ln /*d_merged*/,
        bg->projw, bg->projb, (int)NL, PT_D, PT_D, s);
    pt_heads_split<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->ctx /*d_ctx*/, sc->ln,
        N, L, PT_H, PT_DH);
    PTPB(4);
    // q/k/v come from the saved qkv (bitwise identical to the replay of
    // ln1_apply + qkv GEMM they replace); ln1_out for the dW gemm likewise.
    pt_qkv_split<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->q, sc->k, sc->v,
        ba->qkv_save.data, N, L, PT_H, PT_DH);
    PTPB(5);
    // dV = P^T dctx ; dP = dctx V^T ; softmax bwd ; dK = dS^T Q ; dQ = dS K
    pt_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N, L, PT_DH, L,
        ba->probs.data, (long)L * L, sc->ctx, (long)L * PT_DH,
        sc->dv, (long)L * PT_DH, N * PT_H, s);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_T, L, L, PT_DH,
        sc->ctx, (long)L * PT_DH, sc->v, (long)L * PT_DH,
        sc->dp, (long)L * L, N * PT_H, s);
    PTPB(6);
    pt_attn_softmax_bwd<<<pt_softmax_grid((long)N * PT_H * L), PT_SM_GPT * 128, 0, s>>>(
        sc->dp, ba->probs.data, N, PT_H, L, scale);
    PTPB(7);
    pt_gemm_batched(CUBLAS_OP_T, CUBLAS_OP_N, L, PT_DH, L,
        sc->dp, (long)L * L, sc->q, (long)L * PT_DH,
        sc->dk, (long)L * PT_DH, N * PT_H, s);
    pt_gemm_batched(CUBLAS_OP_N, CUBLAS_OP_N, L, PT_DH, L,
        sc->dp, (long)L * L, sc->k, (long)L * PT_DH,
        sc->dq, (long)L * PT_DH, N * PT_H, s);
    PTPB(8);
    pt_qkv_merge<<<ptg(NL * PT_D), PTB, 0, s>>>(sc->qkv, sc->dq, sc->dk, sc->dv,
        N, L, PT_H, PT_DH);
    pt_linear_bwd(ba->ln1_save.data, bw->qkvw, ba->qkv_save.data, dxB,
        bg->qkvw, bg->qkvb, (int)NL, PT_D, 3 * PT_D, s);
    PTPB(9);
    pt_ln_bwd<<<pt_ln_bwd_grid(NL), 128, 0, s>>>(dxB, bg->ln1w, bg->ln1b, dxB,
        x_in, bw->ln1w, ba->ln1_mean.data, ba->ln1_rstd.data, (int)NL, PT_D);
    pt_add<<<ptg(NL * PT_D), PTB, 0, s>>>(dxA, dxB, NL * PT_D);  // dxA = d_x_in
    PTPB(10);
    if (prof) pt_prof_finish(prof, s);
    #undef PTPB
}

// full activation struct registration (shared by rollout/train, train adds bwd)
struct PtcgActX {
    PtcgAct a;
    BlkScratch sc;
    PrecisionTensor s_ln, s_qkv, s_q, s_k, s_v, s_ctx, s_ffn, s_dffn,
                    s_dq, s_dk, s_dv, s_dp;
    PrecisionTensor tokp;      // (N,160,D) token_proj output
    PrecisionTensor d_tokp, d_tokcat, d_bc, d_bcin, d_decin, d_gin;
    PrecisionTensor g1a;       // global pre-gelu (N,128)
    PrecisionTensor d_otok_in, d_q, d_qin, d_scin, d_out64, d_ocard,
                    d_oatt, d_val, d_vin, d_v1;
    PrecisionTensor d_handlin, d_prizelogit;
    PrecisionTensor ov_out;    // oracle value (N,1)
};

static int chain_on(void);

static void ptcg_reg(PtcgActX* ax, PtcgWeights* w, Allocator* acts,
        Allocator* grads, int N, int train) {
    PtcgAct* a = &ax->a;
    a->N = N;
    a->train = train;
    if (train) {   // grads registered in canonical order == params order
        for (int i = 0; i < PT_NPARAM; i++) {
            w->g[i].shape[0] = PT_PARAMS[i].r;
            w->g[i].shape[1] = PT_PARAMS[i].c > 0 ? PT_PARAMS[i].c : 0;
            alloc_register(grads, &w->g[i]);
        }
    }
    table_reg(&a->tab, acts, grads, train);
    pt_reg(acts, &a->tok_cat, (long)N * PT_LTOK, PT_TOK_IN);
    pt_reg(acts, &ax->tokp, (long)N * PT_LTOK, PT_D);
    pt_reg(acts, &ax->g1a, N, 128);
    pt_reg(acts, &a->g_out, N, 128);
    pt_reg(acts, &a->dec_in, N, PT_DEC_IN);
    pt_reg(acts, &a->d1, N, 64);
    pt_reg(acts, &a->d_out, N, 64);
    pt_reg(acts, &a->bc_in, N, PT_BCAST_IN);
    pt_reg(acts, &a->bc_out, N, PT_D);
    // rollout has no backward: one shared trunk BlkAct + ping-pong x buffers
    int n_x = train ? PT_LAYERS + 1 : 2;
    int n_blk = train ? PT_LAYERS : 1;
    for (int i = 0; i < n_x; i++)
        pt_reg(acts, &a->x[i], (long)N * PT_LSEQ, PT_D);
    for (int i = 0; i < n_blk; i++)
        blk_reg(&a->blk[i], acts, N, PT_LSEQ, train);
    pt_reg(acts, &a->keep, N, PT_LSEQ);
    pt_reg(acts, &a->h, (long)N * PT_LSEQ, PT_D);
    pt_reg(acts, &a->lnf_mean, (long)N * PT_LSEQ, 0);
    pt_reg(acts, &a->lnf_rstd, (long)N * PT_LSEQ, 0);
    pt_reg(acts, &a->q_in, (long)N * PT_NOPT, PT_Q_IN);
    pt_reg(acts, &a->q1a, (long)N * PT_NOPT, PT_D);
    pt_reg(acts, &a->q_vec, (long)N * PT_NOPT, PT_D);
    pt_reg(acts, &a->sc_in, (long)N * PT_NOPT, PT_SCORE_IN);
    pt_reg(acts, &a->sc1, (long)N * PT_NOPT, 256);
    pt_reg(acts, &a->scores, (long)N * PT_NOPT, 0);
    pt_reg(acts, &a->ocard_in, (long)N * PT_NOPT, PT_DC);
    pt_reg(acts, &a->oatt_in, (long)N * PT_NOPT, PT_DA);
    pt_reg(acts, &a->otok_in, (long)N * PT_NORC, PT_ORC_IN);
    pt_reg(acts, &a->otok, (long)N * PT_NORC, PT_D);
    pt_reg(acts, &a->xc[0], (long)N * PT_LCRIT, PT_D);
    pt_reg(acts, &a->xc[1], (long)N * PT_LCRIT, PT_D);
    blk_reg(&a->cblk, acts, N, PT_LCRIT, train);
    pt_reg(acts, &a->ckeep, N, PT_LCRIT);
    pt_reg(acts, &a->ov_in, N, PT_VAL_IN);
    pt_reg(acts, &a->ov1, N, 256);
    pt_reg(acts, &ax->ov_out, N, 1);
#if PT_HL
    pt_reg(acts, &a->ov_bins, N, PT_VBINS);
    if (train) pt_reg(acts, &a->hl_dlogit, N, PT_VBINS);
#endif
    pt_reg(acts, &a->out, N, PT_NOPT + 1);
    // MoE routing scratch (forward for all modes; bwd staging train-only)
    pt_reg(acts, &a->moe_fam, N, 0);
    pt_reg(acts, &a->moe_s, (long)N * PT_NOPT, 0);
    pt_reg(acts, &a->moe_s_e, (long)N * PT_NOPT, 0);
    pt_reg(acts, &a->moe_ov_e, N, 0);
    if (chain_on()) {
        pt_reg(acts, &a->moe_oc64, (long)N * PT_NOPT, 64);
        pt_reg(acts, &a->moe_oa64, (long)N * PT_NOPT, 64);
        pt_reg(acts, &a->moe_qin, (long)N * PT_NOPT, PT_Q_IN);
        pt_reg(acts, &a->moe_q1a, (long)N * PT_NOPT, PT_D);
        pt_reg(acts, &a->moe_qvec, (long)N * PT_NOPT, PT_D);
        pt_reg(acts, &a->moe_scin, (long)N * PT_NOPT, PT_SCORE_IN);
        pt_reg(acts, &a->moe_sc1, (long)N * PT_NOPT, 256);
    }
    if (train) {
        pt_reg(acts, &a->moe_dy0, (long)N * PT_NOPT, 0);
        pt_reg(acts, &a->moe_dv0, N, 0);
        pt_reg(acts, &a->moe_dv, N, 0);
        if (chain_on()) {
            pt_reg(acts, &a->moe_dscin, (long)N * PT_NOPT, PT_SCORE_IN);
            pt_reg(acts, &a->moe_dq, (long)N * PT_NOPT, PT_D);
            pt_reg(acts, &a->moe_dqin, (long)N * PT_NOPT, PT_Q_IN);
        }
    }
    // scratch (max seq = LCRIT for ln/qkv/heads buffers)
    long NLC = (long)N * PT_LCRIT;
    pt_reg(acts, &ax->s_ln, NLC, PT_D);
    pt_reg(acts, &ax->s_qkv, NLC, 3 * PT_D);
    pt_reg(acts, &ax->s_q, NLC, PT_D);
    pt_reg(acts, &ax->s_k, NLC, PT_D);
    pt_reg(acts, &ax->s_v, NLC, PT_D);
    pt_reg(acts, &ax->s_ctx, NLC, PT_D);
    pt_reg(acts, &ax->s_ffn, NLC, PT_FFN);
    if (train) {
        pt_reg(acts, &a->bv_in, N, PT_VAL_IN);
        pt_reg(acts, &a->bv1, N, 256);
        pt_reg(acts, &a->blind_v, N, 1);
        pt_reg(acts, &a->hand_lin, N, PT_NC);
        pt_reg(acts, &a->prize_logit, N, PT_LTOK);
        pt_reg(acts, &a->hand_tgt, N, PT_NC);
        pt_reg(acts, &a->prized, N, PT_NC);
        pt_reg(acts, &a->prize_cnt, 1, 0);
        pt_reg(acts, &ax->s_dffn, NLC, PT_FFN);
        pt_reg(acts, &ax->s_dq, NLC, PT_D);
        pt_reg(acts, &ax->s_dk, NLC, PT_D);
        pt_reg(acts, &ax->s_dv, NLC, PT_D);
        pt_reg(acts, &ax->s_dp, (long)N * PT_H * PT_LCRIT, PT_LCRIT);
        pt_reg(acts, &a->dx_a, NLC, PT_D);
        pt_reg(acts, &a->dx_b, NLC, PT_D);
        pt_reg(acts, &a->d_g, N, 128);
        pt_reg(acts, &a->d_dvec, N, 64);
        pt_reg(acts, &a->d_rg, N, PT_VAL_IN);
        pt_reg(acts, &ax->d_tokp, (long)N * PT_LTOK, PT_D);
        pt_reg(acts, &ax->d_tokcat, (long)N * PT_LTOK, PT_TOK_IN);
        pt_reg(acts, &ax->d_bc, N, PT_D);
        pt_reg(acts, &ax->d_bcin, N, PT_BCAST_IN);
        pt_reg(acts, &ax->d_decin, N, PT_DEC_IN);
        pt_reg(acts, &ax->d_gin, N, 128);
        pt_reg(acts, &ax->d_otok_in, (long)N * PT_NORC, PT_ORC_IN);
        pt_reg(acts, &ax->d_q, (long)N * PT_NOPT, PT_D);
        pt_reg(acts, &ax->d_qin, (long)N * PT_NOPT, PT_Q_IN);
        pt_reg(acts, &ax->d_scin, (long)N * PT_NOPT, PT_SCORE_IN);
        pt_reg(acts, &ax->d_out64, (long)N * PT_NOPT, 256);
        pt_reg(acts, &ax->d_ocard, (long)N * PT_NOPT, PT_DC);
        pt_reg(acts, &ax->d_oatt, (long)N * PT_NOPT, PT_DA);
        pt_reg(acts, &ax->d_val, N, 1);
        pt_reg(acts, &ax->d_vin, N, PT_VAL_IN);
        pt_reg(acts, &ax->d_v1, N, 256);
        pt_reg(acts, &ax->d_handlin, N, PT_NC);
        pt_reg(acts, &ax->d_prizelogit, N, PT_LTOK);
    }
}

static void scratch_bind(PtcgActX* ax) {
    ax->sc.ln = ax->s_ln.data;   ax->sc.qkv = ax->s_qkv.data;
    ax->sc.q = ax->s_q.data;     ax->sc.k = ax->s_k.data;
    ax->sc.v = ax->s_v.data;     ax->sc.ctx = ax->s_ctx.data;
    ax->sc.ffn = ax->s_ffn.data; ax->sc.dffn = ax->s_dffn.data;
    ax->sc.dq = ax->s_dq.data;   ax->sc.dk = ax->s_dk.data;
    ax->sc.dv = ax->s_dv.data;   ax->sc.dp = ax->s_dp.data;
}

static void k_prize_head_launch(PtcgAct* a, PtcgWeights* w, int N, cudaStream_t s);

static void pt_dump_stage(const char* dir, const char* name,
        const float* dev, long n) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s.bin", dir, name);
    float* h = (float*)malloc(n * sizeof(float));
    cudaMemcpy(h, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    FILE* f = fopen(path, "wb");
    if (f) { fwrite(h, sizeof(float), n, f); fclose(f); }
    free(h);
}

// PTCG_SYNC_CHECK=1: sync + error-check after each forward stage
static int pt_sync_check_on(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PTCG_SYNC_CHECK"); v = e && e[0] == '1'; }
    return v;
}
#define PT_CK(stage) do { if (pt_sync_check_on()) { \
    cudaStreamSynchronize(s); cudaError_t _e = cudaGetLastError(); \
    if (_e != cudaSuccess) { fprintf(stderr, "PTCG stage %s: %s (N=%d)\n", \
        stage, cudaGetErrorString(_e), N); abort(); } } } while (0)

// ----------------------------------------------------------------- forward
// ---------------------------------------------------------------- MoE routing
// PTCG_MOE=1: five per-family expert heads (score + PPO value) selected by the
// obs family slot; 0 = shared (default). Unset/0 keeps the shared-head path
// byte-identical to the C1 model.
static int moe_on(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PTCG_MOE"); v = e && e[0] == '1'; }
    return v;
}
__global__ void k_moe_fam(float* fam, const float* obs, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) fam[n] = obs[(long)n * PT_OBS + PO_MOE_FAM];
}
__global__ void k_moe_pick_s(float* dst, const float* src, const float* fam,
        int N, int e) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT) return;
    int n = (int)(i / PT_NOPT);
    if ((int)fam[n] == e) dst[i] = src[i];
}
__global__ void k_moe_pick_v(float* dst, const float* src, const float* fam,
        int N, int e) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N && (int)fam[n] == e) dst[n] = src[n];
}
__global__ void k_moe_mask_s(float* dst, const float* src, const float* fam,
        int N, int e) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_NOPT) return;
    int n = (int)(i / PT_NOPT);
    dst[i] = ((int)fam[n] == e) ? src[i] : 0.f;
}
__global__ void k_moe_mask_v(float* dst, const float* src, const float* fam,
        int N, int e) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) dst[n] = ((int)fam[n] == e) ? src[n] : 0.f;
}
// e==0 -> shared param name; e>0 -> "m{e-1}_{suffix}"
static const char* moe_pname(char* buf, int e, const char* suf) {
    if (e == 0) return suf;
    snprintf(buf, 32, "m%d_%s", e - 1, suf);
    return buf;
}

// PTCG_MOE_CHAIN=1: full-chain experts — per-family opt_card/opt_attack + q1/q2
// + score1/2 (+ oracle_v driven by the value block). dec1/dec2 stay shared
// because d_out feeds bcast -> x0 -> trunk; a per-family dec would make the
// trunk family-dependent.
static int chain_on(void) {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("PTCG_MOE_CHAIN");
        v = (e && e[0] == '1' && moe_on()) ? 1 : 0;
    }
    return v;
}
// forward one expert's option-query-score chain into the moe_* scratch and
// moe_s_e. Leaves s_v=gelu(q1a), moe_sc1, s_ctx=gelu(sc1) for the backward.
static void moe_chain_fwd(PtcgWeights* w, PtcgAct* a, PtcgActX* ax,
        const float* obs, int N, int e, cudaStream_t s) {
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "opt_card_w");
        const char* p2 = moe_pname(x2, e, "opt_card_b");
        pt_linear(a->ocard_in.data, W(p1), W(p2), a->moe_oc64.data,
            N * PT_NOPT, PT_DC, 64, s);
    }
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "opt_attack_w");
        const char* p2 = moe_pname(x2, e, "opt_attack_b");
        pt_linear(a->oatt_in.data, W(p1), W(p2), a->moe_oa64.data,
            N * PT_NOPT, PT_DA, 64, s);
    }
    k_q_in<<<ptg((long)N * PT_NOPT * PT_Q_IN), PTB, 0, s>>>(a->moe_qin.data,
        obs, a->h.data, W("opt_type_emb"), a->moe_oc64.data, a->moe_oa64.data, N);
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "q1_w");
        const char* p2 = moe_pname(x2, e, "q1_b");
        pt_linear(a->moe_qin.data, W(p1), W(p2), a->moe_q1a.data,
            N * PT_NOPT, PT_Q_IN, PT_D, s);
    }
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->s_v.data,
        a->moe_q1a.data, (long)N * PT_NOPT * PT_D);
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "q2_w");
        const char* p2 = moe_pname(x2, e, "q2_b");
        pt_linear(ax->s_v.data, W(p1), W(p2), a->moe_qvec.data,
            N * PT_NOPT, PT_D, PT_D, s);
    }
    k_sc_in<<<ptg((long)N * PT_NOPT * PT_SCORE_IN), PTB, 0, s>>>(
        a->moe_scin.data, a->moe_qvec.data, a->h.data, a->g_out.data,
        a->d_out.data, N);
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "score1_w");
        const char* p2 = moe_pname(x2, e, "score1_b");
        pt_linear(a->moe_scin.data, W(p1), W(p2), a->moe_sc1.data,
            N * PT_NOPT, PT_SCORE_IN, 256, s);
    }
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(ax->s_ctx.data,
        a->moe_sc1.data, (long)N * PT_NOPT * 256);
    {
        char x1[32], x2[32];
        const char* p1 = moe_pname(x1, e, "score2_w");
        const char* p2 = moe_pname(x2, e, "score2_b");
        pt_linear(ax->s_ctx.data, W(p1), W(p2), a->moe_s_e.data,
            N * PT_NOPT, 256, 1, s);
    }
}

static void ptcg_forward_core(PtcgWeights* w, PtcgActX* ax, const float* obs,
        int N, int train, cudaStream_t s) {
    PtcgAct* a = &ax->a;
    a->obs_src = obs;
    scratch_bind(ax);
    if (pt_prof_active(&g_pp_fwd) && !g_pp_fwd.init)
        pt_prof_init(&g_pp_fwd, 16, "fwdT");
    if (pt_prof_active(&g_pp_fwdR) && !g_pp_fwdR.init)
        pt_prof_init(&g_pp_fwdR, 16, "fwdR");
    PtProfSite* pp = train ? &g_pp_fwd : &g_pp_fwdR;
    bool ptp = pp->init && pp->calls < PT_PROF_CALLS;
    #define PTP(i) do { if (ptp) cudaEventRecord(pp->e[i], s); } while (0)
    PTP(0);
    pt_static_upload(s);
    tables_fwd(w, &a->tab, s);
    PT_CK("tables");
    PTP(1);
    const float* cv = a->tab.card_vec.data;
    const float* av = a->tab.att_tab.data;

    k_tok_cat<<<ptg((long)N * PT_LTOK * PT_TOK_IN), PTB, 0, s>>>(
        a->tok_cat.data, obs, cv, W("zone_emb"), W("owner_emb"), N);
    pt_linear(a->tok_cat.data, W("token_proj_w"), W("token_proj_b"),
        ax->tokp.data, N * PT_LTOK, PT_TOK_IN, PT_D, s);
    PTP(2);

    // global: g1a = lin1(glob); g_out = lin2(gelu(g1a)) — gelu into s_ln staging
    pt_slice_block<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_ln.data, obs, N,
        64, PT_OBS, PO_GLOB);
    pt_linear(ax->s_ln.data, W("global1_w"), W("global1_b"), ax->g1a.data,
        N, 64, 128, s);
    pt_gelu_fwd<<<ptg((long)N * 128), PTB, 0, s>>>(ax->s_qkv.data, ax->g1a.data, (long)N * 128);
    pt_linear(ax->s_qkv.data, W("global2_w"), W("global2_b"), a->g_out.data,
        N, 128, 128, s);
    PTP(3);

    // decision: d1 = lin1(dec_in); d_out = lin2(gelu(d1))
    k_dec_in<<<ptg((long)N * PT_DEC_IN), PTB, 0, s>>>(a->dec_in.data, obs,
        cv, av, W("sel_type_emb"), W("ctx_emb"), N);
    pt_linear(a->dec_in.data, W("dec1_w"), W("dec1_b"), a->d1.data, N,
        PT_DEC_IN, 64, s);
    pt_gelu_fwd<<<ptg((long)N * 64), PTB, 0, s>>>(ax->s_qkv.data, a->d1.data, (long)N * 64);
    pt_linear(ax->s_qkv.data, W("dec2_w"), W("dec2_b"), a->d_out.data, N, 64, 64, s);
    PTP(4);

    // bcast: bc_in = [g_out || d_out]; bc_out = bcast(bc_in)
    pt_copy_block<<<ptg((long)N * 128), PTB, 0, s>>>(a->bc_in.data,
        a->g_out.data, N, 128, PT_BCAST_IN, 0);
    pt_copy_block<<<ptg((long)N * 64), PTB, 0, s>>>(a->bc_in.data,
        a->d_out.data, N, 64, PT_BCAST_IN, 128);
    pt_linear(a->bc_in.data, W("bcast_w"), W("bcast_b"), a->bc_out.data,
        N, PT_BCAST_IN, PT_D, s);

    k_x0<<<ptg((long)N * PT_LSEQ * PT_D), PTB, 0, s>>>(a->x[0].data,
        a->keep.data, ax->tokp.data, a->bc_out.data, W("readout"), obs, N);
    PT_CK("x0");
    PTP(5);

    BlkW bw;
    static const char* bpre[4] = {"b0_", "b1_", "b2_", "b3_"};
    const float* x_last = nullptr;
    for (int i = 0; i < PT_LAYERS; i++) {
        blkw(w, bpre[i], &bw, nullptr);
        BlkAct* ba = train ? &a->blk[i] : &a->blk[0];
        const float* xi = train ? a->x[i].data : a->x[i % 2].data;
        float* xo = train ? a->x[i + 1].data : a->x[(i + 1) % 2].data;
        if (ptp && i == 0) {
            if (!g_pp_blkf.init) pt_prof_init(&g_pp_blkf, 8, "blkf");
            blk_fwd(&bw, ba, &ax->sc, xi, xo, a->keep.data, N, PT_LSEQ, s,
                &g_pp_blkf);
        } else {
            blk_fwd(&bw, ba, &ax->sc, xi, xo, a->keep.data, N, PT_LSEQ, s);
        }
        PT_CK("trunk_blk");
        x_last = xo;
        PTP(6 + i);
    }
    pt_ln_fwd<<<N * PT_LSEQ, 128, 0, s>>>(a->h.data, a->lnf_mean.data,
        a->lnf_rstd.data, x_last, W("lnf_w"), W("lnf_b"),
        N * PT_LSEQ, PT_D);
    PTP(10);

    // option scorer
    k_opt_gather<<<ptg((long)N * PT_NOPT * PT_DC), PTB, 0, s>>>(
        a->ocard_in.data, a->oatt_in.data, obs, cv, av, N);
    if (chain_on()) {
        k_moe_fam<<<ptg(N), PTB, 0, s>>>(a->moe_fam.data, obs, N);
        pt_zero<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(a->scores.data,
            (long)N * PT_NOPT);
        for (int e = 0; e < 6; e++) {
            moe_chain_fwd(w, a, ax, obs, N, e, s);
            k_moe_pick_s<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(a->scores.data,
                a->moe_s_e.data, a->moe_fam.data, N, e);
        }
    } else {
    // opt_card/opt_attack linears into staging then q_in gather
    pt_linear(a->ocard_in.data, W("opt_card_w"), W("opt_card_b"),
        ax->s_q.data, N * PT_NOPT, PT_DC, 64, s);
    pt_linear(a->oatt_in.data, W("opt_attack_w"), W("opt_attack_b"),
        ax->s_k.data, N * PT_NOPT, PT_DA, 64, s);
    k_q_in<<<ptg((long)N * PT_NOPT * PT_Q_IN), PTB, 0, s>>>(a->q_in.data, obs,
        a->h.data, W("opt_type_emb"), ax->s_q.data, ax->s_k.data, N);
    pt_linear(a->q_in.data, W("q1_w"), W("q1_b"), a->q1a.data,
        N * PT_NOPT, PT_Q_IN, PT_D, s);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * PT_D), PTB, 0, s>>>(ax->s_v.data,
        a->q1a.data, (long)N * PT_NOPT * PT_D);
    pt_linear(ax->s_v.data, W("q2_w"), W("q2_b"), a->q_vec.data,
        N * PT_NOPT, PT_D, PT_D, s);
    PT_CK("options_pre");
    k_sc_in<<<ptg((long)N * PT_NOPT * PT_SCORE_IN), PTB, 0, s>>>(a->sc_in.data,
        a->q_vec.data, a->h.data, a->g_out.data, a->d_out.data, N);
    if (moe_on()) {
        char n1[32], n2[32], n3[32], n4[32];
        k_moe_fam<<<ptg(N), PTB, 0, s>>>(a->moe_fam.data, obs, N);
        pt_zero<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(a->scores.data,
            (long)N * PT_NOPT);
        for (int e = 0; e < 6; e++) {
            const char* w1 = moe_pname(n1, e, "score1_w");
            const char* b1 = moe_pname(n2, e, "score1_b");
            const char* w2 = moe_pname(n3, e, "score2_w");
            const char* b2 = moe_pname(n4, e, "score2_b");
            pt_linear(a->sc_in.data, W(w1), W(b1), a->sc1.data,
                N * PT_NOPT, PT_SCORE_IN, 256, s);
            pt_gelu_fwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(
                ax->s_ctx.data, a->sc1.data, (long)N * PT_NOPT * 256);
            pt_linear(ax->s_ctx.data, W(w2), W(b2), a->moe_s_e.data,
                N * PT_NOPT, 256, 1, s);
            k_moe_pick_s<<<ptg((long)N * PT_NOPT), PTB, 0, s>>>(a->scores.data,
                a->moe_s_e.data, a->moe_fam.data, N, e);
        }
    } else {
    pt_linear(a->sc_in.data, W("score1_w"), W("score1_b"), a->sc1.data,
        N * PT_NOPT, PT_SCORE_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)N * PT_NOPT * 256), PTB, 0, s>>>(ax->s_ctx.data,
        a->sc1.data, (long)N * PT_NOPT * 256);
    pt_linear(ax->s_ctx.data, W("score2_w"), W("score2_b"), a->scores.data,
        N * PT_NOPT, 256, 1, s);
    }
    }
    PTP(11);

    // oracle critic top
    k_otok_in<<<ptg((long)N * PT_NORC * PT_ORC_IN), PTB, 0, s>>>(
        a->otok_in.data, a->ckeep.data, obs, cv, a->keep.data, N);
    pt_linear(a->otok_in.data, W("oracle_proj_w"), W("oracle_proj_b"),
        a->otok.data, N * PT_NORC, PT_ORC_IN, PT_D, s);
    k_xc<<<ptg((long)N * PT_LCRIT * PT_D), PTB, 0, s>>>(a->xc[0].data,
        a->h.data, a->otok.data, W("critic_readout"), N);
    PT_CK("otok");
    blkw(w, "c0_", &bw, nullptr);
    blk_fwd(&bw, &a->cblk, &ax->sc, a->xc[0].data, a->xc[1].data,
        a->ckeep.data, N, PT_LCRIT, s);
    PT_CK("critic_blk");
    PTP(12);
    k_val_in<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->ov_in.data,
        a->xc[1].data, PT_LCRIT, a->g_out.data, N);
#if PT_HL
    // HL-Gauss shared critic: K bins from the oracle value features; the
    // expectation is the PPO value (out slot). MoE value routing is not
    // supported in this variant (PTCG_MOE must stay off).
    pt_linear(a->ov_in.data, W("oracle_v1_w"), W("oracle_v1_b"), a->ov1.data,
        N, PT_VAL_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->ov1.data,
        (long)N * 256);
    pt_linear(ax->s_k.data, W("hl_v_bins_w"), W("hl_v_bins_b"),
        a->ov_bins.data, N, 256, PT_VBINS, s);
    k_hl_expect<<<N, 64, 0, s>>>(ax->ov_out.data, a->ov_bins.data, N);
#else
    if (moe_on() || chain_on()) {
        char n1[32], n2[32], n3[32], n4[32];
        pt_zero<<<ptg(N), PTB, 0, s>>>(ax->ov_out.data, N);
        for (int e = 0; e < 6; e++) {
            const char* w1 = moe_pname(n1, e, "oracle_v1_w");
            const char* b1 = moe_pname(n2, e, "oracle_v1_b");
            const char* w2 = moe_pname(n3, e, "oracle_v2_w");
            const char* b2 = moe_pname(n4, e, "oracle_v2_b");
            pt_linear(a->ov_in.data, W(w1), W(b1), a->ov1.data,
                N, PT_VAL_IN, 256, s);
            pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data,
                a->ov1.data, (long)N * 256);
            pt_linear(ax->s_k.data, W(w2), W(b2), a->moe_ov_e.data,
                N, 256, 1, s);
            k_moe_pick_v<<<ptg(N), PTB, 0, s>>>(ax->ov_out.data,
                a->moe_ov_e.data, a->moe_fam.data, N, e);
        }
    } else {
    pt_linear(a->ov_in.data, W("oracle_v1_w"), W("oracle_v1_b"), a->ov1.data,
        N, PT_VAL_IN, 256, s);
    pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->ov1.data, (long)N * 256);
    pt_linear(ax->s_k.data, W("oracle_v2_w"), W("oracle_v2_b"), ax->ov_out.data,
        N, 256, 1, s);
    }
#endif
    PTP(13);

    k_fuse_out<<<ptg((long)N * (PT_NOPT + 1)), PTB, 0, s>>>(a->out.data,
        a->scores.data, ax->ov_out.data, N);
    PT_CK("fuse");
    PTP(14);

    if (train) {
        // blind V + aux heads (targets consumed in backward)
        k_val_in<<<ptg((long)N * PT_VAL_IN), PTB, 0, s>>>(a->bv_in.data,
            a->h.data, PT_LSEQ, a->g_out.data, N);
        pt_linear(a->bv_in.data, W("value1_w"), W("value1_b"), a->bv1.data,
            N, PT_VAL_IN, 256, s);
        pt_gelu_fwd<<<ptg((long)N * 256), PTB, 0, s>>>(ax->s_k.data, a->bv1.data, (long)N * 256);
        pt_linear(ax->s_k.data, W("value2_w"), W("value2_b"), a->blind_v.data,
            N, 256, 1, s);
        pt_linear(a->bv_in.data, W("aux_hand_w"), W("aux_hand_b"),
            a->hand_lin.data, N, PT_VAL_IN, PT_NC, s);
        k_prize_head_launch(a, w, N, s);
    }
    PTP(15);
    if (ptp) pt_prof_finish(pp, s);
    #undef PTP

    // Stage-parity instrumentation: PTCG_DUMP_DIR=<dir> dumps every named
    // stage as raw f32 after the forward. native/tools/stage_parity.py reads these
    // against torch-hook captures to localize a divergence. Debug-only path.
    const char* dump_dir = getenv("PTCG_DUMP_DIR");
    if (dump_dir) {
        cudaStreamSynchronize(s);
        pt_dump_stage(dump_dir, "card_vec", cv, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "att_tab", av, (long)PT_NA * PT_DA);
        pt_dump_stage(dump_dir, "sk_tab", a->tab.sk_tab.data, (long)PT_NS * PT_DC);
        pt_dump_stage(dump_dir, "a_mean", a->tab.a_mean.data, (long)PT_NC * 64);
        pt_dump_stage(dump_dir, "a_max", a->tab.a_max.data, (long)PT_NC * 64);
        pt_dump_stage(dump_dir, "s_mean", a->tab.s_mean.data, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "s_max", a->tab.s_max.data, (long)PT_NC * PT_DC);
        pt_dump_stage(dump_dir, "card_in", a->tab.card_in.data, (long)PT_NC * PT_CARD_IN);
        pt_dump_stage(dump_dir, "card_ln", a->tab.card_ln.data, (long)PT_NC * PT_CARD_IN);
        pt_dump_stage(dump_dir, "card_h1", a->tab.card_h1.data, (long)PT_NC * 256);
        pt_dump_stage(dump_dir, "tok_cat", a->tok_cat.data, (long)N * PT_LTOK * PT_TOK_IN);
        pt_dump_stage(dump_dir, "tokp", ax->tokp.data, (long)N * PT_LTOK * PT_D);
        pt_dump_stage(dump_dir, "dec_in", a->dec_in.data, (long)N * PT_DEC_IN);
        pt_dump_stage(dump_dir, "g_out", a->g_out.data, (long)N * 128);
        pt_dump_stage(dump_dir, "d_out", a->d_out.data, (long)N * 64);
        pt_dump_stage(dump_dir, "bc_out", a->bc_out.data, (long)N * PT_D);
        pt_dump_stage(dump_dir, "keep", a->keep.data, (long)N * PT_LSEQ);
        pt_dump_stage(dump_dir, "x0", a->x[0].data, (long)N * PT_LSEQ * PT_D);
        if (train) {
            for (int i = 1; i <= PT_LAYERS; i++) {
                char nm[8];
                snprintf(nm, sizeof nm, "x%d", i);
                pt_dump_stage(dump_dir, nm, a->x[i].data, (long)N * PT_LSEQ * PT_D);
            }
        }
        pt_dump_stage(dump_dir, "h", a->h.data, (long)N * PT_LSEQ * PT_D);
        pt_dump_stage(dump_dir, "q_in", a->q_in.data, (long)N * PT_NOPT * PT_Q_IN);
        pt_dump_stage(dump_dir, "q_vec", a->q_vec.data, (long)N * PT_NOPT * PT_D);
        pt_dump_stage(dump_dir, "sc_in", a->sc_in.data, (long)N * PT_NOPT * PT_SCORE_IN);
        pt_dump_stage(dump_dir, "out", a->out.data, (long)N * (PT_NOPT + 1));
    }
}

// prize head needs a strided pass over h[:,1:,:] -> (N, LTOK)
__global__ void k_prize_head(float* logit, const float* h, const float* w,
        const float* b, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK) return;
    long n = i / PT_LTOK;
    int l = (int)(i % PT_LTOK);
    const float* hv = h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    float s = b[0];
    for (int d = 0; d < PT_D; d++) s += hv[d] * w[d];
    logit[i] = s;
}

__global__ void k_prize_head_bwd(float* d_h, float* d_w, float* d_b,
        const float* d_logit, const float* h, const float* w, int N) {
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    if (i >= (long)N * PT_LTOK) return;
    long n = i / PT_LTOK;
    int l = (int)(i % PT_LTOK);
    float g = d_logit[i];
    if (g == 0.f) return;
    const float* hv = h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    float* dh = d_h + ((long)n * PT_LSEQ + 1 + l) * PT_D;
    for (int d = 0; d < PT_D; d++) {
        atomicAdd(&dh[d], g * w[d]);
        atomicAdd(&d_w[d], g * hv[d]);
    }
    atomicAdd(d_b, g);
}

#include "ptcg_model4.cu"
