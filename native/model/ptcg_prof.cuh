// Per-stage cudaEvent probes for ptcg_forward_core / ptcg_backward_core.
// Enabled only when PTCG_PT_DEBUG=2; probed for the first PT_PROF_CALLS train
// calls, then fully dormant (one int compare per call). Diagnostic only —
// no kernel, buffer or ordering change on the default path (same contract as
// PTCG_CG_DEBUG in pufferlib.cu).
#ifndef PTCG_PROF_CUH
#define PTCG_PROF_CUH

#include <cuda_runtime.h>
#include <stdio.h>

#define PT_PROF_CALLS 4          // probed train fwd/bwd invocations
#define PT_PROF_MAX 20           // boundaries per site

struct PtProfSite {
    cudaEvent_t e[PT_PROF_MAX];
    int n;        // boundaries used by this site
    long calls;   // completed invocations
    bool init;
    const char* tag;
};

static PtProfSite g_pp_fwd, g_pp_fwdR, g_pp_bwd, g_pp_blkf, g_pp_blkb;

static inline bool pt_prof_on(void) {
    static int v = -1;
    if (v < 0) { const char* e = getenv("PTCG_PT_DEBUG"); v = e && e[0] == '2'; }
    return v;
}

static inline void pt_prof_init(PtProfSite* s, int n, const char* tag) {
    for (int i = 0; i < n; i++) cudaEventCreate(&s->e[i]);
    s->n = n;
    s->calls = 0;
    s->init = true;
    s->tag = tag;
}

// gate: probes active for this invocation?
static inline bool pt_prof_active(PtProfSite* s) {
    return pt_prof_on() && s->calls < PT_PROF_CALLS;
}

static inline void pt_prof_finish(PtProfSite* s, cudaStream_t stream) {
    cudaEventSynchronize(s->e[s->n - 1]);
    fprintf(stderr, "[PTP %s]", s->tag);
    for (int i = 0; i + 1 < s->n; i++) {
        float ms = -1.f;
        cudaError_t err = cudaEventElapsedTime(&ms, s->e[i], s->e[i + 1]);
        fprintf(stderr, " %d=%.3f", i, err == cudaSuccess ? ms : -1.f);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
    s->calls++;
}

#endif // PTCG_PROF_CUH
