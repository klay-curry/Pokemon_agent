// Standalone AdamW-split unit test for the native optimizer (V21).
// Builds with nvcc only, no pybind/env/tables:
//   nvcc -O2 -std=c++17 -arch=sm_XX -DPRECISION_FLOAT -I../src \
//       tests/test_adamw.cu -o ../build/test_adamw -lcublas -lcudart
// Covers the four properties the training runs depend on:
//   1. AdamW-tagged params follow torch.optim.AdamW exactly (decoupled wd,
//      bias correction) over several steps, including a warm start (nonzero
//      initial m/v/t restored like a PTCGSTA2 resume).
//   2. Untagged params are NOT touched by the AdamW path (their m/v stay 0).
//   3. A vetoed step (gate = 0) is a full no-op: weights, m, v, t unchanged.
//   4. The step counter t advances only on applied steps.
#include "kernels.cu"
#include "muon.cu"

#include <cmath>
#include <cstdio>
#include <vector>

#define CHECK(cond) \
    do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, \
        __LINE__, #cond); return 1; } } while (0)

// CPU reference in the kernel's exact op order (fp32): torch AdamW with
// decoupled weight decay and bias correction.
static void adamw_ref(float& w, float& m, float& v, float g, float lr,
        float b1, float b2, float wd, float eps, long t) {
    float inv1 = 1.0f / (float)(1.0 - std::pow((double)b1, (double)t));
    float inv2 = 1.0f / (float)(1.0 - std::pow((double)b2, (double)t));
    float m_new = b1 * m + (1.0f - b1) * g;
    float v_new = b2 * v + (1.0f - b2) * g * g;
    m = m_new;
    v = v_new;
    float update = (m_new * inv1) / (std::sqrt(v_new * inv2) + eps);
    w = w * (1.0f - lr * wd) - lr * update;
}

int main() {
    // Three params: a 2-D tagged (AdamW), a 2-D untagged (Muon), a 1-D
    // tagged (AdamW) — the walk order and offsets mirror the real arena.
    const long N0 = 6, N2 = 3;   // p1's 4 elements are addressed via SPAN
    // Arena layout (alloc_create's 16B alignment = 4-float steps): p0 at 0,
    // p1 at 8, p2 at 12; span = 15 floats. The adamw/mb state buffers mirror
    // this layout, so every manual offset below must use these.
    const long OFF0 = 0, OFF1 = 8, OFF2 = 12, SPAN = 15;
    Allocator param_alloc = {};
    Allocator state_alloc = {};
    PrecisionTensor p0 = {.shape = {N0, 1}};   // (6,1) 2-D tagged
    PrecisionTensor p1 = {.shape = {2, 2}};    // (2,2) 2-D untagged
    PrecisionTensor p2 = {.shape = {N2}};      // 1-D tagged
    alloc_register(&param_alloc, &p0);
    alloc_mark_last_adamw(&param_alloc);
    alloc_register(&param_alloc, &p1);
    alloc_register(&param_alloc, &p2);
    alloc_mark_last_adamw(&param_alloc);

    Muon muon = {};
    muon_init(&muon, &param_alloc, 0.01, 0.95, 1e-12, 0.0, 0.01, &state_alloc);
    muon_enable_adamw(&muon, &state_alloc, 0.003, 0.9, 0.95, 0.01);
    CHECK(alloc_create(&param_alloc) == cudaSuccess);
    CHECK(alloc_create(&state_alloc) == cudaSuccess);
    muon_post_create(&muon);

    // Deterministic weights and grads on host, uploaded in place.
    std::vector<float> w0 = {0.10f, -0.20f, 0.30f, -0.40f, 0.50f, -0.60f};
    std::vector<float> w1 = {1.0f, 0.5f, -0.5f, -1.0f};
    std::vector<float> w2 = {0.25f, -0.75f, 1.25f};
    // Warm-start state (as if restored from a PTCGSTA2 file at t=1).
    std::vector<float> m0 = {0.01f, -0.02f, 0.03f, -0.04f, 0.05f, -0.06f};
    std::vector<float> v0 = {0.002f, 0.003f, 0.004f, 0.005f, 0.006f, 0.007f};
    std::vector<float> m2 = {0.01f, 0.02f, 0.03f}, v2 = {0.001f, 0.002f, 0.003f};
    long t_start = 1;
    cudaMemcpy(p0.data, w0.data(), N0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p1.data, w1.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(p2.data, w2.data(), N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(muon.adamw_m_puf.data, m0.data(), N0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(muon.adamw_v_puf.data, v0.data(), N0 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(muon.adamw_m_puf.data + OFF2, m2.data(), N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(muon.adamw_v_puf.data + OFF2, v2.data(), N2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(muon.adamw_t_puf.data, &t_start, sizeof(long), cudaMemcpyHostToDevice);
    // Muon momentum for the untagged param so its path is exercised too.
    std::vector<float> mb1 = {0.1f, -0.1f, 0.2f, -0.2f};
    cudaMemcpy(muon.mb_puf.data + OFF1, mb1.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    // Gate = 1 (apply), one step of grads.
    float gate = 1.0f;
    FloatTensor gate_puf = {.shape = {1}};
    alloc_register(&state_alloc, &gate_puf);
    // Re-alloc just this tensor: easier to cudaMalloc directly instead.
    float* gate_dev = nullptr;
    cudaMalloc(&gate_dev, sizeof(float));
    cudaMemcpy(gate_dev, &gate, sizeof(float), cudaMemcpyHostToDevice);
    muon.gate_ptr = gate_dev;

    std::vector<float> g0 = {0.001f, -0.002f, 0.003f, -0.004f, 0.005f, -0.006f};
    std::vector<float> g1 = {0.01f, -0.02f, 0.03f, -0.04f};  // p1 at OFF1
    std::vector<float> g2 = {0.002f, -0.003f, 0.004f};
    std::vector<float> grad(SPAN, 0.0f);
    for (int i = 0; i < N0; i++) grad[OFF0 + i] = g0[i];
    for (int i = 0; i < 4; i++) grad[OFF1 + i] = g1[i];
    for (int i = 0; i < N2; i++) grad[OFF2 + i] = g2[i];
    PrecisionTensor grads = {.data = (precision_t*)param_alloc.mem, .shape = {0}};
    // Use a dedicated grads buffer (same total span as the params arena).
    precision_t* grad_dev = nullptr;
    cudaMalloc(&grad_dev, param_alloc.total_bytes);
    cudaMemcpy(grad_dev, grad.data(), param_alloc.total_bytes, cudaMemcpyHostToDevice);
    grads = {.data = grad_dev, .shape = {param_alloc.total_bytes / 4}};
    FloatTensor weights = {.data = (float*)param_alloc.mem,
                           .shape = {param_alloc.total_bytes / 4}};

    const float LR = 0.003f, B1 = 0.9f, B2 = 0.95f, WD = 0.01f, EPS = 1e-8f;
    // CPU reference state.
    std::vector<float> rw0 = w0, rm0 = m0, rv0 = v0;
    std::vector<float> rw2 = w2, rm2 = m2, rv2 = v2;
    long rt = t_start;

    for (int step = 0; step < 5; step++) {
        // Refresh grads (same values every step is fine for a math check).
        cudaMemcpy(grad_dev, grad.data(), param_alloc.total_bytes, cudaMemcpyHostToDevice);
        muon_step(&muon, weights, grads, 1.5f);
        cudaDeviceSynchronize();
        rt += 1;
        for (int i = 0; i < N0; i++)
            adamw_ref(rw0[i], rm0[i], rv0[i], g0[i], LR, B1, B2, WD, EPS, rt);
        for (int i = 0; i < N2; i++)
            adamw_ref(rw2[i], rm2[i], rv2[i], g2[i], LR, B1, B2, WD, EPS, rt);
    }

    std::vector<float> hw(SPAN, 0.0f), hm(SPAN, 0.0f), hv(SPAN, 0.0f);
    cudaMemcpy(hw.data(), weights.data, hw.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), muon.adamw_m_puf.data, hm.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv.data(), muon.adamw_v_puf.data, hv.size() * sizeof(float), cudaMemcpyDeviceToHost);
    long t_host = 0;
    cudaMemcpy(&t_host, muon.adamw_t_puf.data, sizeof(long), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N0; i++) {
        if (std::fabs(hw[OFF0 + i] - rw0[i]) >= 1e-6f ||
            std::fabs(hm[OFF0 + i] - rm0[i]) >= 1e-6f ||
            std::fabs(hv[OFF0 + i] - rv0[i]) >= 1e-6f)
            fprintf(stderr, "p0[%d] w %.9g vs %.9g | m %.9g vs %.9g | "
                    "v %.9g vs %.9g\n", i, hw[OFF0 + i], rw0[i],
                    hm[OFF0 + i], rm0[i], hv[OFF0 + i], rv0[i]);
        CHECK(std::fabs(hw[OFF0 + i] - rw0[i]) < 1e-6f);
        CHECK(std::fabs(hm[OFF0 + i] - rm0[i]) < 1e-6f);
        CHECK(std::fabs(hv[OFF0 + i] - rv0[i]) < 1e-6f);
    }
    for (int i = 0; i < N2; i++) {
        if (std::fabs(hw[OFF2 + i] - rw2[i]) >= 1e-6f)
            fprintf(stderr, "p2[%d] w %.9g vs %.9g | m %.9g vs %.9g | "
                    "v %.9g vs %.9g\n", i, hw[OFF2 + i], rw2[i],
                    hm[OFF2 + i], rm2[i], hv[OFF2 + i], rv2[i]);
        CHECK(std::fabs(hw[OFF2 + i] - rw2[i]) < 1e-6f);
        CHECK(std::fabs(hm[OFF2 + i] - rm2[i]) < 1e-6f);
        CHECK(std::fabs(hv[OFF2 + i] - rv2[i]) < 1e-6f);
    }
    CHECK(t_host == t_start + 5);

    // 2. Untagged param: AdamW state untouched (still the warm-start zeros
    //    we never wrote for it) and its weights moved via the Muon path only
    //    (a different rule; here we only assert the AdamW m/v stayed 0).
    for (int i = 0; i < 4; i++) {
        CHECK(hm[OFF1 + i] == 0.0f);
        CHECK(hv[OFF1 + i] == 0.0f);
    }

    // 3. Vetoed step: everything frozen.
    std::vector<float> hw_before = hw;
    gate = 0.0f;
    cudaMemcpy(gate_dev, &gate, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_dev, grad.data(), param_alloc.total_bytes, cudaMemcpyHostToDevice);
    muon_step(&muon, weights, grads, 1.5f);
    cudaDeviceSynchronize();
    std::vector<float> hw_after(hw.size(), 0.0f), hm_after(hm.size(), 0.0f),
        hv_after(hv.size(), 0.0f);
    cudaMemcpy(hw_after.data(), weights.data, hw.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hm_after.data(), muon.adamw_m_puf.data, hm.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hv_after.data(), muon.adamw_v_puf.data, hv.size() * sizeof(float), cudaMemcpyDeviceToHost);
    long t_veto = -1;
    cudaMemcpy(&t_veto, muon.adamw_t_puf.data, sizeof(long), cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < hw.size(); i++) CHECK(hw_after[i] == hw_before[i]);
    // The veto must leave the AdamW state untouched too, not just the
    // weights (a multiply-gate variant would still trip the weights check,
    // but assert the m/v contract directly).
    for (size_t i = 0; i < hm.size(); i++) {
        CHECK(hm_after[i] == hm[i]);
        CHECK(hv_after[i] == hv[i]);
    }
    CHECK(t_veto == t_host);

    // 4. Gate back on: exactly one more applied step.
    gate = 1.0f;
    cudaMemcpy(gate_dev, &gate, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grad_dev, grad.data(), param_alloc.total_bytes, cudaMemcpyHostToDevice);
    muon_step(&muon, weights, grads, 1.5f);
    cudaDeviceSynchronize();
    cudaMemcpy(&t_veto, muon.adamw_t_puf.data, sizeof(long), cudaMemcpyDeviceToHost);
    CHECK(t_veto == t_host + 1);
    rt += 1;
    for (int i = 0; i < N0; i++)
        adamw_ref(rw0[i], rm0[i], rv0[i], g0[i], LR, B1, B2, WD, EPS, rt);
    cudaMemcpy(hw_after.data(), weights.data, hw.size() * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N0; i++)
        CHECK(std::fabs(hw_after[OFF0 + i] - rw0[i]) < 1e-6f);

    printf("test_adamw: PASS (torch-order AdamW over 5+1 steps, warm start, "
           "veto no-op, t counting, untagged m/v zero)\n");
    return 0;
}
