// Deterministic muon_step dump: compiles against ANY src/ include path
// (pristine or patched) and runs an identical fixed optimizer sequence,
// dumping weights + momentum for a bitwise cross-version comparison.
// This is the no-split bit-identity check the plan requires; a full-run
// weight compare is meaningless because the game engine itself is unseeded
// (pt_battle_start takes no seed), so two training processes never match.
//   nvcc -O2 -std=c++17 -arch=sm_89 -DPRECISION_FLOAT -I<src> -c dump.cu
//   g++ dump.o -o dump -L$CUDA_HOME/lib64 -lcublas -lcudart -lnccl -lcurand
//   ./dump OUT.bin
#include "kernels.cu"
#include "muon.cu"

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s OUT.bin\n", argv[0]); return 2; }

    // Layout exercising every branch of the legacy walk: 2-D tall, 2-D wide,
    // 2-D square, 1-D, plus an alignment gap between two params.
    Allocator param_alloc = {};
    Allocator state_alloc = {};
    PrecisionTensor p_tall  = {.shape = {5, 3}};
    PrecisionTensor p_wide  = {.shape = {3, 7}};
    PrecisionTensor p_sq    = {.shape = {4, 4}};
    PrecisionTensor p_1d    = {.shape = {9}};
    PrecisionTensor p_sq2   = {.shape = {6, 6}};
    alloc_register(&param_alloc, &p_tall);
    alloc_register(&param_alloc, &p_wide);
    alloc_register(&param_alloc, &p_sq);
    alloc_register(&param_alloc, &p_1d);
    alloc_register(&param_alloc, &p_sq2);

    Muon muon = {};
    muon_init(&muon, &param_alloc, 0.0004, 0.95, 1e-12, 0.0, 0.01, &state_alloc);

    // Fixed allocator, then deterministic fill: offsets per alloc_create.
    if (alloc_create(&param_alloc) != cudaSuccess) return 1;
    if (alloc_create(&state_alloc) != cudaSuccess) return 1;
    muon_post_create(&muon);

    auto lcg = [](unsigned long& s) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return (float)((s >> 33) & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    };
    unsigned long seed = 12345;
    const long SPAN = param_alloc.total_bytes / 4;
    std::vector<float> w(SPAN), g(SPAN), mb(SPAN);
    for (long i = 0; i < SPAN; i++) { w[i] = lcg(seed); g[i] = lcg(seed) * 0.01f; mb[i] = lcg(seed) * 0.001f; }
    cudaMemcpy(param_alloc.mem, w.data(), SPAN * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(muon.mb_puf.data, mb.data(), SPAN * 4, cudaMemcpyHostToDevice);

    // Warm momentum with one step, then 3 gated steps + 2 vetoed + 2 gated.
    FloatTensor weights = {.data = (float*)param_alloc.mem, .shape = {SPAN}};
    precision_t* grad_dev = nullptr;
    cudaMalloc(&grad_dev, SPAN * 4);
    PrecisionTensor grads = {.data = grad_dev, .shape = {SPAN}};

    float gate = 1.0f;
    float* gate_dev = nullptr;
    cudaMalloc(&gate_dev, 4);
    cudaMemcpy(gate_dev, &gate, 4, cudaMemcpyHostToDevice);
    muon.gate_ptr = gate_dev;

    auto one_step = [&](float gv) {
        cudaMemcpy(gate_dev, &gv, 4, cudaMemcpyHostToDevice);
        cudaMemcpy(grad_dev, g.data(), SPAN * 4, cudaMemcpyHostToDevice);
        muon_step(&muon, weights, grads, 1.5f);
        cudaDeviceSynchronize();
    };
    one_step(1.0f);   // warm momentum
    one_step(1.0f); one_step(1.0f); one_step(1.0f);
    one_step(0.0f); one_step(0.0f);   // vetoed: must be exact no-ops
    one_step(1.0f); one_step(1.0f);

    std::vector<float> hw(SPAN), hm(SPAN);
    cudaMemcpy(hw.data(), weights.data, SPAN * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hm.data(), muon.mb_puf.data, SPAN * 4, cudaMemcpyDeviceToHost);
    FILE* f = fopen(argv[1], "wb");
    fwrite(hw.data(), 4, SPAN, f);
    fwrite(hm.data(), 4, SPAN, f);
    fclose(f);
    printf("dumped %ld floats to %s\n", SPAN, argv[1]);
    return 0;
}
