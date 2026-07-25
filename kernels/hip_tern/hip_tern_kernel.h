// hip_tern_kernel.h
// AMD RDNA3 HIP kernel for Ternary-Exponential Attention
// Target: RX 7600 (gfx1102)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Wavefront-level reduction using AMD VPOPCOUNT
__device__ inline int popcount_xnor(uint64_t a, uint64_t b) {
    return __popcll(~(a ^ b));
}

// Main kernel
extern "C" __global__ void hip_tern_kernel(
    const uint64_t* __restrict__ Q,   // [tokens, d64]
    const uint64_t* __restrict__ K,   // [seq_len, d64]
    const half*    __restrict__ V,   // [seq_len, d_head]
    half*          __restrict__ Out, // [tokens, d_head]
    const int tokens,
    const int seq_len,
    const int d64,
    const int wavefront_size
) {
    const int lane = threadIdx.x % wavefront_size;
    const int waveId = threadIdx.x / wavefront_size;
    const int token = blockIdx.x * wavefront_size + waveId;

    if (token >= tokens) return;

    // Load query for this token
    uint64_t qword[32]; // Max d64=32 for 2048 dim
    for (int p = 0; p < d64; ++p) {
        qword[p] = Q[token * d64 + p];
    }

    // Accumulate scores in FP16
    half score_accum[32] = {0};

    // Process sequence in wavefront-sized blocks
    for (int s_blk = 0; s_blk < (seq_len + wavefront_size - 1) / wavefront_size; ++s_blk) {
        // Load key for this lane
        uint64_t kword[32];
        int s_idx = min(seq_len - 1, s_blk * wavefront_size + lane);
        for (int p = 0; p < d64; ++p) {
            kword[p] = K[s_idx * d64 + p];
        }

        // Compute XNOR popcount
        int pop = 0;
        for (int p = 0; p < d64; ++p) {
            pop += popcount_xnor(qword[p], kword[p]);
        }

        // Convert to FP16 and scale
        half sim = __int2half_rn(pop - (d64 * 32));
        half scaled = __hmul(sim, __float2half_rn(1.0f / sqrtf(float(d64 * 64))));
        half expv = hexp(scaled);

        // Wavefront reduction for softmax
        half wave_sum = expv;
        for (int offset = wavefront_size / 2; offset > 0; offset >>= 1) {
            wave_sum = __hadd(wave_sum, __shfl_xor(wave_sum, offset, wavefront_size));
        }

        // Broadcast denominator
        half denom = __shfl(wave_sum, 0, wavefront_size);
        half a = __hdiv(expv, denom);

        // Load value vector
        half vvec[32]; // Max d_head=256
        for (int p = 0; p < (d_head + 7) / 8; ++p) {
            vvec[p] = V[s_idx * d_head + p];
        }

        // Accumulate weighted sum
        for (int p = 0; p < (d_head + 7) / 8; ++p) {
            score_accum[p] = __hfma(a, vvec[p], score_accum[p]);
        }
    }

    // Write output
    for (int p = 0; p < (d_head + 7) / 8; ++p) {
        Out[token * d_head + p] = score_accum[p];
    }
}
