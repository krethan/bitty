// Flash-TERN kernel for RDNA3 (RX 7600)
// Implements: XNOR + popcount attention with Hamming similarity
#include "hip_tern_kernel.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

__global__ void flash_tern_attention(
    const uint64_t* __restrict__ Q,
    const uint64_t* __restrict__ K,
    const half*    __restrict__ V,
    half*          __restrict__ O,
    int seq_len, int d, int d_words) {

    const int query_idx = blockIdx.x;
    if (query_idx >= seq_len) return;

    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    extern __shared__ float sdata[];
    float* scores = sdata;
    half*  weighted_vals = reinterpret_cast<half*>(sdata + seq_len);

    float q_acc = 0.0f;

    for (int key_idx = tid; key_idx < seq_len; key_idx += block_size) {
        float sim = 0.0f;
        for (int w = 0; w < d_words; ++w) {
            uint64_t q_word = Q[query_idx * d_words + w];
            uint64_t k_word = K[key_idx * d_words + w];
            sim += static_cast<float>(__popcll(~(q_word ^ k_word)));
        }
        scores[key_idx] = sim;
    }
    __syncthreads();

    float max_score = -1e30f;
    for (int i = tid; i < seq_len; i += block_size) {
        max_score = fmaxf(max_score, scores[i]);
    }
    sdata[tid] = max_score;
    __syncthreads();
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
        __syncthreads();
    }
    float global_max = sdata[0];

    float sum_exp = 0.0f;
    for (int i = tid; i < seq_len; i += block_size) {
        float exp_val = expf(scores[i] - global_max);
        scores[i] = exp_val;
        sum_exp += exp_val;
    }
    sdata[tid] = sum_exp;
    __syncthreads();
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }
    float global_sum = sdata[0];
    float inv_sum = 1.0f / global_sum;

    for (int p = 0; p < (d + 7) / 8; ++p) {
        half acc = __float2half(0.0f);
        for (int key_idx = tid; key_idx < seq_len; key_idx += block_size) {
            float w = scores[key_idx] * inv_sum;
            half v = V[key_idx * d + p * 8];
            acc = __hadd(acc, __hmul(__float2half(w), v));
        }
        weighted_vals[tid * ((d + 7) / 8) + p] = acc;
    }
    __syncthreads();

    if (tid == 0) {
        for (int p = 0; p < (d + 7) / 8; ++p) {
            half acc = __float2half(0.0f);
            for (int i = 0; i < block_size; ++i) {
                acc = __hadd(acc, weighted_vals[i * ((d + 7) / 8) + p]);
            }
            O[query_idx * d + p * 8] = acc;
        }
    }
}