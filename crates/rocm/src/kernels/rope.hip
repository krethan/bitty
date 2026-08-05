#pragma once

extern "C" __global__ void bitllm_rope(
    float* q, float* k,
    int num_heads, int head_dim, int seq_len,
    int position, float theta
) {
    int half_dim = head_dim / 2;
    // One thread per (head, pair) so each thread owns both halves of its pair
    // and reads both values before writing either (no cross-thread race).
    int total = num_heads * half_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total) return;

    int head = idx / half_dim;
    int i = idx % half_dim;

    float freq = 1.0f / powf(theta, (float)(2 * i) / (float)head_dim);
    float angle = (float)position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int lo = head * head_dim + i;
    int hi = head * head_dim + i + half_dim;

    float q_lo = q[lo];
    float q_hi = q[hi];
    q[lo] = q_lo * cos_val - q_hi * sin_val;
    q[hi] = q_lo * sin_val + q_hi * cos_val;

    float k_lo = k[lo];
    float k_hi = k[hi];
    k[lo] = k_lo * cos_val - k_hi * sin_val;
    k[hi] = k_lo * sin_val + k_hi * cos_val;
}
