#pragma once

extern "C" __global__ void bitllm_rope(
    float* q, float* k,
    int num_heads, int head_dim, int seq_len,
    int position, float theta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total = num_heads * head_dim;

    if (idx >= total) return;

    int head = idx / head_dim;
    int d = idx % head_dim;
    int pair = d % half_dim;

    float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
    float angle = (float)position * freq;

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    float q_val = q[head * head_dim + d];
    float q_pair = q[head * head_dim + ((d + half_dim) % head_dim)];

    if (d < half_dim) {
        q[head * head_dim + d] = q_val * cos_val - q_pair * sin_val;
    } else {
        q[head * head_dim + d] = q_val * cos_val + q_pair * sin_val;
    }

    float k_val = k[head * head_dim + d];
    float k_pair = k[head * head_dim + ((d + half_dim) % head_dim)];

    if (d < half_dim) {
        k[head * head_dim + d] = k_val * cos_val - k_pair * sin_val;
    } else {
        k[head * head_dim + d] = k_val * cos_val + k_pair * sin_val;
    }
}
