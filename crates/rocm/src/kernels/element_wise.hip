#pragma once

extern "C" __global__ void bitllm_f32_add(
    const float* a, const float* b, float* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

extern "C" __global__ void bitllm_f32_sub(
    const float* a, const float* b, float* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] - b[idx];
}

extern "C" __global__ void bitllm_f32_mul(
    const float* a, const float* b, float* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

extern "C" __global__ void bitllm_f32_scale(
    const float* a, float scale, float* out, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * scale;
}

extern "C" __global__ void bitllm_f32_sum(
    const float* a, float* out, int n
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}
