#pragma once

extern "C" __global__ void bitllm_softmax(
    const float* input, float* output, int rows, int cols
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row >= rows) return;

    const float* in_row = input + row * cols;
    float* out_row = output + row * cols;

    float local_max = -1e30f;
    for (int i = tid; i < cols; i += blockSize) {
        local_max = fmaxf(local_max, in_row[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockSize) {
        local_sum += expf(in_row[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];

    for (int i = tid; i < cols; i += blockSize) {
        out_row[i] = expf(in_row[i] - row_max) / row_sum;
    }
}

extern "C" __global__ void bitllm_scaled_dot_softmax(
    float* qk, float* out, int rows, int cols, float scale
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    if (row >= rows) return;

    float* row_data = qk + row * cols;

    float local_max = -1e30f;
    for (int i = tid; i < cols; i += blockSize) {
        row_data[i] *= scale;
        local_max = fmaxf(local_max, row_data[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockSize) {
        local_sum += expf(row_data[i] - row_max);
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];

    for (int i = tid; i < cols; i += blockSize) {
        out[row * cols + i] = expf(row_data[i] - row_max) / row_sum;
    }
}
