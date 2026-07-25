#pragma once

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

extern "C" __global__ void bitllm_f32_matmul(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        int a_col = t * TILE_K + tx;
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        int b_row = t * TILE_K + ty;
        if (b_row < K && col < N)
            Bs[ty][tx] = B[b_row * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" __global__ void bitllm_f32_matmul_transB(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_N][TILE_K];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_M + ty;
    int col = bx * TILE_N + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; t++) {
        int a_col = t * TILE_K + tx;
        if (row < M && a_col < K)
            As[ty][tx] = A[row * K + a_col];
        else
            As[ty][tx] = 0.0f;

        int b_col = t * TILE_K + ty;
        if (col < N && b_col < K)
            Bs[ty][tx] = B[col * K + b_col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_K; k++) {
            sum += As[ty][k] * Bs[tx][k];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
