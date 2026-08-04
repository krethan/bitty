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

// XNOR + LUT fused BIT1 matmul (single global scale)
// Weight is packed: 1 bit per ternary weight (+1 or -1), MSB-first.
// Each byte stores 8 weights. Group size == 0 means global scale.
extern "C" __global__ void bitllm_bit1_matmul(
    const float* __restrict__ A,       // [M, K] input (FP32)
    const unsigned char* __restrict__ W, // [ceil(N*K/8)] packed ternary weights
    const float* __restrict__ scales,    // [1] global scale (or per-group)
    float* __restrict__ C,               // [M, N] output
    int M, int N, int K,
    int group_size,
    const unsigned char* __restrict__ outlier_mask, // [ceil(N*K/8)] or nullptr
    const float* __restrict__ outlier_vals          // [num_outliers] or nullptr
) {
    // Each thread block handles a tile of output
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    float w_scale = scales[0];

    // Process K in chunks of 8 bits
    for (int k_base = 0; k_base < K; k_base += 8) {
        int k_end = min(k_base + 8, K);
        int nbits = k_end - k_base;
        int chunk_mask = (0xFF >> (8 - nbits));

        // Pack 8 input signs into a byte (bit i = sign of A[row, k_base+i])
        unsigned char in_byte = 0;
        float mag[8] = {0};
        float mag_total = 0.0f;
        for (int b = 0; b < nbits; b++) {
            float v = A[row * K + k_base + b];
            if (v > 0.0f) in_byte |= (1 << b);
            float abs_v = v >= 0.0f ? v : -v;
            mag[b] = abs_v;
            mag_total += abs_v;
        }

        // Build LUT: lut[mask] = sum of mag[b] where mask bit b is set
        float lut[256];
        for (int i = 0; i < 256; i++) lut[i] = 0.0f;
        for (int b = 0; b < nbits; b++) {
            int step = 1 << b;
            for (int mask = step; mask < (1 << nbits); mask++) {
                if (mask & step) {
                    lut[mask] = lut[mask ^ step] + mag[b];
                }
            }
        }

        // Compute contribution to up to 4 output columns per thread
        // (each thread handles 1 column, but we vectorize across columns)
        for (int j = 0; j < 4 && (col + j) < N; j++) {
            // Pack 8 weight signs for this output column
            unsigned char w_byte = 0;
            for (int b = 0; b < nbits; b++) {
                int w_idx = (col + j) * K + k_base + b;
                if ((W[w_idx / 8] >> (w_idx % 8)) & 1) {
                    w_byte |= (1 << b);
                }
            }

            // XNOR: matching bits contribute +mag, non-matching contribute -mag
            unsigned char match = ~(in_byte ^ w_byte) & chunk_mask;
            float contrib = 2.0f * lut[match] - mag_total;

            // Apply group scale
            int g = group_size > 0 ? (k_base / group_size) : 0;
            float gs = group_size > 0 ? scales[g] : w_scale;
            sum += gs * contrib;
        }
    }

    C[row * N + col] = sum;

    // Apply outlier correction if present
    if (outlier_mask != nullptr && outlier_vals != nullptr) {
        for (int o = 0; o < K; o++) {
            int idx = col * K + o;
            int byte_idx = idx / 8;
            int bit_idx = idx % 8;
            if ((outlier_mask[byte_idx] >> bit_idx) & 1) {
                int g = group_size > 0 ? (o / group_size) : 0;
                float ws = group_size > 0 ? scales[g] : w_scale;
                int bit = (W[byte_idx] >> bit_idx) & 1;
                float sign_val = bit ? ws : -ws;
                float correction = outlier_vals[idx] - sign_val;
                float a = A[row * K + o];
                C[row * N + col] += a * correction;
            }
        }
    }
}

// Group-wise scale BIT1 matmul
extern "C" __global__ void bitllm_bit1_matmul_grouped(
    const float* __restrict__ A,
    const unsigned char* __restrict__ W,
    const float* __restrict__ scales,
    float* __restrict__ C,
    int M, int N, int K,
    int group_size
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;

    for (int k_base = 0; k_base < K; k_base += 8) {
        int k_end = min(k_base + 8, K);
        int nbits = k_end - k_base;
        int chunk_mask = (0xFF >> (8 - nbits));

        unsigned char in_byte = 0;
        float mag[8] = {0};
        float mag_total = 0.0f;
        for (int b = 0; b < nbits; b++) {
            float v = A[row * K + k_base + b];
            if (v > 0.0f) in_byte |= (1 << b);
            float abs_v = v >= 0.0f ? v : -v;
            mag[b] = abs_v;
            mag_total += abs_v;
        }

        float lut[256];
        for (int i = 0; i < 256; i++) lut[i] = 0.0f;
        for (int b = 0; b < nbits; b++) {
            int step = 1 << b;
            for (int mask = step; mask < (1 << nbits); mask++) {
                if (mask & step) {
                    lut[mask] = lut[mask ^ step] + mag[b];
                }
            }
        }

        int g = (k_base / group_size);
        float gs = scales[g];

        unsigned char w_byte = 0;
        for (int b = 0; b < nbits; b++) {
            int w_idx = col * K + k_base + b;
            if ((W[w_idx / 8] >> (w_idx % 8)) & 1) {
                w_byte |= (1 << b);
            }
        }

        unsigned char match = ~(in_byte ^ w_byte) & chunk_mask;
        sum += gs * (2.0f * lut[match] - mag_total);
    }

    C[row * N + col] = sum;
}
