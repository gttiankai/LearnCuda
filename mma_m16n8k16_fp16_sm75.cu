// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cassert>

#include "cuda_utils.cuh"


void CopyTile8x8(half *dst, const int dst_stride, const half *src, const int src_stride) {
    assert(dst != nullptr);
    assert(src != nullptr);
    for (int i = 0; i < 8; ++i) {
        memcpy(dst + dst_stride * i, src + src_stride * i, sizeof(half) * 8);
    }
}

void TransposeTile8x8(half *matrix, const int M, const int N) {
    assert(matrix != nullptr);
    assert(M % 8 == 0);
    assert(N % 8 == 0);
    auto buffer = new half[M * N]();
    int tile_m  = M / 8;
    int tile_n  = N / 8;
    for (int m = 0; m < tile_m; ++m) {
        for (int n = 0; n < tile_n; ++n) {
            CopyTile8x8(&buffer[(n * 8) * M + m * 8], N, &matrix[(m * 8) * N + n * 8], N);
        }
    }
    memcpy(matrix, buffer, M * N * sizeof(half));
    delete[] buffer;
}

void Gemm(const half *matrix_a, const half *matrix_b, half *matrix_c, const int M, const int N, const int K) {
    assert(matrix_a != nullptr);
    assert(matrix_b != nullptr);
    assert(matrix_c != nullptr);
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float res = 0.0f;
            for (int k = 0; k < K; k++) {
                res += (float)matrix_a[m * K + k] * (float)matrix_b[k * N + n];
            }
            matrix_c[m * N + n] = (half)res;
        }
    }
}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#    define CUTE_ARCH_CP_ASYNC_SM80_ENABLED
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#    define CUTE_ARCH_CP_ASYNC_SM75_ENABLED
#endif

__device__ __forceinline__ void LdMatrixX1(half2 *r, uint32_t addr) {
    auto *d = reinterpret_cast<uint32_t *>(r);
    // clang-format off
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                 : "=r"(d[0])
                 : "r"(addr));
    // clang-format on
}

__device__ __forceinline__ void LdMatrixX2(half2 *r, uint32_t addr) {
    auto *d = reinterpret_cast<uint32_t *>(r);
    // clang-format off
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(addr));
    // clang-format on
}

__device__ __forceinline__ void LdMatrixX4(half2 *r, uint32_t addr) {
    auto *d = reinterpret_cast<uint32_t *>(r);
    // clang-format off
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(addr));
    // clang-format on
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 8
#define WARP_SIZE 32

__device__ __forceinline__ void MmaM16N8K8(float *matrix_c, half2 *matrix_a, half2 *matrix_b) {
    auto *c = reinterpret_cast<uint32_t *>(matrix_c);
    auto *a = reinterpret_cast<uint32_t *>(matrix_a);
    auto *b = reinterpret_cast<uint32_t *>(matrix_b);
    // clang-format off
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
        "{ %0, %1, %2, %3 },"
        "{ %4, %5},"
        "{ %6 },"
        "{ %7, %8, %9, %10 };"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]),
          "r"(b[0]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
    // clang-format on
}

__device__ __forceinline__ void ConvertFp32ToFp16(half *__restrict__ fp16_register, float *__restrict__ fp32_register) {
    const int N = 4;
    for (int i = 0; i < N; ++i) {
        fp16_register[i] = __float2half_rn(fp32_register[i]);
    }
}

__global__ void MmaNaiveKernel(const half *__restrict__ matrix_a, const half *__restrict__ matrix_b,
                               half *__restrict__ matrix_c, size_t M, size_t N, size_t K) {
    const size_t K_tiles  = div_ceil(K, MMA_K);
    const size_t warp_row = blockIdx.y * MMA_M;
    const size_t warp_col = blockIdx.x * MMA_N;
    if (warp_row >= M || warp_col >= N) {
        return;
    }
    __shared__ half matrix_a_shared_mem[MMA_M][MMA_K];
    __shared__ half matrix_b_shared_mem[MMA_N][MMA_K];
    __shared__ half matrix_c_shared_mem[MMA_M][MMA_N];

    const size_t lane_id           = threadIdx.x % WARP_SIZE;
    float matrix_c_register[4]     = {0.f, 0.f, 0.f, 0.f};
    half matrix_c_register_fp16[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        if (lane_id < MMA_M) {
            // only t0 ~ t7 and t8 ~ t15 load matrix_a
            const size_t row = warp_row + lane_id % 16;
            const size_t col = i * MMA_K;

            *((int4 *)(&matrix_a_shared_mem[row][0])) = *((int4 *)(&matrix_a[row * K + col]));
        }
        if (lane_id < MMA_N) {
            // only t0 ~ t7 load matrix_b
            const size_t row = warp_col + lane_id % 8;
            const size_t col = i * MMA_K;

            *((int4 *)(&matrix_b_shared_mem[row][0])) = *((int4 *)(&matrix_b[row * K + col]));
        }
        __syncthreads();

        half2 matrix_a_register[2];
        half2 matrix_b_register[1];
        // ld row_major matrix_a, shape[16, 8], layout [16, 8]
        uint32_t matrix_a_shared_mem_lane_addr = __cvta_generic_to_shared(&matrix_a_shared_mem[lane_id % 16][0]);
        LdMatrixX2(matrix_a_register, matrix_a_shared_mem_lane_addr);
        // ld col_major matrix_b, shape[16, 8], layout [8, 16]
        uint32_t matrix_b_shared_mem_lane_addr = __cvta_generic_to_shared(&matrix_b_shared_mem[lane_id % 8][0]);
        LdMatrixX1(matrix_b_register, matrix_b_shared_mem_lane_addr);

        MmaM16N8K8(matrix_c_register, matrix_a_register, matrix_b_register);
        __syncthreads();
    }
    ConvertFp32ToFp16(matrix_c_register_fp16, matrix_c_register);
    // store
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4][0]) + lane_id % 4)     = *(uint32_t *)(&matrix_c_register_fp16[0]);
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4 + 8][0]) + lane_id % 4) = *(uint32_t *)(&matrix_c_register_fp16[2]);
    __syncthreads();
    // matrix_c row-major shape:[M, N], layout:[M, N], [16, 8]
    if (lane_id < MMA_M) {
        *((int4 *)(&matrix_c[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&matrix_c_shared_mem[lane_id][0]));
    }
}

/**
 * Matrix multiply-accumulate operation
 *      matrix_c = matrix_a * matrix_b
 * matrix_a:
 *      shape:[M, K], layout:[M, k], row-major
 * matrix_b:
 *      shape:[K, N], layout:[K, N], col-major
 * matrix_c:
 *      shape:[M, N], layout:[M, N], row-major
 *
 * */
void Mma(half *matrix_a, half *matrix_b, half *matrix_c, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    MmaNaiveKernel<<<grid, block>>>(matrix_a, matrix_b, matrix_c, M, N, K);
}

int main(int argc, char *argv[]) {
    int dev = 0;
    cudaSetDevice(dev);

    const int M = 16;
    const int N = 8;
    const int K = 16;
    printf("Matrix Multiply-accumulate: matrix_a[%2d,%2d] * matrix_b[%2d,%2d] = matrix_c[%2d,%2d]\n", M, K, K, N, M, N);
    auto *matrix_a_host = new half[M * K]();
    auto *matrix_b_host = new half[K * N]();
    auto *matrix_c_host = new half[M * N]();

    GenerateRandomFloatData(matrix_a_host, M, K, 0);
    GenerateRandomFloatData(matrix_b_host, K, N, 1);
    // implement gemm with cpu
    Gemm(matrix_a_host, matrix_b_host, matrix_c_host, M, N, K);
    // convert matrix b from row-major to col-major, matrix_b[K, N] -> matrix[N, K]

    /**
     * transpose matrix a
     * { [A, B]  -> { [A, C]
     *   [C, D] }     [B, D] }
     * */
    // TransposeTile8x8(matrix_a_host, M, K);
    /**
     * transpose from row-major to col-major,
     * matrix_b: [K, N] -> [N, K]
     * */
    Transpose2D(matrix_b_host, K, N);
    PrintMatrix<half, M, K>(matrix_a_host);
    PrintMatrix<half, N, K>(matrix_b_host);

    half *matrix_a_device, *matrix_b_device, *matrix_c_device;
    CUDA_CHECK(cudaMalloc(&matrix_a_device, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_b_device, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_c_device, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(matrix_a_device, matrix_a_host, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix_b_device, matrix_b_host, K * N * sizeof(half), cudaMemcpyHostToDevice));

    // invoke mma ptx
    half *matrix_c_device_ptx = nullptr;
    CUDA_CHECK(cudaMalloc(&matrix_c_device_ptx, M * N * sizeof(half)));
    Mma(matrix_a_device, matrix_b_device, matrix_c_device_ptx, M, N, K);
    CUDA_CHECK_LAST_ERROR();
    auto *matrix_c_host_ptx = new half[M * N]();
    CUDA_CHECK(cudaMemcpy(matrix_c_host_ptx, matrix_c_device_ptx, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    printf("compare cpu with ptx mma\n");
    CheckResult(matrix_c_host, matrix_c_host_ptx, M * N, true);

    cudaFree(matrix_a_device);
    cudaFree(matrix_b_device);
    cudaFree(matrix_c_device);
    cudaFree(matrix_c_device_ptx);

    delete[] matrix_a_host;
    delete[] matrix_b_host;
    delete[] matrix_c_host;
    delete[] matrix_c_host_ptx;
}