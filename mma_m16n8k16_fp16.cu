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
#include <iostream>

#include "cuda_utils.cuh"

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
#endif

__device__ __forceinline__ void LdMatrixX2(uint32_t *r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

__device__ __forceinline__ void LdMatrixX4(uint32_t *r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void MmaM16N8K16(uint32_t *c, uint32_t *a, uint32_t *b) {
#ifdef CUTE_ARCH_CP_ASYNC_SM80_ENABLED
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        "{ %0, %1 },"
        "{ %2, %3, %4, %5 },"
        "{ %6, %7 },"
        "{ %8, %9 };"
        : "=r"(c[0]), "=r"(c[1])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(c[0]), "r"(c[1]));
#endif
}

inline __device__ __host__ size_t div_ceil(size_t a, size_t b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

#define MMA_M 16
#define MMA_N 8
#define MMA_K 16
#define WARP_SIZE 32

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

    const size_t lane_id          = threadIdx.x % WARP_SIZE;
    uint32_t matrix_c_register[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&matrix_a_shared_mem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&matrix_a[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&matrix_b_shared_mem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&matrix_b[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }
        __syncthreads();

        uint32_t matrix_a_register[4];
        uint32_t matrix_b_register[2];
        // ld row_major matrix_a, shape[16, 16], layout [16, 16]
        uint32_t matrix_a_shared_mem_lane_addr =
            __cvta_generic_to_shared(&matrix_a_shared_mem[lane_id % 16][(lane_id / 16) * 8]);
        LdMatrixX4(matrix_a_register, matrix_a_shared_mem_lane_addr);
        // ld col_major matrix_b, shape[16, 8], layout [8, 16]
        uint32_t matrix_b_shared_mem_lane_addr =
            __cvta_generic_to_shared(&matrix_b_shared_mem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LdMatrixX2(matrix_b_register, matrix_b_shared_mem_lane_addr);

        MmaM16N8K16(matrix_c_register, matrix_a_register, matrix_b_register);
        __syncthreads();
    }
    // store
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4][0]) + lane_id % 4)     = matrix_c_register[0];
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4 + 8][0]) + lane_id % 4) = matrix_c_register[1];
    __syncthreads();
    // matrix_c row-major shape:[M, N], layout:[M, N], [16, 8]
    if (lane_id < MMA_M) {
        *((int4 *)(&matrix_c[(warp_row + lane_id) * N + warp_col])) = *((int4 *)(&matrix_c_shared_mem[lane_id][0]));
    }
}

__device__ __forceinline__ void MmaM16N8K16Fp32(uint32_t *c, uint32_t *a, uint32_t *b) {
#ifdef CUTE_ARCH_CP_ASYNC_SM80_ENABLED
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"
        "{ %0, %1, %2, %3 },"
        "{ %4, %5, %6, %7 },"
        "{ %8, %9 },"
        "{ %10, %11, %12, %13 };"
        : "=r"(c[0]), "=r"(c[1]),"=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
    );
#endif
}

__device__ __forceinline__ void ConvertFp32ToFp16(uint32_t *__restrict__ fp16_register,
                                                  uint32_t *__restrict__ fp32_register) {
    const int N = 4;
    for (int i = 0; i < N; ++i) {
        *(reinterpret_cast<half *>(fp16_register) + i) = __float2half_rn(*(reinterpret_cast<float *>(fp32_register) + i));
    }
}

__global__ void MmaNaiveKernelFp32(const half *__restrict__ matrix_a, const half *__restrict__ matrix_b,
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

    const size_t lane_id               = threadIdx.x % WARP_SIZE;
    uint32_t matrix_c_register[4]      = {0, 0, 0, 0};
    uint32_t matrix_c_register_fp16[2] = {0, 0};

#pragma unroll
    for (size_t i = 0; i < K_tiles; ++i) {
        *((int4 *)(&matrix_a_shared_mem[lane_id / 2][0]) + lane_id % 2) =
            *((int4 *)(&matrix_a[(warp_row + lane_id / 2) * K + i * MMA_K]) + lane_id % 2);

        if (lane_id < MMA_N * 2) {
            *((int4 *)(&matrix_b_shared_mem[lane_id / 2][0]) + lane_id % 2) =
                *((int4 *)(&matrix_b[i * MMA_K + (warp_col + lane_id / 2) * K]) + lane_id % 2);
        }
        __syncthreads();

        uint32_t matrix_a_register[4];
        uint32_t matrix_b_register[2];
        // ld row_major matrix_a, shape[16, 16], layout [16, 16]
        uint32_t matrix_a_shared_mem_lane_addr =
            __cvta_generic_to_shared(&matrix_a_shared_mem[lane_id % 16][(lane_id / 16) * 8]);
        LdMatrixX4(matrix_a_register, matrix_a_shared_mem_lane_addr);
        // ld col_major matrix_b, shape[16, 8], layout [8, 16]
        uint32_t matrix_b_shared_mem_lane_addr =
            __cvta_generic_to_shared(&matrix_b_shared_mem[lane_id % 8][((lane_id / 8) % 2) * 8]);
        LdMatrixX2(matrix_b_register, matrix_b_shared_mem_lane_addr);

        MmaM16N8K16Fp32(matrix_c_register, matrix_a_register, matrix_b_register);
        __syncthreads();
    }
    ConvertFp32ToFp16(matrix_c_register_fp16, matrix_c_register);
    // store
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4][0]) + lane_id % 4)     = matrix_c_register_fp16[0];
    *((uint32_t *)(&matrix_c_shared_mem[lane_id / 4 + 8][0]) + lane_id % 4) = matrix_c_register_fp16[1];
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
void MMAPTX(const half *matrix_a, const half *matrix_b, half *matrix_c, size_t M, size_t N, size_t K) {
    dim3 block(WARP_SIZE);
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M));
    //MmaNaiveKernel<<<grid, block>>>(matrix_a, matrix_b, matrix_c, M, N, K);
    MmaNaiveKernelFp32<<<grid, block>>>(matrix_a, matrix_b, matrix_c, M, N, K);
}

int main(int argc, char *argv[]) {
    int dev = 0;
    cudaSetDevice(dev);

    const int M = 16;
    const int N = 8;
    const int K = 16;

    auto *matrix_a_host = new half[M * K]();
    auto *matrix_b_host = new half[K * N]();
    auto *matrix_c_host = new half[M * N]();

    GenerateRandomFloatData<half>(matrix_a_host, M, K, 0);
    GenerateRandomFloatData<half>(matrix_b_host, K, N, 1);
    // implement gemm with cpu
    Gemm(matrix_a_host, matrix_b_host, matrix_c_host, M, N, K);
    // convert matrix b from row-major to col-major, matrix_b[K, N] -> matrix[N, K]
    Transpose2D(matrix_b_host, K, N);
    half *matrix_a_device, *matrix_b_device, *matrix_c_device;
    CUDA_CHECK(cudaMalloc(&matrix_a_device, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_b_device, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_c_device, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(matrix_a_device, matrix_a_host, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix_b_device, matrix_b_host, K * N * sizeof(half), cudaMemcpyHostToDevice));
    half alpha_device = 1.0f;
    half beta_device  = 0.0f;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    CUBLAS_CHECK(status);
    status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha_device, matrix_a_device, CUDA_R_16F, K,
                          matrix_b_device, CUDA_R_16F, K, &beta_device, matrix_c_device, CUDA_R_16F, M, CUDA_R_16F,
                          CUBLAS_GEMM_DEFAULT);
    CUBLAS_CHECK(status);

    auto *matrix_c_host_cublas = new half[M * N]();
    CUDA_CHECK(cudaMemcpy(matrix_c_host_cublas, matrix_c_device, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    // convert matrix c from col-major to row-major
    Transpose2D(matrix_c_host_cublas, N, M);
    printf("compare cpu with cublasGemmEx\n");
    CheckResult(matrix_c_host, matrix_c_host_cublas, M * N);

    // invoke mma ptx
    half *matrix_c_device_ptx = nullptr;
    CUDA_CHECK(cudaMalloc(&matrix_c_device_ptx, M * N * sizeof(half)));
    MMAPTX(matrix_a_device, matrix_b_device, matrix_c_device_ptx, M, N, K);
    auto *matrix_c_host_ptx = new half[M * N]();
    CUDA_CHECK(cudaMemcpy(matrix_c_host_ptx, matrix_c_device_ptx, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    printf("compare cpu with ptx mma\n");
    CheckResult(matrix_c_host, matrix_c_host_ptx, M * N);

    printf("compare cublas with ptx mma\n");
    CheckResult(matrix_c_host_cublas, matrix_c_host_ptx, M * N, true);

    printf("compare cpu with ptx mma\n");
    CheckResult(matrix_c_host, matrix_c_host_ptx, M * N, true);

    cudaFree(matrix_a_device);
    cudaFree(matrix_b_device);
    cudaFree(matrix_c_device);
    cudaFree(matrix_c_device_ptx);
    cublasDestroy(handle);

    delete[] matrix_a_host;
    delete[] matrix_b_host;
    delete[] matrix_c_host;
    delete[] matrix_c_host_cublas;
    delete[] matrix_c_host_ptx;
}