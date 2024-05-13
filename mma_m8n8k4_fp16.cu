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
#include <random>

#include "cuda_utils.cuh"

void GenerateRandomData(half *matrix, int m, int n) {
    assert(matrix != nullptr);
    assert(m >= 0);
    assert(n >= 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0, 1.0);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] = (half)dis(gen);
        }
    }
}

void Transpose2D(half *matrix, const int M, const int N) {
    assert(matrix != nullptr);
    auto buffer = new half[M * N]();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            buffer[n * M + m] = matrix[m * N + n];
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

int main(int argc, char *argv[]) {
    int dev = 0;
    cudaSetDevice(dev);
    const int M = 64;
    const int N = 128;
    const int K = 256;

    auto *matrix_a_host = new half[M * K]();
    auto *matrix_b_host = new half[K * N]();
    auto *matrix_c_host = new half[M * N]();

    GenerateRandomData(matrix_a_host, M, K);
    GenerateRandomData(matrix_b_host, K, N);
    // implement gemm with cpu
    Gemm(matrix_a_host, matrix_b_host, matrix_c_host, M, N, K);

    Transpose2D(matrix_a_host, M, K);
    half *matrix_a_device, *matrix_b_device, *matrix_c_device;
    CUDA_CHECK(cudaMalloc(&matrix_a_device, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_b_device, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&matrix_c_device, M * N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(matrix_a_device, matrix_a_host, M * K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(matrix_b_device, matrix_b_host, K * N * sizeof(half), cudaMemcpyHostToDevice));
    //    half *alpha_device;
    //    half *beta_device;
    //    CUDA_CHECK(cudaMalloc(&alpha_device, N * sizeof(half)));
    //    CUDA_CHECK(cudaMalloc(&beta_device, N * sizeof(half)));
    //    CUDA_CHECK(cudaMemset(alpha_device, 1.0f, N * sizeof(half)));
    //    CUDA_CHECK(cudaMemset(beta_device, 0.0f, N * sizeof(half)));
    half alpha_device = 1.0f;
    half beta_device  = 0.0f;
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS error: " << status << " at " << __FILE__ << ":" << __LINE__ << std::endl;
        return -1;
    }
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K, &alpha_device, matrix_a_device, CUDA_R_16F, M,
                          matrix_b_device, CUDA_R_16F, N, &beta_device, matrix_c_device, CUDA_R_16F, M, CUDA_R_16F,
                          CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaError_t error = cudaGetLastError();
        std::cerr << "CUBLAS error: " << __FILE__ << ":" << __LINE__ << " line get cublas error(" << status << ") "
                  << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    auto *matrix_c_host_check = new half[M * N]();
    CUDA_CHECK(cudaMemcpy(matrix_c_host_check, matrix_c_device, M * N * sizeof(half), cudaMemcpyDeviceToHost));
    // checkResult(matrix_c_host, matrix_c_host_check, M * N);
    Transpose2D(matrix_c_host_check, N, M);
    CheckResult(matrix_c_host, matrix_c_host_check, M * N);

    cudaFree(matrix_a_device);
    cudaFree(matrix_b_device);
    cudaFree(matrix_c_device);
    cublasDestroy(handle);

    delete[] matrix_a_host;
    delete[] matrix_b_host;
    delete[] matrix_c_host;
    delete[] matrix_c_host_check;
}