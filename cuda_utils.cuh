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

#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess) {                                          \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);                     \
            printf("code:%d,reason:%s\n", error, cudaGetErrorString(error)); \
            exit(1);                                                         \
        }                                                                    \
    }

#define CUDA_CHECK_LAST_ERROR()                                \
    {                                                          \
        cudaError_t error = cudaGetLastError();                \
        if (error != cudaSuccess) {                            \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);       \
            printf("Error: %s.\n", cudaGetErrorString(error)); \
            exit(1);                                           \
        }                                                      \
    }

const char *cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "Unknown error";
    }
}

#define CUBLAS_CHECK(status)                                      \
    {                                                             \
        if (status != CUBLAS_STATUS_SUCCESS) {                    \
            printf("ERROR: %s:%d,", __FILE__, __LINE__);          \
            printf("Error: %s.\n", cublasGetErrorString(status)); \
            exit(1);                                              \
        }                                                         \
    }

#include <time.h>

#ifdef _WIN32
#    include <windows.h>
#else

#    include <sys/time.h>

#endif
#ifdef _WIN32
int gettimeofday(struct timeval *tp, void *tzp) {
    time_t clock;
    struct tm tm;
    SYSTEMTIME wtm;
    GetLocalTime(&wtm);
    tm.tm_year  = wtm.wYear - 1900;
    tm.tm_mon   = wtm.wMonth - 1;
    tm.tm_mday  = wtm.wDay;
    tm.tm_hour  = wtm.wHour;
    tm.tm_min   = wtm.wMinute;
    tm.tm_sec   = wtm.wSecond;
    tm.tm_isdst = -1;
    clock       = mktime(&tm);
    tp->tv_sec  = clock;
    tp->tv_usec = wtm.wMilliseconds * 1000;
    return (0);
}
#endif

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, nullptr);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void initialData(float *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xffff) / 1000.0f;
    }
}

void initialData_int(int *ip, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = int(rand() & 0xff);
    }
}

void printMatrix(float *C, const int nx, const int ny) {
    float *ic = C;
    printf("Matrix<%d,%d>:\n", ny, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f ", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
}

void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    CUDA_CHECK(cudaSetDevice(dev));
}

template <typename T>
void CheckResult(T *stand, T *ref, const int N, bool verbose = false) {
    double epsilon = 1.0E-3;
    int print_num  = std::min(128, N);
    for (int i = 0; i < N; i++) {
        if (abs((float)stand[i] - (float)ref[i]) > epsilon) {
            printf("the matrix does not align at index: %d\n", i);
            for (int j = 0; j < print_num; ++j) {
                printf("stand[%3d]:%f vs ref[%3d]:%f diff: %f\n", j, (float)stand[j], j, (float)ref[j],
                       (float)stand[j] - (float)ref[j]);
            }
            return;
        }
    }
    if (verbose) {
        for (int i = 0; i < print_num; ++i) {
            printf("stand[%3d]:%f vs ref[%3d]:%f diff: %f\n", i, (float)stand[i], i, (float)ref[i],
                   (float)stand[i] - (float)ref[i]);
        }
    }
    printf("Check result success!\n");
}

/**
 * Transpose Matrix
 * Matrix[M, N] -> Matrix[N, M]
 *
 * */
template <typename T>
void Transpose2D(T *matrix, const int M, const int N) {
    assert(matrix != nullptr);
    auto buffer = new float[M * N]();
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            buffer[n * M + m] = matrix[m * N + n];
        }
    }
    memcpy(matrix, buffer, M * N * sizeof(T));
    delete[] buffer;
}
