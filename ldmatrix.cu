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
#include <cstdio>
#include <iostream>
#include "cuda_utils.cuh"

__global__ void LdMatrixM8N8X1() {
    __shared__ uint32_t tile[8 * 4];

    uint32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
    if (thread_idx == 0) {
        for (int i = 0; i < 8 * 4; ++i) {
            tile[i] = i;
        }
    }
    __syncthreads();
    uint32_t a[1];
    uint32_t tile_index = thread_idx % 8 * 4;
    uint32_t smem = __cvta_generic_to_shared(tile + tile_index);
    int index = smem / 8;

    asm("ldmatrix.sync.aligned.m8n8.x1.shared.b16 { %0}, [ %1 ];\n" : "=r"(a[0]) : "r"(smem));

    if (true) {
        printf("thread_idx: %3u, load mem ptr:%3u matrix[%3u,%3u]: %3d\n", thread_idx, smem, index, index + 1, a[0]);
    }
}

__global__ void LdMatrixM8N8X2() {
    __shared__ uint32_t tile[2 * 8 * 4];

    uint32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
    if (thread_idx == 0) {
        for (int i = 0; i < 2 * 8 * 4; ++i) {
            tile[i] = i;
        }
    }
    __syncthreads();
    uint32_t a[2];
    // uint32_t tile_index = thread_idx % 16 * 4;
    uint32_t tile_index = thread_idx % 8 * 8 + thread_idx / 8 * 4;
    uint32_t smem = __cvta_generic_to_shared(tile + tile_index);
    uint32_t index = smem / 8;

    asm("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(a[0]), "=r"(a[1]) : "r"(smem));

    if (true) {
        printf("thread_idx: %3u, load mem ptr:%3u matrix[%3u,%3u]: %3d [%3u,%3u]: %3d \n", thread_idx, smem, index,
               index + 1, a[0], index + 64, index + 64 + 1, a[1]);
    }
}

__global__ void LdMatrixM8N8X4() {
    __shared__ uint32_t tile[4 * 8 * 4];

    uint32_t thread_idx = threadIdx.x + blockDim.x * threadIdx.y;
    if (thread_idx == 0) {
        for (int i = 0; i < 4 * 8 * 4; ++i) {
            tile[i] = i;
        }
    }
    __syncthreads();
    uint32_t a[4];
    // uint32_t tile_index = thread_idx % 32 * 4;
    uint32_t tile_index = thread_idx % 16 * 8 + thread_idx / 16 * 4;
    uint32_t smem = __cvta_generic_to_shared(tile + tile_index);
    uint32_t index = smem / 8;
    asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 { %0, %1, %2, %3 }, [ %4 ];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(smem));

    if (true) {
        printf("thread_idx: %3u, load mem ptr:%3u matrix[%3u,%3u]: %3d [%3u,%3u]: %3d [%3u,%3u]: %3d [%3u,%3u]: %3d\n",
               thread_idx, smem, index, index + 1, a[0], index + 64, index + 64 + 1, a[1], index + 64 * 2,
               index + 64 * 2 + 1, a[2], index + 64 * 3, index + 64 * 3 + 1, a[3]);
    }
}

int main(int argc, char* argv[]) {
    uint3 block = {32, 1, 1};
    uint3 grid = {1, 1, 1};
    //    printf("invoke LdMatrixM8N8X1\n");
    //    LdMatrixM8N8X1<<<grid, block>>>();
    //    cudaDeviceSynchronize();
    //    CUDA_CHECK_LAST_ERROR();

    printf("invoke LdMatrixM8N8X2\n");
    LdMatrixM8N8X2<<<grid, block>>>();
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    //    printf("invoke LdMatrixM8N8X4\n");
    //    LdMatrixM8N8X4<<<grid, block>>>();
    //    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR();

    cudaDeviceReset();
    return 0;
}