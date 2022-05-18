/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"
#include "sampling_gpu.h"

__global__ void gather_points_kernel_fast(int b, int c, int n, int m, 
    const float *__restrict__ points, const int *__restrict__ idx, float *__restrict__ out) {
    // points: (B, C, N)
    // idx: (B, M)
    // output:
    //      out: (B, C, M)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    points += bs_idx * c * n + c_idx * n;
    out[0] = points[idx[0]];
}

void gather_points_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *points, const int *idx, float *out) {
    // points: (B, C, N)
    // idx: (B, npoints)
    // output:
    //      out: (B, C, npoints)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, points, idx, out);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void gather_points_grad_kernel_fast(int b, int c, int n, int m, const float *__restrict__ grad_out, 
    const int *__restrict__ idx, float *__restrict__ grad_points) {
    // grad_out: (B, C, M)
    // idx: (B, M)
    // output:
    //      grad_points: (B, C, N)

    int bs_idx = blockIdx.z;
    int c_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= b || c_idx >= c || pt_idx >= m) return;

    grad_out += bs_idx * c * m + c_idx * m + pt_idx;
    idx += bs_idx * m + pt_idx;
    grad_points += bs_idx * c * n + c_idx * n;

    atomicAdd(grad_points + idx[0], grad_out[0]);
}

void gather_points_grad_kernel_launcher_fast(int b, int c, int n, int npoints, 
    const float *grad_out, const int *idx, float *grad_points) {
    // grad_out: (B, C, npoints)
    // idx: (B, npoints)
    // output:
    //      grad_points: (B, C, N)

    cudaError_t err;
    dim3 blocks(DIVUP(npoints, THREADS_PER_BLOCK), c, b);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);

    gather_points_grad_kernel_fast<<<blocks, threads>>>(b, c, n, npoints, grad_out, idx, grad_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


__device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i, int idx1, int idx2){
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

template <unsigned int block_size>
__global__ void furthest_point_sampling_kernel(int b, int n, int m, 
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, 3)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)
    printf("-----------------------");
    printf("idxs1 is %d\n",m);
    
    printf("dataset is %f\n",dataset[0]);
    printf("temp is %f\n",temp[0]);
    if (m <= 0) return;
    // shared memory can be accessed by all the threads inside the block
    __shared__ float dists[block_size];     // store distances
    __shared__ int dists_i[block_size];     // index?
    printf("idxs2 is %d\n",m);
    int batch_index = blockIdx.x;           // seems that blockIdx.x means the frame idx inside the batch
    dataset += batch_index * n * 3;         // ptr offset, now the ptr dataset point to the first element in the frame
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;                  // tid in (1024/512/256...)
    const int stride = block_size;          // num of threads, may contains 1 / multiple points to be sampled from
    printf("idxs3 is %d\n",threadIdx.x);
    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;                          // the first point to be sampled -> picked the first point in n directly
    printf("idxs4 is %d\n",m);
    printf("idxs4 is %d\n",m);
    // synchronizing... as shared memory accessed by all the threads in the block, make idxs[0] visible to all threads?
    __syncthreads();
    printf("idxs5 is %d\n",m);
    for (int j = 1; j < m; j++) {           // the first point already picked (seed point), loop m-1 to be sampled pts
    printf("idxs6 is %d\n",idxs[j]);
    int besti = 0;                          // best index
    float best = -1;                        // best distance?
    float x1 = dataset[old * 3 + 0];        // get the coordinates of current idx point
    float y1 = dataset[old * 3 + 1];
    float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {     // start from first source point in the thread
        float x2, y2, z2;
        x2 = dataset[k * 3 + 0];                // get the coordinates of the k+stride idx point
        y2 = dataset[k * 3 + 1];
        z2 = dataset[k * 3 + 2];
        // float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
        // if (mag <= 1e-3)
        // continue;

        // TODO try to store the d
        float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d2 = min(d, temp[k]);             // the min distance of other points to the k-th point
        temp[k] = d2;                           // update the min distance of the k-th point
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
    
}

void furthest_point_sampling_kernel_launcher(int64_t b, int64_t n, int64_t m, 
    const float *dataset, float *temp, int64_t *idxs) {
    // dataset: (b, n, 3) float
    // tmp: (b, n) float
    // output:
    //      idx: (b, m)
    printf("=======cuda");
    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);    // get the optimal num of threads? 1-dim structure

    // set the block size in template
    // grid size equal to batch size
    // block size equal to n_threads according to n input num of points
    switch (n_threads) {
        case 1024:
        printf("=======1");
        furthest_point_sampling_kernel<1024><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        printf("=======2");
        furthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        printf("=======3");
        furthest_point_sampling_kernel<256><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        printf("=======4");
        furthest_point_sampling_kernel<128><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        printf("=======5");
        furthest_point_sampling_kernel<64><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        printf("=======6");
        furthest_point_sampling_kernel<32><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        printf("=======7");
        furthest_point_sampling_kernel<16><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        printf("=======9");
        furthest_point_sampling_kernel<8><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        printf("=======10");
        furthest_point_sampling_kernel<4><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        printf("=======8");
        furthest_point_sampling_kernel<2><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        printf("=======11");
        furthest_point_sampling_kernel<1><<<b, n_threads>>>(b, n, m, dataset, temp, idxs); break;
        default:
        printf("=======12");
        furthest_point_sampling_kernel<512><<<b, n_threads>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        printf("=======22");
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}


template <unsigned int block_size>
__global__ void furthest_point_sampling_with_dist_kernel(int b, int n, int m,
    const float *__restrict__ dataset, float *__restrict__ temp, int *__restrict__ idxs) {
    // dataset: (B, N, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    if (m <= 0) return;
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int batch_index = blockIdx.x;
    dataset += batch_index * n * n;
    temp += batch_index * n;
    idxs += batch_index * m;

    int tid = threadIdx.x;
    const int stride = block_size;

    int old = 0;
    if (threadIdx.x == 0)
    idxs[0] = old;

    __syncthreads();
    for (int j = 1; j < m; j++) {
    int besti = 0;
    float best = -1;
    // float x1 = dataset[old * 3 + 0];
    // float y1 = dataset[old * 3 + 1];
    // float z1 = dataset[old * 3 + 2];
    for (int k = tid; k < n; k += stride) {
        // float x2, y2, z2;
        // x2 = dataset[k * 3 + 0];
        // y2 = dataset[k * 3 + 1];
        // z2 = dataset[k * 3 + 2];

        // float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
        float d = dataset[old * n + k];

        float d2 = min(d, temp[k]);
        temp[k] = d2;
        besti = d2 > best ? k : besti;
        best = d2 > best ? d2 : best;
    }
    dists[tid] = best;
    dists_i[tid] = besti;
    __syncthreads();

    if (block_size >= 1024) {
        if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
        }
        __syncthreads();
    }

    if (block_size >= 512) {
        if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
        }
        __syncthreads();
    }
    if (block_size >= 64) {
        if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
        }
        __syncthreads();
    }

    old = dists_i[0];
    if (tid == 0)
        idxs[j] = old;
    }
}


void furthest_point_sampling_with_dist_kernel_launcher(int b, int n, int m,
    const float *dataset, float *temp, int *idxs, cudaStream_t stream) {
    // dataset: (B, N, N)
    // tmp: (B, N)
    // output:
    //      idx: (B, M)

    cudaError_t err;
    unsigned int n_threads = opt_n_threads(n);

    switch (n_threads) {
        case 1024:
        furthest_point_sampling_with_dist_kernel<1024><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 512:
        furthest_point_sampling_with_dist_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 256:
        furthest_point_sampling_with_dist_kernel<256><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 128:
        furthest_point_sampling_with_dist_kernel<128><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 64:
        furthest_point_sampling_with_dist_kernel<64><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 32:
        furthest_point_sampling_with_dist_kernel<32><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 16:
        furthest_point_sampling_with_dist_kernel<16><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 8:
        furthest_point_sampling_with_dist_kernel<8><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 4:
        furthest_point_sampling_with_dist_kernel<4><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 2:
        furthest_point_sampling_with_dist_kernel<2><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        case 1:
        furthest_point_sampling_with_dist_kernel<1><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs); break;
        default:
        furthest_point_sampling_with_dist_kernel<512><<<b, n_threads, 0, stream>>>(b, n, m, dataset, temp, idxs);
    }

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
