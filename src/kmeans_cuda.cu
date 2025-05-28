#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void assign_clusters(const float* __restrict__ points, const float* __restrict__ centroids,int* assignments,int N, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int best_cluster = 0;
    float best_dist = FLT_MAX;

    for (int c = 0; c < K; c++) {
        float dist = 0.f;
        for (int d = 0; d < D; d++) {
            float diff = points[idx * D + d] - centroids[c * D + d];
            dist += diff * diff;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_cluster = c;
        }
    }
    assignments[idx] = best_cluster;
}

__global__ void update_sums(const float* __restrict__ points,const int* __restrict__ assignments,float* sums,  int* counts,int N, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int cluster = assignments[idx];
    atomicAdd(&counts[cluster], 1);
    for (int d = 0; d < D; d++) {
        atomicAdd(&sums[cluster * D + d], points[idx * D + d]);
    }
}

__global__ void finalize_centroids(float* centroids, float* sums, int* counts, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= K) return;

    int count = counts[idx];
    if (count > 0) {
        for (int d = 0; d < D; d++) {
            centroids[idx * D + d] = sums[idx * D + d] / count;
        }
    }
}

extern "C" void run_kmeans_cuda(float* h_points, int* h_assignments, float* h_centroids, int N, int K, int D, int max_iters) {

    float *d_points, *d_centroids, *d_sums;
    int *d_assignments, *d_counts;

    cudaMalloc(&d_points, sizeof(float) * N * D);
    cudaMalloc(&d_centroids, sizeof(float) * K * D);
    cudaMalloc(&d_assignments, sizeof(int) * N);
    cudaMalloc(&d_sums, sizeof(float) * K * D);
    cudaMalloc(&d_counts, sizeof(int) * K);

    cudaMemcpy(d_points, h_points, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, sizeof(float) * K * D, cudaMemcpyHostToDevice);

    int numBlocksPoints = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int numBlocksCentroids = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int iter = 0; iter < max_iters; iter++) {
        assign_clusters<<<numBlocksPoints, BLOCK_SIZE>>>(d_points, d_centroids, d_assignments, N, K, D);
        cudaDeviceSynchronize();
        cudaMemset(d_sums, 0, sizeof(float) * K * D);
        cudaMemset(d_counts, 0, sizeof(int) * K);
        update_sums<<<numBlocksPoints, BLOCK_SIZE>>>(d_points, d_assignments, d_sums, d_counts, N, K, D);
        cudaDeviceSynchronize();
        finalize_centroids<<<numBlocksCentroids, BLOCK_SIZE>>>(d_centroids, d_sums, d_counts, K, D);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_assignments, d_assignments, sizeof(int) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_centroids, d_centroids, sizeof(float) * K * D, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_sums);
    cudaFree(d_counts);
}
