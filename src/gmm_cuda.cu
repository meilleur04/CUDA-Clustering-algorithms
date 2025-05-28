#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#define BLOCK_SIZE 256
#define PI 3.14159265358979323846f

__device__ float calculate_pdf_device(const float* point, const float* mean, const float* covariance, int D) {
    float det_cov = 1.0f;
    for (int d = 0; d < D; ++d) {
        det_cov *= covariance[d];
    }
    if (det_cov <= 0.0f) return 0.0f;

    float exponent = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = point[d] - mean[d];
        exponent += (diff * diff) / covariance[d];
    }

    float coeff = 1.0f / sqrtf(powf(2.0f * PI, D) * det_cov);
    return coeff * expf(-0.5f * exponent);
}

__global__ void e_step_kernel(const float* points, float* responsibilities, const float* means,const float* covariances, const float* weights, int N, int K, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float total_prob = 0.0f;
    for (int k = 0; k < K; k++) {
        float prob = weights[k] * calculate_pdf_device(&points[idx * D], &means[k * D], &covariances[k * D], D);
        responsibilities[idx * K + k] = prob;
        total_prob += prob;
    }

    if (total_prob > 0) {
        for (int k = 0; k < K; k++) {
            responsibilities[idx * K + k] /= total_prob;
        }
    }
}

__global__ void m_step_update_kernel(const float* points, const float* responsibilities, float* means, float* covariances, float* weights, float* sum_resp, int N, int K, int D) {
    int k = blockIdx.x; 
    int tid = threadIdx.x;

    extern __shared__ float s_sum_resp[];
    float* s_means = &s_sum_resp[K];
    float* s_covariances = &s_means[K * D];

    if (tid < K) s_sum_resp[tid] = 0;
    if (tid < K*D) s_means[tid] = 0;
    if (tid < K*D) s_covariances[tid] = 0;
    __syncthreads();

    float local_sum_resp = 0.0;
    float local_sum_means[10] = {0};
    
    for (int i = tid; i < N; i += blockDim.x) {
        float resp = responsibilities[i * K + k];
        local_sum_resp += resp;
        for (int d = 0; d < D; d++) {
            local_sum_means[d] += resp * points[i * D + d];
        }
    }
    
    atomicAdd(&s_sum_resp[k], local_sum_resp);
    for (int d = 0; d < D; d++) {
        atomicAdd(&s_means[k * D + d], local_sum_means[d]);
    }
    __syncthreads();

    if (tid == 0) {
        sum_resp[k] = s_sum_resp[k];
        if (sum_resp[k] > 0) {
            weights[k] = sum_resp[k] / N;
            for (int d = 0; d < D; d++) {
                means[k * D + d] = s_means[k * D + d] / sum_resp[k];
            }
        }
    }
     __syncthreads();

    float local_sum_cov[10] = {0}; 
    for(int i = tid; i < N; i+= blockDim.x) {
        float resp = responsibilities[i * K + k];
        for(int d=0; d < D; d++) {
            float diff = points[i*D + d] - means[k*D + d];
            local_sum_cov[d] += resp * diff * diff;
        }
    }

    for (int d = 0; d < D; d++) {
        atomicAdd(&s_covariances[k * D + d], local_sum_cov[d]);
    }
    __syncthreads();

    if (tid == 0 && sum_resp[k] > 0) {
        for (int d = 0; d < D; d++) {
            covariances[k * D + d] = s_covariances[k * D + d] / sum_resp[k] + 1e-6f; // Add epsilon
        }
    }
}


extern "C" void run_gmm_cuda(const float* h_points, float* h_responsibilities, float* h_means,float* h_covariances, float* h_weights, int N, int K, int D, int max_iters) {
    
    float *d_points, *d_responsibilities, *d_means, *d_covariances, *d_weights, *d_sum_resp;
    
    cudaMalloc(&d_points, sizeof(float) * N * D);
    cudaMalloc(&d_responsibilities, sizeof(float) * N * K);
    cudaMalloc(&d_means, sizeof(float) * K * D);
    cudaMalloc(&d_covariances, sizeof(float) * K * D);
    cudaMalloc(&d_weights, sizeof(float) * K);
    cudaMalloc(&d_sum_resp, sizeof(float) * K);

    cudaMemcpy(d_points, h_points, sizeof(float) * N * D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_means, h_means, sizeof(float) * K * D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_covariances, h_covariances, sizeof(float) * K * D, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeof(float) * K, cudaMemcpyHostToDevice);

    int numBlocksPoints = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        e_step_kernel<<<numBlocksPoints, BLOCK_SIZE>>>(d_points, d_responsibilities, d_means, d_covariances, d_weights, N, K, D);
        cudaDeviceSynchronize();

        size_t shared_mem_size = (K + 2 * K * D) * sizeof(float);
        m_step_update_kernel<<<K, BLOCK_SIZE, shared_mem_size>>>(d_points, d_responsibilities, d_means, d_covariances, d_weights, d_sum_resp, N, K, D);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_responsibilities, d_responsibilities, sizeof(float) * N * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_means, d_means, sizeof(float) * K * D, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_covariances, h_covariances, sizeof(float) * K * D, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_weights, h_weights, sizeof(float) * K, cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_responsibilities);
    cudaFree(d_means);
    cudaFree(d_covariances);
    cudaFree(d_weights);
    cudaFree(d_sum_resp);
}