#ifndef GMM_CUDA_H
#define GMM_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

// Function to run Gaussian Mixture Model on the GPU
void run_gmm_cuda(const float* h_points, float* h_responsibilities, float* h_means,
                  float* h_covariances, float* h_weights, int N, int K, int D, int max_iters);

#ifdef __cplusplus
}
#endif

#endif // GMM_CUDA_H