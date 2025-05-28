#ifndef KMEANS_CUDA_H // Include guard to prevent multiple inclusions
#define KMEANS_CUDA_H 

#ifdef __cplusplus // Use C linkage for C++ compilers
extern "C" { // Prevent name mangling
#endif

void run_kmeans_cuda(float* h_points, int* h_assignments, float* h_centroids, int N, int K, int D, int max_iters); // Declaration for CUDA version

#ifdef __cplusplus // End of C linkage
}
#endif 

#endif 