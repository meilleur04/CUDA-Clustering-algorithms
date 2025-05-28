#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

const float PI = 3.14159265358979323846;

double calculate_pdf(const float* point, const float* mean, const float* covariance, int D) {
    double det_cov = 1.0;
    for (int d = 0; d < D; ++d) {
        det_cov *= covariance[d];
    }

    double exponent = 0.0;
    for (int d = 0; d < D; ++d) {
        float diff = point[d] - mean[d];
        exponent += (diff * diff) / covariance[d];
    }

    double coeff = 1.0 / sqrt(pow(2 * PI, D) * det_cov);
    return coeff * exp(-0.5 * exponent);
}

void run_gmm_cpu(const float* points, float* responsibilities, float* means, float* covariances, float* weights, int N, int K, int D, int max_iters) {

    std::vector<double> responsibility_sum(K);
    std::vector<double> temp_means(K * D);
    std::vector<double> temp_covariances(K * D);

    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < N; ++i) {
            double total_prob = 0.0;
            for (int k = 0; k < K; ++k) {
                double prob = weights[k] * calculate_pdf(&points[i * D], &means[k * D], &covariances[k * D], D);
                responsibilities[i * K + k] = prob;
                total_prob += prob;
            }
            if (total_prob > 0) {
                for (int k = 0; k < K; ++k) {
                    responsibilities[i * K + k] /= total_prob;
                }
            }
        }

        std::fill(responsibility_sum.begin(), responsibility_sum.end(), 0.0);
        std::fill(temp_means.begin(), temp_means.end(), 0.0);
        std::fill(temp_covariances.begin(), temp_covariances.end(), 0.0);

        for (int k = 0; k < K; ++k) {
            for (int i = 0; i < N; ++i) {
                float resp = responsibilities[i * K + k];
                responsibility_sum[k] += resp;
                for (int d = 0; d < D; ++d) {
                    temp_means[k * D + d] += resp * points[i * D + d];
                }
            }
        }

        for (int k = 0; k < K; ++k) {
            if (responsibility_sum[k] > 0) {
                weights[k] = responsibility_sum[k] / N;
                for (int d = 0; d < D; ++d) {
                    means[k * D + d] = temp_means[k * D + d] / responsibility_sum[k];
                }
                for (int i = 0; i < N; ++i) {
                    float resp = responsibilities[i * K + k];
                    for (int d = 0; d < D; ++d) {
                        float diff = points[i * D + d] - means[k * D + d];
                        temp_covariances[k * D + d] += resp * diff * diff;
                    }
                }
                for (int d = 0; d < D; ++d) {
                    covariances[k * D + d] = temp_covariances[k * D + d] / responsibility_sum[k] + 1e-6; 
                }
            }
        }
    }
}