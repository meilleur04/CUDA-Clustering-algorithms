#include <cmath>
#include <vector>
#include <limits>
#include <iostream>

void run_kmeans_cpu(float* points, int* assignments, float* centroids, int N, int K, int D, int max_iters) {
    if (D != 2) {
        std::cerr << "Only 2D data supported in this CPU implementation." << std::endl;
        return;
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        for (int i = 0; i < N; ++i) {
            float px = points[2 * i];
            float py = points[2 * i + 1];
            float min_dist = std::numeric_limits<float>::max();
            int cluster = 0;

            for (int c = 0; c < K; ++c) {
                float cx = centroids[2 * c];
                float cy = centroids[2 * c + 1];
                float dist = (px - cx)*(px - cx) + (py - cy)*(py - cy);
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = c;
                }
            }
            assignments[i] = cluster;
        }

        float sum_x[100] = {0}, sum_y[100] = {0};
        int count[100] = {0};

        for (int i = 0; i < N; ++i) {
            int c = assignments[i];
            sum_x[c] += points[2 * i];
            sum_y[c] += points[2 * i + 1];
            count[c]++;
        }

        for (int c = 0; c < K; ++c) {
            if (count[c] == 0) continue;
            centroids[2 * c] = sum_x[c] / count[c];
            centroids[2 * c + 1] = sum_y[c] / count[c];
        }
    }
}
