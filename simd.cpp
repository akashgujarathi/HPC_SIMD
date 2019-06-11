#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>

const int N = 16*1'000'000;

double
time(const std::function<void ()> &f) {
    f(); // Run once to warmup.
    // Now time it for real.
    auto start = std::chrono::system_clock::now();
    f();
    auto stop = std::chrono::system_clock::now();
    return std::chrono::duration<double>(stop - start).count();
}

int
main() {

    alignas(32) static float w[N], x[N], y[N], z[N];
    alignas(32) static float w_1[N], x_1[N], y_1[N], z_1[N];

    /*
     * Generate data.
     */

    std::default_random_engine eng;
    std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < N; i++) {
        w[i] = dist(eng);
        x[i] = dist(eng);
        y[i] = dist(eng);
        z[i] = dist(eng);

        w_1[i] = dist(eng);
        x_1[i] = dist(eng);
        y_1[i] = dist(eng);
        z_1[i] = dist(eng);

    }

    /*
     * Sequential.
     */

    static float l_s[N];
    auto seq = [&]() {
        for (int i = 0; i < N; i++) {
            //l_s[i] = std::sqrt( pow((w[i]-w_1[i]),2) + pow((x[i]-x_1[i]),2) + pow((y[i]-y_1[i]),2) + pow((z[i]-z_1[i]),2) );
            l_s[i] = std::sqrt(   (w[i]-w_1[i]) * (w[i]-w_1[i])  + 
                                  (x[i]-x_1[i]) * (x[i]-x_1[i])  + 
                                  (y[i]-y_1[i]) * (y[i]-y_1[i])  + 
                                  (z[i]-z_1[i]) * (z[i]-z_1[i]))  ;
        }
    };

    std::cout << "Sequential: " << (N/time(seq))/1000000 << " Mops/s" << std::endl;

    alignas(32) static float l_v[N];
    auto vec = [&]() {
        for (int i = 0; i < N/8; i++) {
            __m256 ymm_w = _mm256_load_ps(w + 8*i);
            __m256 ymm_x = _mm256_load_ps(x + 8*i);
            __m256 ymm_y = _mm256_load_ps(y + 8*i);
            __m256 ymm_z = _mm256_load_ps(z + 8*i);

            __m256 ymm_w_1 = _mm256_load_ps(w_1 + 8*i);
            __m256 ymm_x_1 = _mm256_load_ps(x_1 + 8*i);
            __m256 ymm_y_1 = _mm256_load_ps(y_1 + 8*i);
            __m256 ymm_z_1 = _mm256_load_ps(z_1 + 8*i);

            __m256 mm_l = _mm256_sqrt_ps(
                _mm256_mul_ps(ymm_w - ymm_w_1, ymm_w - ymm_w_1) +
                _mm256_mul_ps(ymm_x - ymm_x_1, ymm_x - ymm_x_1) +
                _mm256_mul_ps(ymm_y - ymm_y_1, ymm_y - ymm_y_1) +
                _mm256_mul_ps(ymm_z - ymm_z_1, ymm_z - ymm_z_1));

            _mm256_store_ps(l_v + 8*i, mm_l);
        }
    };

    std::cout << "Vector: " << (N/time(vec))/1000000 << " Mops/s" << std::endl;

    for (int i = 0; i < N; i++) {
        if (l_s[i] - l_v[i] != 0) {
            assert(false);
        }
    }
}
