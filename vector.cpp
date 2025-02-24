#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>

const int VECTOR_SIZE = 1000000;

void scalarAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result) {
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result[i] = a[i] + b[i];
    }
}

void simdAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result) {
    __m256 va, vb, vresult;
    for (int i = 0; i < VECTOR_SIZE; i += 8) {
        va = _mm256_loadu_ps(&a[i]); // Unaligned load
        vb = _mm256_loadu_ps(&b[i]);
        vresult = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vresult);
    }
}

int main() {
    std::vector<float> a(VECTOR_SIZE);
    std::vector<float> b(VECTOR_SIZE);
    std::vector<float> scalarResult(VECTOR_SIZE);
    std::vector<float> simdResult(VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; ++i) {
        a[i] = i;
        b[i] = i + 1;
    }

    auto startScalar = std::chrono::steady_clock::now();
    scalarAdd(a, b, scalarResult);
    auto endScalar = std::chrono::steady_clock::now();
    auto scalarDuration = std::chrono::duration_cast<std::chrono::microseconds>(endScalar - startScalar).count();

    auto startSimd = std::chrono::steady_clock::now();
    simdAdd(a, b, simdResult);
    auto endSimd = std::chrono::steady_clock::now();
    auto simdDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSimd - startSimd).count();

    std::cout << "Scalar Addition Time: " << scalarDuration << " microseconds" << std::endl;
    std::cout << "SIMD Addition Time: " << simdDuration << " microseconds" << std::endl;

    std::cout << "First scalar result: " << scalarResult[0] << std::endl;
    std::cout << "First SIMD result: " << simdResult[0] << std::endl;

    return 0;
}
