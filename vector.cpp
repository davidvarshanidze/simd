#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>

const int VECTOR_SIZE = 1000000;

void scalarAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result)
{
    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        result[i] = a[i] + b[i];
    }
}

void simdAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result)
{
    __m256 va, vb, vresult;
    for (int i = 0; i < VECTOR_SIZE; i += 8)
    {
        va = _mm256_loadu_ps(&a[i]);
        vb = _mm256_loadu_ps(&b[i]);
        vresult = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vresult);
    }
}

int main()
{
    std::vector<float> a;
    a.reserve(VECTOR_SIZE);
    std::vector<float> b;
    b.reserve(VECTOR_SIZE);
    std::vector<float> scalarResult;
    scalarResult.reserve(VECTOR_SIZE);
    std::vector<float> simdResult;
    simdResult.reserve(VECTOR_SIZE);

    for (int i = 0; i < VECTOR_SIZE; ++i)
    {
        a.emplace_back(i);
        b.emplace_back(i + 1);
        scalarResult.emplace_back(0);
        simdResult.emplace_back(0);
    }

    auto startScalar = std::chrono::high_resolution_clock::now();
    scalarAdd(a, b, scalarResult);
    auto endScalar = std::chrono::high_resolution_clock::now();
    auto scalarDuration = std::chrono::duration_cast<std::chrono::microseconds>(endScalar - startScalar).count();

    auto startSimd = std::chrono::high_resolution_clock::now();
    simdAdd(a, b, simdResult);
    auto endSimd = std::chrono::high_resolution_clock::now();
    auto simdDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSimd - startSimd).count();

    std::cout << "Scalar Addition Time: " << scalarDuration << " microseconds" << std::endl;
    std::cout << "SIMD Addition Time: " << simdDuration << " microseconds" << std::endl;

    return 0;
}
