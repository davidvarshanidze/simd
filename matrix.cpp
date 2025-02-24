#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>

const int MATRIX_ROWS = 1000;
const int MATRIX_COLS = 1000;

void scalarMatrixAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result)
{
    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; ++j)
        {
            int index = i * MATRIX_COLS + j;
            result[index] = a[index] + b[index];
        }
    }
}

void simdMatrixAdd(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &result)
{
    __m256 va, vb, vresult;
    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; j += 8)
        {
            int index = i * MATRIX_COLS + j;
            va = _mm256_load_ps(&a[index]); // Aligned load
            vb = _mm256_load_ps(&b[index]);
            vresult = _mm256_add_ps(va, vb);
            _mm256_store_ps(&result[index], vresult); // Aligned store
        }
    }
}

int main()
{
    alignas(32) std::vector<float> a(MATRIX_ROWS * MATRIX_COLS);
    alignas(32) std::vector<float> b(MATRIX_ROWS * MATRIX_COLS);
    alignas(32) std::vector<float> scalarResult(MATRIX_ROWS * MATRIX_COLS);
    alignas(32) std::vector<float> simdResult(MATRIX_ROWS * MATRIX_COLS);

    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; ++j)
        {
            int index = i * MATRIX_COLS + j;
            a[index] = i + j;
            b[index] = i + j + 1;
        }
    }

    auto startScalar = std::chrono::high_resolution_clock::now();
    scalarMatrixAdd(a, b, scalarResult);
    auto endScalar = std::chrono::high_resolution_clock::now();
    auto scalarDuration = std::chrono::duration_cast<std::chrono::microseconds>(endScalar - startScalar).count();

    auto startSimd = std::chrono::high_resolution_clock::now();
    simdMatrixAdd(a, b, simdResult);
    auto endSimd = std::chrono::high_resolution_clock::now();
    auto simdDuration = std::chrono::duration_cast<std::chrono::microseconds>(endSimd - startSimd).count();

    std::cout << "Scalar Matrix Addition Time: " << scalarDuration << " microseconds" << std::endl;
    std::cout << "SIMD Matrix Addition Time: " << simdDuration << " microseconds" << std::endl;

    return 0;
}
