#include <immintrin.h>
#include <iostream>
#include <vector>
#include <chrono>

const int MATRIX_ROWS = 1000;
const int MATRIX_COLS = 1000;

void scalarMatrixAdd(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, std::vector<std::vector<float>> &result)
{
    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; ++j)
        {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

void simdMatrixAdd(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b, std::vector<std::vector<float>> &result)
{
    __m256 va, vb, vresult;
    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; j += 8)
        {
            va = _mm256_loadu_ps(&a[i][j]);
            vb = _mm256_loadu_ps(&b[i][j]);
            vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&result[i][j], vresult);
        }
    }
}

int main()
{
    std::vector<std::vector<float>> a(MATRIX_ROWS, std::vector<float>(MATRIX_COLS));
    std::vector<std::vector<float>> b(MATRIX_ROWS, std::vector<float>(MATRIX_COLS));
    std::vector<std::vector<float>> scalarResult(MATRIX_ROWS, std::vector<float>(MATRIX_COLS));
    std::vector<std::vector<float>> simdResult(MATRIX_ROWS, std::vector<float>(MATRIX_COLS));

    for (int i = 0; i < MATRIX_ROWS; ++i)
    {
        for (int j = 0; j < MATRIX_COLS; ++j)
        {
            a[i][j] = i + j;
            b[i][j] = i + j + 1;
            scalarResult[i][j] = 0;
            simdResult[i][j] = 0;
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
