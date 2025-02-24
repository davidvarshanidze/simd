#include <iostream>
#include <immintrin.h>

int main() {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];

    __m128 vec_a = _mm_loadu_ps(a);
    __m128 vec_b = _mm_loadu_ps(b);

    __m128 vec_4a = _mm_mul_ps(vec_a, _mm_set1_ps(4.0f));
    __m128 vec_result = _mm_add_ps(vec_4a, vec_b);

    _mm_storeu_ps(result, vec_result);

    std::cout << "Result: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << result[i] << " ";
    }

    return 0;
}
