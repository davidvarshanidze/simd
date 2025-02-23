#include <emmintrin.h>
#include <iostream>

int main() {
    alignas(16) float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    // loading arrays that is declared above into the registers
    __m128 vecA = _mm_load_ps(a);
    __m128 vecB = _mm_load_ps(b);

    // SIMD addition
    __m128 result = _mm_add_ps(vecA, vecB);

    // store "result" back in the array
    alignas(16) float res[4];
    _mm_store_ps(res, result);

    // print results
    std::cout << "results: ";
    for (float f : res) {
        std::cout << f << " ";
    }
    std::cout << std::endl;
    return 0;
}