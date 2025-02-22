#include <iostream>
#include <immintrin.h>  // For SIMD intrinsics

int main() {
    // Create two arrays of 4 elements
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    float result[4];

    // Load the arrays into SIMD registers
    __m128 vec_a = _mm_loadu_ps(a);  // Load 'a' into a 128-bit register
    __m128 vec_b = _mm_loadu_ps(b);  // Load 'b' into a 128-bit register

    // Multiply 'a' by 4 and add 'b'
    __m128 vec_4a = _mm_mul_ps(vec_a, _mm_set1_ps(4.0f));  // Multiply each element of 'a' by 4
    __m128 vec_result = _mm_add_ps(vec_4a, vec_b);  // Add 'b' to each element of '4a'

    // Store the result back into the result array
    _mm_storeu_ps(result, vec_result);

    // Print the result
    std::cout << "Result: ";
    for (int i = 0; i < 4; ++i) {
        std::cout << result[i] << " ";
    }

    return 0;
}
