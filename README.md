# OCP MXFP8 (E4M3) Dot Product

Implementation of dot product in Python using MXFP8 and E8M0 microscaling. Project completed during my internship at Axiomise in August 2026.

## Summary
- Implemented FMA (Fused Multiply-Add) to prevent unnecessary rounding and improve accuracy
- Implemented decoding and conversion between FP32 and E4M3
- Implemented E8M0 scale selection for microscaling across MXFP8 blocks
- Handled edge cases including subnormals, overflow, underflow, zero, NaN, and infinities

## How it works
- `mx_dot_block(A, B)` is the main function, computing the dot product of two vectors (of length 32). It immediately calls `quantize_mx_block()` for both A and B, which selects the E8M0 scale factor and converts each block into E4M3 for arithmetic
- For each pair of elements in A and B, it calls `FMA_multiply_add(a, b, c)`, in order to multiply A[i] and B[i], which is added to the current sum
- The sum is then scaled by the E8M0 scale factor, and the result of the dot product is returned as an FP32

## Remaining work
- Fix remaining implementation issues: `reconstruct_mx_block()` should take the scale as an argument, `e4m3_to_fp32()` needs to account for more edge cases, `choose_scale()` doesn't always assign an exponent, reverse order of scale and quantization to E4M3 in `quantize_mx_block()`
- Implement `mx_dot(A, B)` as the sum of block dot products (extension to arbitrary vector lengths)
- Extensive testing and verification on regular and edge cases including: all ones, zeros, mixed signs, mixed magnitudes and random vectors
- Write up project report, including absolute/relative error with a comparison to NumPy FP32 dot product
