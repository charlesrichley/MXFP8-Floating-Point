# OCP MXFP8 (E4M3) Dot Product

Implementation of dot product in Python using MXFP8 and E8M0 microscaling. Project completed during my internship at Axiomise in August 2026.

## Summary
- Implemented FMA (Fused Multiply-Add) to prevent unnecessary rounding and improve accuracy
- Implemented decoding and conversion between FP32 and E4M3
- Implemented E8M0 scale selection for microscaling across MXFP8 blocks
- Handled edge cases including subnormals, overflow, underflow, zero, NaN, and infinities

## How it works
- mx_dot_block(A,B) is the main function, computing the dot product of two vectors (of length 32). It immediately calls quantize_mx_block for both A and B, which allows us to find the E8M0 scale factor, and convert each block into E4M3 for arithmetic
- For each element in the A,B arrays, it calls FMA_multiply_add, to multiply A[i] and B[i], added to the current sum
- The sum is then scaled by the E8M0 scale factor ,and the result of the dot product is then returned as an FP32

## Remaining work
- Implement mx_dot(A,B) as the sum of block dot products (extension to arbitrary vector lengths)
- Extensive testing on edge cases including (all ones, zeros, mixed signs, mixed magnitudes and random vectors)
- Report of project, including absolute/relative error with a comparison to NumPy FP32 dot product.
