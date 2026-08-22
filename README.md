# OCP MXFP8 (E4M3) Dot Product

Implementation of dot product in Python using MXFP8 and E8M0 microscaling

### Summary
- Implemented FMA (Fused Multiply-Add) to prevent unnecessary rounding and improve accuracy
- Implemented decoding and conversion between FP32 and E4M3
- Implemented choose scale function for E8M0 microscaling across MXFP8 blocks
- Handled edge cases including subnormals, overflow, underflow, zero, NAN, and infinities.
