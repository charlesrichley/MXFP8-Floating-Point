from e4m3 import decode_e4m3, construct_e4m3, quantise_e4m3, e4m3_to_fp32, find_e4m3_sign, return_sign_exp_mant
from fp32 import decode_fp32, is_greater_absolute_fp32
from functools import cmp_to_key

# E8M0 - 8 bit exponent 0 bit mantissa, with bias = 127
# MXFP8 microscaling 8 bit floating point w/ single E8M0 scale factor
# 32 values of E4M3 (block-wise scaling) for computing dot product

def shift_with_negatives(x: int, scale: int, direction='left') -> int:
    # Default is a left shift
    if direction == 'left':
        if scale > 0:
            x <<= scale
        else: # scale > 0 
            x >>= abs(scale)

    elif direction == 'right':
        if scale > 0:
            x >>= scale
        else:
            x <<= abs(scale)
    return x

def choose_scale(values: list[int]) -> int:
    # Values is an array (block) of 32 FP32 numbers
    # We aim to maximise e8m0, and abs(e4m3) <= 448

    # Sort the array so we know the largest
    sorted_values = sorted(values, key=cmp_to_key(is_greater_absolute_fp32))

    # Last element in the sorted array is the maximum
    absolute_max = decode_fp32(sorted_values[-1])

    max_exp = absolute_max["exponent"]
    max_mantissa = absolute_max["mantissa"]

    # NORMAL
    if absolute_max["subnormal"] == 0:
        # From setting 1.f x 2^e / 2^s ≤ 448, where 2^s is the scale factor
        # We find that 1.f x 2^(e-8) ≤ 1.75 x 2^s
        if max_mantissa > 0b1110000000000000000000000: # 1.75
            scale_exponent = max_exp - 8 - 1
        else:
            scale_exponent = max_exp - 8

    # SUBNORMAL
    else:
        # From setting up a similar inequality
        # We get 0.f x 2^(-s) ≤ 1.75 x 2^(134)
        # So M x 2^(-s) ≤ 7 x 2^(155)
        # E8M0 ranges from -127 to +127, so we use a simple for loop
        for s in range(-127, 127 + 1):
            if shift_with_negatives(max_mantissa, s, direction='right') > (7 * (2**155)):
                scale_exponent = s - 1
                break

    # Has bias 127
    return scale_exponent + 127

# FP32 to MXFP8 conversion
def quantize_mx_block(values: list[int]) -> tuple[int, int]:
    # Argument is values, array of FP32 numbers

    scale_exp = choose_scale(values) - 127

    e4m3_values = []
    for value in values:
        value_e4m3 = quantise_e4m3(value)

        value_e4m3_sign = (value_e4m3 >> 7) & 0b1
        value_e4m3_exp = (value_e4m3 >> 3) & 0b1111
        value_e4m3_mantissa = value_e4m3 & 0b111

        # Check value is non zero
        if value_e4m3_exp == 0b0000 and value_e4m3_mantissa == 0b000:
             pass
        else:
            value_e4m3_exp -= scale_exp

        e4m3_values.append(construct_e4m3(value_e4m3_sign, value_e4m3_exp, value_e4m3_mantissa))

    return (e4m3_values, scale_exp) # meaning MXFP8

def reconstruct_mx_block(values: list[int]) -> list[int]:
    # Convert from E4M3 back to FP32
    FP32_values = []
    for value in values:
        FP32_values.append(e4m3_to_fp32(value))

    return FP32_values

# Normalises after FMA
def normalise(x: int) -> tuple[int, int]:
    x_original = abs(x)
    x = x_original
    count = 0
    while True:
        if (x == 1):
            break
        x >>= 1
        count += 1

    mantissa_bits = shift_with_negatives(x_original, (count - 3), direction='right') & 0b111
    third_bit = mantissa_bits & 0b1
    if count < 4:
        guard_bit = 0 
        rest_of_bits = 0
    else:
        guard_bit = (x_original >> (count - 4)) & 0b1
        rest_of_bits = x_original & (2**(count - 4) - 1)

    # Round to nearest, ties to even (0)
    # If exactly halfway, rounds to one whose lowest bit is 0, prevents upward drift
    if guard_bit == 1 and rest_of_bits == 0:
        # Round to nearest, ties to even

        # If third_bit = 0, leave it, else mantissa += 1
        if third_bit == 1:
            mantissa_bits += 1

    elif guard_bit == 1:
        # More than half, so round up
        if rest_of_bits != 0:
            mantissa_bits += 1

    # Check for overflow
    if mantissa_bits > 0b111:
        mantissa_bits = 0
        count += 1

    return (count, mantissa_bits)

# FMA (Fused Multiply-Add) implementation - only one rounding step for optimisation
def FMA_multiply_add(a: int, b: int, c: int) -> int:
    # c represents the previous sum, which is E4M3
    # a and b are E4M3 numbers from the dot product of vectors
    a_dict = decode_e4m3(a)
    b_dict = decode_e4m3(b)
    c_dict = decode_e4m3(c)

    # Check for NANs - if there is a NAN, return NAN
    if a_dict["NAN"] == 1 or b_dict["NAN"] == 1 or c_dict["NAN"] == 1:
        return construct_e4m3(0b0, 0b1111, 0b111)

    a_sign, a_exponent, a_mantissa = return_sign_exp_mant(a_dict)
    b_sign, b_exponent, b_mantissa = return_sign_exp_mant(b_dict)
    c_sign, c_exponent, c_mantissa = return_sign_exp_mant(c_dict)

    # Check that a and b are non zero, else just return c
    if (a_exponent == 0 and a_mantissa == 0) or (b_exponent == 0 and b_mantissa == 0):
        return c

    # Now a x b is higher precision than E4M3 to prevent unnecessary rounding
    ab_sign = a_sign ^ b_sign # XOR so sign is negative only when a,b have different signs, else +ve
    ab_exponent = a_exponent + b_exponent
    ab_mantissa = a_mantissa * b_mantissa

    # Get signs for arithmetic
    if ab_sign == 0:
        ab_sign = 1
    else:
        ab_sign = -1

    if c_sign == 0:
        c_sign = 1
    else:
        c_sign = -1

    # The mantissa is now divided by (2^3)^2 = 2^6
    ab_exp = ab_exponent - 6
    c_exp = c_exponent - 3

    # shift_factor represents a left shift (ie multiplying by 2^shift_factor)
    if ab_exp > c_exp:
        num, shift_factor = (((ab_mantissa << (ab_exp - c_exp)) * ab_sign + c_mantissa * c_sign), c_exp)
    else:
        num, shift_factor = (((c_mantissa << (c_exp - ab_exp)) * c_sign + ab_mantissa * ab_sign), ab_exp)

    # Check that if num is 0, just return a 0 (in E4M3)
    if num == 0:
        E4M3_sign = 0b0
        E4M3_exp = 0b0000
        E4M3_mantissa = 0b000

    else:
        E4M3_sign = find_e4m3_sign(num)
        power, E4M3_mantissa = normalise(num)

        E4M3_exp = power + shift_factor + 7 # add 7 for bias
            
        # OVERFLOW
        if E4M3_exp > 15:
            # Set to maximum value
            E4M3_exp = 0b1111
            E4M3_mantissa = 0b110

        # SUBNORMAL RESULT
        elif E4M3_exp < 1:
            # Mantissa = num x 2^(shift_factor + 9), equivalent to left shift
            E4M3_mantissa = shift_with_negatives(num, shift_factor + 9) & 0b111
            E4M3_exp = 0 # standard for subnormals

    return construct_e4m3(E4M3_sign, E4M3_exp, E4M3_mantissa)

def mx_dot_block(A: list[int], B: list[int]) -> int:
    # Dot product between vectors of length 32 (FP32)
    # Scale factors taken outside the sum

    A_e4m3, A_scale = quantize_mx_block(A)
    B_e4m3, B_scale = quantize_mx_block(B)

    dot_E4M3 = 0
    for i in range(32):
        dot_E4M3 = FMA_multiply_add(A_e4m3[i], B_e4m3[i], dot_E4M3)

    if dot_E4M3 & ((2**7) - 1) == 0:
        return 0

    # Convert to FP32, then multiply by 2^(A_scale + B_scale)
    dot_FP32 = e4m3_to_fp32(dot_E4M3)
    dot_sign_FP32 = dot_FP32 & (2**31)
    dot_mantissa_FP32 = dot_FP32 & ((2**23) - 1)

    # Adjust the exponent and then shift back
    dot_exp_FP32 = (dot_FP32 >> (31 - 8)) & ((2**8) - 1)
    dot_exp_FP32 += (A_scale + B_scale)
    dot_exp_FP32 <<= (31 - 8)

    return dot_sign_FP32 | dot_exp_FP32 | dot_mantissa_FP32


# <----- TESTING ----->
A = [0x3F800000] + [0x00000000] * 31
B = [0x00000000, 0x3F800000] + [0x00000000] * 30

result = mx_dot_block(A, B)
print(bin(result))
