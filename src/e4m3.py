def decode_e4m3(x: int) -> dict:
    # E4M3: 8 bit input (1 sign, 4 exponent, 3 mantissa)
    # NAN exists, but range extended by excluding infinity

    sign = x >> 7  & 0b1
    exponent = (x >> 3) & 0b1111
    mantissa = x & 0b111
    is_NAN = 0
    is_subnormal = 0

    # Checking for NAN
    if mantissa == 0b111 and exponent == 0b1111:
        is_NAN = 1

    # Checking for subnormal number - implicit 1 becomes a 0
    elif exponent == 0b0000:
        # ((-1)^sign) x (2 ^(1-6)) x (M/8)
        is_subnormal = 1
        exponent = 1 - 7

    # Regular finite number
    else:
        # (-1)^sign x 2^(E-bias) x (1 + M/8), E = exponent, M = mantissa
        mantissa += (2**3) # add 8 (2^3) to account for implicit 1
        exponent -= 7

    return_dict = {
            "sign": sign,
            "exponent": exponent,
            "mantissa": mantissa,
            "NAN": is_NAN,
            "subnormal": is_subnormal
        }
    
    return return_dict

def construct_e4m3(sign: int, exponent: int, mantissa: int) -> int:
    shifted_sign = sign << 7
    shifted_exponent = exponent << 3
    return shifted_sign | shifted_exponent | mantissa

def quantise_e4m3(x: int) -> int:
    # FP32 -> E4M3 (FP8) conversion

    sign = x >> 31 & 0b1
    fp32_exponent = x >> 23 & (2**8 - 1)
    fp32_mantissa = x & (2**23 - 1)

    e4m3_exponent = fp32_exponent - 127 + 7

    # NORMAL
    if fp32_exponent >= 121:

        # Overflow
        if ((fp32_exponent > 135) or (fp32_exponent == 135 and fp32_mantissa > 0b11100000000000000000000)):
            e4m3_exponent = 15
            e4m3_mantissa = 7

        # Normal and no overflow
        else:
            # Round 23 bit mantissa to 3 bits
            first_3_bits_mantissa = fp32_mantissa >> 20
            guard_bit = (fp32_mantissa >> 19) & 0b1
            rest_of_mantissa_bits = fp32_mantissa & (2**19 - 1)
            third_bit = (fp32_mantissa >> 20) & 0b1

            e4m3_mantissa = first_3_bits_mantissa

            # Round to nearest, ties to even (0)
            # If exactly halfway, rounds to one whose lowest bit is 0, prevents upward drift
            if guard_bit == 1 and rest_of_mantissa_bits == 0:
                # Round to nearest, ties to even

                # If third_bit = 0, leave it, else mantissa += 1
                if third_bit == 1:
                    e4m3_mantissa += 1

            elif guard_bit == 1:
                # More than half, so round up
                if rest_of_mantissa_bits != 0:
                    e4m3_mantissa += 1

            # Check for overflow
            if e4m3_mantissa > 0b111:
                e4m3_mantissa = 0
                e4m3_exponent += 1

    # SUBNORMAL
    elif 118 <= fp32_exponent <= 120:
        e4m3_exponent = 0b0000

        # M = (1.f) x 2^(E-118)
        significand = (1 << 23) | fp32_mantissa 

        # Still need to round to nearest even
        first_3_bits_mantissa = significand >> (141 - fp32_exponent)
        guard_bit = significand >> (141 - fp32_exponent - 1) & 0b1
        rest_of_mantissa_bits = significand & (2**(141-fp32_exponent-1) - 1)
        third_bit = first_3_bits_mantissa & 0b1
        e4m3_mantissa = first_3_bits_mantissa

        # Round to nearest, ties to even (0)
        # If exactly halfway, rounds to one whose lowest bit is 0, prevents upward drift
        if guard_bit == 1 and rest_of_mantissa_bits == 0:
            # Round to nearest, ties to even

            # If third_bit = 0, leave it, else mantissa += 1
            if third_bit == 1:
                e4m3_mantissa += 1

        elif guard_bit == 1:
            # More than half, so round up
            if rest_of_mantissa_bits != 0:
                e4m3_mantissa += 1

        # Check for overflow
        if e4m3_mantissa > 0b111:
            e4m3_mantissa = 0
            e4m3_exponent += 1

    # ALL ZEROS
    elif (fp32_exponent < 117 or (fp32_exponent == 117 and fp32_mantissa == 0)):
        e4m3_exponent = 0b0000
        e4m3_mantissa = 0b000

    # BOUNDARY CASE
    elif fp32_exponent == 117:
        # Round upwards to smallest subnormal (exp directly between 2^-9 and 0, so 2^-10)
        e4m3_exponent = 0b0000
        e4m3_mantissa = 0b001

    e4m3_number = construct_e4m3(sign, e4m3_exponent, e4m3_mantissa)

    return e4m3_number

def e4m3_to_fp32(x: int) -> int:

    x_dict = decode_e4m3(x)
    e4m3_sign = x_dict["sign"]
    e4m3_exponent = x_dict["exponent"]
    e4m3_mantissa = x_dict["mantissa"]

    shifted_sign = e4m3_sign << 31

    # NORMAL
    if x_dict["subnormal"] == 0:
        shifted_exponent = (e4m3_exponent + 127) << (31 - 8)
        shifted_mantissa = e4m3_mantissa << (23 - 3)

    # SUBNORMAL
    else:
        # e4m3_exponent is always -6 for a subnormal
         
        if 0 <= e4m3_mantissa < 2:
            fp32_mantissa = e4m3_mantissa >> 3
            fp32_exponent = e4m3_exponent - 3

        elif 2 <= e4m3_mantissa < 4:
            fp32_mantissa = e4m3_mantissa >> 2
            fp32_exponent = e4m3_exponent - 2 
 
        elif 4 <= e4m3_mantissa < 8:
            fp32_mantissa = e4m3_mantissa >> 1
            fp32_exponent = e4m3_exponent - 1 

        shifted_mantissa = fp32_mantissa << (23 - 3)
        shifted_exponent = (fp32_exponent + 127) << (31 - 8)

    return shifted_sign | shifted_exponent | shifted_mantissa

def find_e4m3_sign(x: int) -> int:
    if x > 0:
        return 0
    return 1

def return_sign_exp_mant(x_dict: dict) -> tuple[int, int, int]:
    return (x_dict["sign"], x_dict["exponent"], x_dict["mantissa"])
