def decode_fp32(x: int) -> int:
    # FP32: 32 bit input (1 sign, 8 exponent, 23 mantissa)

    sign = x >> 31  & 0b1
    exponent = (x >> 23) & 0b11111111
    mantissa = x & (2**(23) - 1)
    is_NAN = 0
    is_subnormal = 0
    is_infinity = 0

    # Checking for NAN
    if exponent == 0b11111111 and mantissa != 0:
        is_NAN = 1

    # Checking for infinities
    elif exponent == 0b11111111 and mantissa == 0:
        is_infinity = 1

    # Checking for subnormal number - implicit 1 becomes a 0
    elif exponent == 0b0000:
        is_subnormal = 1
        exponent = 0

    # Regular normal (finite) number
    else:
        mantissa += (2 ** (23)) # add 2^23 to account for implicit 1
        exponent -= 127

    return_dict = {
            "sign": sign,
            "exponent": exponent,
            "mantissa": mantissa,
            "NAN": is_NAN,
            "subnormal": is_subnormal,
            "infinity": is_infinity
        }
    
    return return_dict

def is_greater_absolute_fp32(x: int, y: int) -> int:
    # Arguments are x, y both FP32 numbers

    dict_x = decode_fp32(x)
    dict_y = decode_fp32(y)

    # Check for NAN's
    if dict_x["NAN"] == 1 and dict_y["NAN"] == 0:
        return y
    elif dict_x["NAN"] == 0 and dict_y["NAN"] == 1:
        return x
    elif dict_x["NAN"] == 1 and dict_y["NAN"] == 1:
        return None

    # Compare exponents
    elif dict_x["exponent"] > dict_y["exponent"]:
        return x
    elif dict_x["exponent"] < dict_y["exponent"]:
        return y

    # Must have equal exponents, so compare mantissas
    elif dict_x["mantissa"] > dict_y["mantissa"]:
        return x
    elif dict_x["mantissa"] < dict_y["mantissa"]:
        return y

    # Absolute value must be equal (same mantissa and exponent)
    return y
