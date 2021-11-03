###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Part number and device type conversion
"""
import argparse

CMSISNN = -1


def device(astring: str) -> int:
    """
    Take die type, or part number, and return the die type.
    """
    s = astring.lower()

    if s.startswith('max'):
        s = s[3:]  # Strip 'MAX' from part number
    elif s.startswith('ai'):
        s = s[2:]  # Strip 'AI' from die type
    elif s == 'cmsis-nn':
        return CMSISNN

    try:
        num = int(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(astring, 'is not a supported device type') from exc
    if num in [85, 87]:  # Die types
        dev = num
    elif num == 78000:  # Part numbers
        dev = 85
    elif num == 78002:
        dev = 87
    else:
        raise argparse.ArgumentTypeError(astring, 'is not a supported device type')

    return dev
