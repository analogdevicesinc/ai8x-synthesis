###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Operators
"""
NONE = 0
CONV1D = 1
CONV2D = 2
LINEAR = 3

ACT_RELU = 1
ACT_ABS = 2

ELTWISE_ADD = -1
ELTWISE_MUL = -2
ELTWISE_OR = -3
ELTWISE_SUB = -4
ELTWISE_XOR = -5

NAMES = {
    NONE: 'passthrough',
    CONV1D: 'conv1d',
    CONV2D: 'conv2d',
}

ELT_NAMES = {
    NONE: 'none',
    ELTWISE_ADD: 'add',
    ELTWISE_SUB: 'sub',
    ELTWISE_MUL: 'mul',
    ELTWISE_XOR: 'xor',
    ELTWISE_OR: 'or',
}

ENCODING = {
    ELTWISE_ADD: 0b01,
    ELTWISE_SUB: 0b00,
    ELTWISE_XOR: 0b11,
    ELTWISE_OR: 0b10,
}


def string(
        op,
        elt=False,
):
    """
    Return string rep[reaentation of operator `op`
    """
    if not elt:
        return NAMES[op] if op in NAMES else '????'
    else:
        return ELT_NAMES[op] if op in ELT_NAMES else '????'


def eltwise(
        op,
):
    """
    Returns `True` when `op` is an element-wise operator.
    """
    return op in [ELTWISE_ADD, ELTWISE_MUL, ELTWISE_SUB, ELTWISE_XOR, ELTWISE_OR]


def eltwise_fn(
        op,
):
    """
    Returns the bit encoding for `op`, where `op` is an element-wise operator.
    """
    if op in ENCODING:
        return ENCODING[op]
    raise NotImplementedError
