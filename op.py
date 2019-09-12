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
ELTWISE_ADD = -1
ELTWISE_SUB = -2
ELTWISE_MUL = -3
ELTWISE_XOR = -4
CONV1D = 1
CONV2D = 2

NAMES = {NONE: 'none',
         ELTWISE_ADD: 'add',
         ELTWISE_SUB: 'sub',
         ELTWISE_MUL: 'mul',
         ELTWISE_XOR: 'xor',
         CONV1D: 'conv1d',
         CONV2D: 'conv2d'}


def string(op):
    """
    Return string rep[reaentation of operator `op`
    """
    return NAMES[op] if op in NAMES else '????'
