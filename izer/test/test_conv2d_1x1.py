#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test the conv2d operator with 1x1 kernels.
"""
import os
import sys

import numpy as np
import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from izer import compute, state  # noqa: E402 pylint: disable=wrong-import-position


def convolve(data, weight, expected):
    """Convolve data"""
    print('Input:\n', data)

    t = torch.nn.functional.conv2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=[1, 1],
        padding=0,  # Keep data dimensions
        groups=1,
        dilation=1,
    ).int().squeeze().numpy()

    output = compute.conv2d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=[1, 1],
        stride=[1, 1],
        pad=[0, 0],
        dilation=[1, 1],
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=1,
    )

    print("PYTORCH OK" if np.array_equal(output, t) else "*** FAILURE ***")
    assert np.array_equal(output, t)

    print('Output before division:\n', output)
    output += 64
    output //= 128
    print('Output:\n', output)

    print('Expected:\n', expected)
    print("SUCCESS" if np.array_equal(output, expected) else "*** FAILURE ***")
    assert np.array_equal(output, expected)

    # Create 3x3 weights from 1x1 weights
    # and emulate using 3x3 kernels
    shape33 = (weight.shape[0], weight.shape[1], 3, 3)
    weight33 = np.zeros(shape33, dtype=np.int64)
    weight33[:, :, 1, 1] = weight[:, :, 0, 0]

    output = compute.conv2d(
        data,
        weight33,
        None,
        data.shape,
        expected.shape,
        kernel_size=[3, 3],
        stride=[1, 1],
        pad=[1, 1],
        dilation=[1, 1],
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=1,
    )
    print("PYTORCH OK" if np.array_equal(output, t) else "*** FAILURE ***")
    assert np.array_equal(output, t)


def test_conv2d():
    """Main program to test compute.conv2d."""
    state.debug = True

    # 3x4x4 (CHW)
    d0 = np.array(
        [[[-41, -98, 16, 73],
          [49, 73, 28, 25],
          [35, 104, -27, -107],
          [111, 42, -46, -10]],
         [[-114, -28, -31, 21],
          [103, -76, 27, 78],
          [-51, -74, 57, -76],
          [-126, -71, 17, -40]],
         [[-98, 31, 109, 33],
          [-59, 86, -51, 69],
          [1, 85, -95, 121],
          [-93, 8, -103, 73]]],
        dtype=np.int64,
    )

    # 3x3x1x1
    w0 = np.array(
        [[[[-128]], [[0]], [[0]]],
         [[[0]], [[-128]], [[0]]],
         [[[0]], [[0]], [[-128]]]],
        dtype=np.int64,
    )

    # 3x4x4
    e0 = np.array(
        [[[41, 98, -16, -73],
          [-49, -73, -28, -25],
          [-35, -104, 27, 107],
          [-111, -42, 46, 10]],
         [[114, 28, 31, -21],
          [-103, 76, -27, -78],
          [51, 74, -57, 76],
          [126, 71, -17, 40]],
         [[98, -31, -109, -33],
          [59, -86, 51, -69],
          [-1, -85, 95, -121],
          [93, -8, 103, -73]]],
        dtype=np.int64,
    )

    convolve(d0, w0, e0)

    d1 = d0

    # 4x3x1x1
    w1 = np.array(
        [[[[166]], [[27]], [[-33]]],
         [[[15]], [[-12]], [[78]]],
         [[[-114]], [[-128]], [[10]]],
         [[[93]], [[-88]], [[71]]]],
        dtype=np.int64,
    )

    # 4x4x4
    e1 = np.array(
        [[[-52, -141, -14, 91],
          [100, 56, 55, 31],
          [34, 97, 2, -186],
          [141, 37, -30, -40]],
         [[-54, 10, 71, 27],
          [-40, 68, -30, 38],
          [9, 71, -66, 68],
          [-32, 16, -70, 47]],
         [[143, 118, 25, -83],
          [-151, 18, -56, -95],
          [20, -12, -40, 181],
          [20, 34, 16, 55]],
         [[-6, -35, 93, 57],
          [-68, 153, -27, 3],
          [61, 174, -111, 42],
          [116, 84, -102, 61]]],
        dtype=np.int64,
    )

    convolve(d1, w1, e1)


if __name__ == '__main__':
    test_conv2d()
