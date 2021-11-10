#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test the depthwise conv1d and conv2d operators.
"""
import os
import sys

import numpy as np
import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from izer import compute, state  # noqa: E402 pylint: disable=wrong-import-position


def depthwise2d(groups, data, weight, expected, kernel_size=3, padding=1):
    """Depthwise 2D convolution"""
    print('Input:\n', data)

    t = torch.nn.functional.conv2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=1,
        padding=padding,  # Keep data dimensions
        dilation=1,
        groups=groups,
    ).int().squeeze().numpy()

    output = compute.conv2d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=[kernel_size, kernel_size],
        stride=[1, 1],
        pad=[padding, padding],
        dilation=[1, 1],
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=groups,
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


def depthwise1d(groups, data, weight, expected, kernel_size=9, padding=0):
    """Depthwise 1D convolution"""
    print('Input:\n', data)

    t = torch.nn.functional.conv1d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=1,
        padding=padding,
        dilation=1,
        groups=groups,
    ).int().squeeze().numpy()

    output = compute.conv1d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=kernel_size,
        stride=1,
        pad=padding,
        dilation=1,
        groups=groups,
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


def test_depthwise():
    """Main program to test compute.conv1d and compute.conv2d with groups > 1."""
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

    # 3x1x3x3 (out_ch, in_ch // groups, kH, kW)
    w0 = np.array(
        [[[[-54, -64, 14],
           [52, -44, -60],
           [-90, -52, 42]]],
         [[[1, -77, 58],
           [25, 108, -18],
           [-30, 113, 37]]],
         [[[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]]]],
        dtype=np.int64,
    )

    # 3x4x4
    e0 = np.array(
        [[[64, -45, -134, -48],
          [-21, -26, -48, 22],
          [-109, -173, 46, 38],
          [-64, -17, -5, 50]],
         [[-23, -125, 30, 74],
          [87, -83, 71, -22],
          [-261, -50, 83, -139],
          [-99, -17, -63, 16]],
         [[99, -12, 74, -112],
          [48, 26, 232, 7],
          [80, -152, 112, 19],
          [96, 14, 90, 117]]],
        dtype=np.int64,
    )

    depthwise2d(3, d0, w0, e0)

    d1 = np.array(
        [[[-41, -98],
          [49, 73]],
         [[-98, 31],
          [-59, 86]]],
        dtype=np.int64,
    )

    w1 = np.array(
        [[[[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]]],
         [[[110, 105, 56],
           [-81, -9, 69],
           [121, -69, -10]]]],
        dtype=np.int64,
    )

    e1 = np.array(
        [[[40, 44],
          [-120, -121]],
         [[49, -42],
          [-16, -27]]],
        dtype=np.int64,
    )

    depthwise2d(2, d1, w1, e1)

    d2 = np.array(
        [[[-41, -98],
          [49, 73]],
         [[-98, 31],
          [-59, 86]],
         [[64, -45],
          [-21, -26]],
         [[-109, -173],
          [-64, -17]]],
        dtype=np.int64,
    )

    w2 = np.array(
        [[[[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]],
          [[110, 105, 56],
           [-81, -9, 69],
           [121, -69, -10]]],
         [[[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]],
          [[110, 105, 56],
           [-81, -9, 69],
           [121, -69, -10]]]],
        dtype=np.int64,
    )

    e2 = np.array(
        [[[88, 2],
          [-136, -148]],
         [[-80, -8],
          [-169, -192]]],
        dtype=np.int64,
    )

    depthwise2d(2, d2, w2, e2)

    d3 = np.array(
        [[[-41, -98],
          [49, 73]],
         [[-98, 31],
          [-59, 86]]],
        dtype=np.int64,
    )

    w3 = np.array(
        [[[[33],
           ]],
         [[[110],
           ]]],
        dtype=np.int64,
    )

    e3 = np.array(
        [[[-11, -25],
          [13, 19]],
         [[-84, 27],
          [-51, 74]]],
        dtype=np.int64,
    )

    depthwise2d(2, d3, w3, e3, kernel_size=1, padding=0)

    d4 = np.array(
        [[-41, -98, 16, 73, 49],
         [35, 104, -27, -107, 111]],
        dtype=np.int64,
    )

    w4 = np.array(
        [[[-54, -64, 14]],
         [[52, -44, -60]]],
        dtype=np.int64,
    )

    e4 = np.array(
        [[68, 41, -38],
         [-9, 102, -26]],
        dtype=np.int64,
    )

    depthwise1d(2, d4, w4, e4, kernel_size=3, padding=0)


if __name__ == '__main__':
    test_depthwise()
