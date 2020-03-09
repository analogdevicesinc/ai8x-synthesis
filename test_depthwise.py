#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Test the depthwise conv2d operator.
"""
import numpy as np
import torch
import compute


def depthwise2d(groups, data, weight, expected):
    """Depthwise 2D convolution"""
    print('Input:\n', data)

    t = torch.nn.functional.conv2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=1,
        padding=1,  # Keep data dimensions
        dilation=1,
        groups=groups,
    ).int().squeeze().numpy()

    output = compute.conv2d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=[3, 3],
        stride=[1, 1],
        pad=[1, 1],
        dilation=[1, 1],
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=groups,
        debug=True,
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


def test_depthwise2d():
    """Main program to test compute.conv2d with groups > 1."""

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


if __name__ == '__main__':
    test_depthwise2d()
