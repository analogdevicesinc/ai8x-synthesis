#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test the conv2d operator.
"""
import os
import sys

import numpy as np
import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from izer import compute, state  # noqa: E402 pylint: disable=wrong-import-position


def convolve(data, weight, pad, stride, expected):
    """Convolve data"""
    print('Input:\n', data)

    t = torch.nn.functional.conv2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=stride,
        padding=pad,  # Keep data dimensions
        groups=1,
        dilation=1,
    ).int().squeeze().numpy()

    output = compute.conv2d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=[3, 3],
        stride=[stride, stride],
        pad=[pad, pad],
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

    # 5x3x3x3
    w0 = np.array(
        [[[[-54, -64, 14],
           [52, -44, -60],
           [-90, -52, 42]],
          [[1, -77, 58],
           [25, 108, -18],
           [-30, 113, 37]],
          [[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]]],
         [[[-104, -48, 34],
           [-41, 8, 11],
           [33, 96, 79]],
          [[11, 111, -65],
           [-82, 121, 94],
           [-49, -67, -29]],
          [[-81, 96, -89],
           [-109, -109, 98],
           [-46, 41, 99]]],
         [[[-67, 51, 43],
           [-7, -12, 118],
           [102, -68, 54]],
          [[102, -110, -127],
           [49, 14, 36],
           [-26, -23, -7]],
          [[-76, -56, 19],
           [49, -79, -79],
           [112, 52, 1]]],
         [[[62, -23, 31],
           [15, -50, -46],
           [72, 36, -53]],
          [[-62, -100, 31],
           [24, -108, 81],
           [-72, 85, 30]],
          [[-11, 62, 44],
           [70, 78, -108],
           [-45, -50, 87]]],
         [[[-85, -107, 3],
           [-90, -112, 47],
           [-74, -101, 71]],
          [[86, 111, -18],
           [6, 72, -99],
           [54, -76, -114]],
          [[110, 105, 56],
           [-81, -9, 69],
           [121, -69, -10]]]],
        dtype=np.int64,
    )

    # 5x4x4
    e011 = np.array(
        [[[140, -183, -31, -86],
          [113, -82, 255, 6],
          [-290, -375, 240, -82],
          [-67, -21, 22, 182]],
         [[61, 261, 7, -67],
          [177, -54, 216, -32],
          [165, -72, -106, 49],
          [-126, -60, -79, 143]],
         [[-98, -167, 102, -23],
          [199, 40, 136, -145],
          [30, 87, -294, -224],
          [225, -152, -147, 7]],
         [[159, -229, 200, 80],
          [-287, 218, 277, -195],
          [-260, 338, 44, 51],
          [10, 198, -156, 128]],
         [[14, 138, -27, -292],
          [140, -324, 33, 92],
          [255, -600, 28, 112],
          [-109, -376, 95, 177]]],
        dtype=np.int64,
    )
    e013 = np.array(
        [[[140, -86],
          [-67, 182]],
         [[61, -67],
          [-126, 143]],
         [[-98, -23],
          [225, 7]],
         [[159,  80],
          [10, 128]],
         [[14, -292],
          [-109, 177]]],
        dtype=np.int64,
    )
    e023 = np.array(
        [[[-124, 87],
          [-88, 240]],
         [[-75, 103],
          [-7, -106]],
         [[-12, 26],
          [-24, -294]],
         [[-76, -112],
          [-168, 44]],
         [[86, 40],
          [195, 28]]],
        dtype=np.int64,
    )

    convolve(d0, w0, 1, 1, e011)
    convolve(d0, w0, 1, 3, e013)
    convolve(d0, w0, 2, 3, e023)

    # 2x2x2
    d1 = np.array(
        [[[-41, -98],
          [49, 73]],
         [[-98, 31],
          [-59, 86]]],
        dtype=np.int64,
    )

    # 3x2x3x3
    w1 = np.array(
        [[[[-54, -64, 14],
           [52, -44, -60],
           [-90, -52, 42]],
          [[33, 90, 123],
           [-82, -17, 17],
           [55, -29, 102]]],
         [[[-104, -48, 34],
           [-41, 8, 11],
           [33, 96, 79]],
          [[-81, 96, -89],
           [-109, -109, 98],
           [-46, 41, 99]]],
         [[[-67, 51, 43],
           [-7, -12, 118],
           [102, -68, 54]],
          [[110, 105, 56],
           [-81, -9, 69],
           [121, -69, -10]]]],
        dtype=np.int64,
    )

    # 3x2x2
    e1 = np.array(
        [[[163, -33],
          [-61, 84]],
         [[226, 180],
          [20, 121]],
         [[-33, -31],
          [-3, -55]]],
        dtype=np.int64,
    )

    convolve(d1, w1, 1, 1, e1)


if __name__ == '__main__':
    test_conv2d()
