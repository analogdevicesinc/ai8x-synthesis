#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
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


def convolve(pool_dilation, data, weight, expected):
    """Convolve data"""
    print('Input:\n', data)

    t = torch.nn.functional.max_pool2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        kernel_size=2,
        stride=1,
        dilation=pool_dilation,
    )

    pool_shape = t.squeeze().numpy().shape

    t = torch.nn.functional.conv2d(
        t,
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=1,
        padding=1,  # Keep data dimensions
        groups=1,
        dilation=1,
    ).int().squeeze().numpy()

    print(t.shape)
    print(t)

    data = compute.pool2d(
        data,
        data.shape,
        pool_shape,
        pool=[2, 2],
        stride=[1, 1],
        dilation=[pool_dilation, pool_dilation],
        average=False,
    )

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
    """Main program to test compute.conv2d with pool dilation."""
    state.debug = True

    # 2x8x8
    d1 = np.array(
        [[[106, -99, 45, -57, 109, 7, -83, 125],
          [-82, 98, 56, -14, 12, -56, -54, -38],
          [52, 116, -13, 101, 93, 47, -123, 20],
          [60, -97, 99, -29, 114, -90, 9, 80],
          [-77, 30, 89, -53, -93, 67, 95, 26],
          [16, -45, 117, 108, -9, -58, 92, -51],
          [50, -32, -49, 100, -74, -10, 10, -122],
          [6, -101, -31, -58, 63, -98, -50, 8]],
         [[-71, 60, -17, 114, -40, 17, 32, -43],
          [-75, -116, -112, 108, 49, -76, -70, -95],
          [-38, 99, -121, -74, -96, 69, -105, -112],
          [55, 120, -47, -48, -116, -102, 76, 118],
          [-118, 94, 35, -66, -69, 110, 6, -54],
          [-100, -2, 60, 68, -97, -30, 34, 13],
          [121, -91, 60, 32, 114, -51, -104, 72],
          [116, -78, 7, 57, 84, -121, -90, 48]]],
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

    # 3x6x6
    e1 = np.array(
        [[[2, -91, -87, -21, -10, -135],
          [59, -182, 9, -96, -65, -214],
          [109, -52, -12, 37, 83, -95],
          [49, -16, -23, -25, 7, -110],
          [46, -34, -75, -78, -66, -100],
          [45, 12, 42, 55, 92, 8]],
         [[364, 113, 141, 88, 122, -13],
          [185, 94, -136, 59, -48, -193],
          [275, 80, -82, 46, 115, -107],
          [252, 36, -92, 135, -74, -151],
          [174, 24, -55, -38, -163, -284],
          [-12, -285, -234, -241, -281, -233]],
         [[109, 136, 157, 234,  76, 14],
          [234, 198, 252, 249, 391, -7],
          [261, 325, 397, 281, 396, 168],
          [218, 345, 305, 331, 319, 91],
          [226, 395, 362, 326, 273, 166],
          [327, 313, 310, 319, 214, 103]]],
        dtype=np.int64,
    )

    convolve(2, d1, w1, e1)


if __name__ == '__main__':
    test_conv2d()
