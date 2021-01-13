#!/usr/bin/env python3
###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test the conv1d operator.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import izer.compute as compute  # noqa: E402 pylint: disable=wrong-import-position, import-error


def convolve1d(groups, data, weight, bias, expected):
    """Convolve 1d data"""
    print('Input:\n', data)

    if bias is not None:
        bias = torch.as_tensor(bias, dtype=torch.float)

    t = torch.nn.functional.conv1d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=bias,
        stride=1,
        padding=0,
        groups=groups,
        dilation=1,
    ).int().squeeze().numpy()

    output = compute.conv1d(data, weight, bias, data.shape, expected.shape, weight.shape[0],
                            kernel_size=9, stride=1, pad=0, dilation=1, groups=groups, debug=True)

    print("PYTORCH OK" if np.array_equal(output, t) else "*** FAILURE ***")
    assert np.array_equal(output, t)

    print('Output before division:\n', output)
    output += 64
    output //= 128
    print('Output:\n', output)

    expected += 64
    expected //= 128

    print('Expected:\n', expected)
    print("SUCCESS" if np.array_equal(output, expected) else "*** FAILURE ***")

    assert np.array_equal(output, expected)


def test_conv1d():
    """Main program to test compute.conv1d."""

    d0 = np.array(
        [[-31, 45, 119, 29, 103, 127, -92, -42, 13, 127, 127, 105, -128, 40, -128, 25, -34],
         [-81, 127, 54, -25, -23, 49, 19, 96, 127, 67, -128, -8, -128, 108, 80, 127, -90],
         [-128, -128, 64, 25, 127, 26, 127, -112, -128, -62, -60, 127, -47, 61, -128, -67, -33]],
        dtype=np.int64)

    w0 = np.array(
        [[[-1, -56, -24, 125, 90, 127, -55, 37, -33],
          [-119, 127, -33, -128, 97, 101, -128, -128, -12],
          [-60, 116, -128, -62, -56, 85, 108, -11, 42]],
         [[60, 127, -128, -117, -128, 87, 127, -128, 127],
          [7, 120, -92, -128, -51, -44, -128, -128, 97],
          [-70, 127, 127, 36, 124, -80, 44, 127, -82]],
         [[-82, 42, 48, 127, -92, 63, 127, 127, -128],
          [91, -5, 4, -33, 83, -28, -128, 127, -119],
          [-22, 97, 118, -49, -128, -128, 60, -128, 69]],
         [[-84, 127, -128, -70, 19, -58, -128, 127, -2],
          [127, 81, -60, 33, -128, -55, 10, -46, 127],
          [-71, -114, 98, 105, 64, -2, -67, 64, 82]],
         [[-71, 78, -128, 127, 1, -128, -81, 127, -64],
          [-9, 127, -83, -128, -61, -65, 127, 118, -67],
          [-56, 127, 127, 127, -119, 77, 95, 4, 99]]],
        dtype=np.int64)

    e0 = np.array(
        [[28176, -23269, -80134, -5457, 20372, 60838, 43704, -25237, -29043],
         [5219, -24047, -24911, 13560, 35007, 24223, -106564, -18514, -51368],
         [-31662, -18736, 38504, 49859, 95049, -6485, -21362, -18296, 23517],
         [14631, 45702, -15616, -18696, -32693, -79537, 12559, 4471, 51313],
         [-13630, 77478, 3115, -9839, -25421, -89124, -25032, 3245, 69909]],
        dtype=np.int64)

    convolve1d(1, d0, w0, None, e0)

    d1 = np.array(
        [[-31, 45, 119, 29, 103, 127, -92, -42, 13, 127, 127, 105, -128, 40, -128, 25, -34],
         [-81, 127, 54, -25, -23, 49, 19, 96, 127, 67, -128, -8, -128, 108, 80, 127, -90],
         [-128, -128, 64, 25, 127, 26, 127, -112, -128, -62, -60, 127, -47, 61, -128, -67, -33]],
        dtype=np.int64)

    w1 = np.array(
        [[[-1, -56, -24, 125, 90, 127, -55, 37, -33]],
         [[60, 127, -128, -117, -128, 87, 127, -128, 127]],
         [[-56, 127, 127, 127, -119, 77, 95, 4, 99]]],
        dtype=np.int64)

    e1 = np.array(
        [[26756, 3816, -2161, -28225, 8166, 23386, 55516, -2426, 5599],
         [20743, 20195, -5507, 9722, -50736, -12422, -14995, 37782, 41703],
         [-11951, 23995, -23063, 44075, -1292, 4867, -45440, -45560, 2398]],
        dtype=np.int64)

    convolve1d(3, d1, w1, None, e1)

    d2 = d1
    w2 = w1

    b2 = np.array(
        [100, -200, 300],
        dtype=np.int64)

    e2 = np.array(
        [[26856, 3916, -2061, -28125, 8266, 23486, 55616, -2326, 5699],
         [20543, 19995, -5707, 9522, -50936, -12622, -15195, 37582, 41503],
         [-11651, 24295, -22763, 44375, -992, 5167, -45140, -45260, 2698]],
        dtype=np.int64)

    convolve1d(3, d2, w2, b2, e2)


if __name__ == '__main__':
    test_conv1d()
