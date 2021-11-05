#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test the convtranspose1d operator.
"""
import os
import sys

import numpy as np
import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from izer import compute, state  # noqa: E402 pylint: disable=wrong-import-position


def deconvolve(groups, data, weight, expected, pad=1, output_pad=1):
    """Upsample data"""
    print('Input:\n', data)

    t = torch.nn.functional.conv_transpose1d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(axis=0),  # add batch dimension
        torch.as_tensor(np.flip(weight, axis=2).swapaxes(0, 1).copy(), dtype=torch.float),
        bias=None,
        stride=2,
        padding=pad,
        output_padding=output_pad,
        groups=groups,
        dilation=1,
    ).int().squeeze(axis=0).numpy()

    assert t.shape == expected.shape, f'got {t.shape}, expected {expected.shape}'

    # pad: Add dilation * (kernel_size - 1) - padding
    output = compute.convtranspose1d(
        data,
        weight,
        None,
        data.shape,
        expected.shape,
        kernel_size=3,
        stride=1,
        pad=pad,
        dilation=1,
        fractional_stride=2,
        output_pad=output_pad,
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


def test_convtranspose1d():
    """Main program to test compute.conv1d with fractional stride."""
    state.debug = True

    # 3x4 (CL)
    d0 = np.array(
        [[-41, -98, 16, 73],
         [-114, -28, -31, 21],
         [-98, 31, 109, 33]],
        dtype=np.int64,
    )

    # 3x5x3
    w0 = np.array(
        [[[-54, -64, 14],
          [1, -77, 58],
          [33, 90, 123]],
         [[-104, -48, 34],
          [11, 111, -65],
          [-81, 96, -89]],
         [[-67, 51, 43],
          [102, -110, -127],
          [-76, -56, 19]],
         [[62, -23, 31],
          [-62, -100, 31],
          [-11, 62, 44]],
         [[-85, -107, 3],
          [86, 111, -18],
          [110, 105, 56]]],
        dtype=np.int64,
    )

    e00 = np.array(
        [[-150, 20, -2, 88, 142, 87, 70, -26, -22, 0],
         [115, -157, 52, 36, 2, 49, -99, 16, -78, 0],
         [85, 125, -12, -29, 63, -15, -89, -3, -41, 0],
         [-71, 49, 24, 55, -3, 74, 48, -14, 22, 0],
         [-28, -145, -118, 83, 125, 49, 75, -16, -6, 0]],
        dtype=np.int64,
    )
    e01 = np.array(
        [[20, -2, 88, 142, 87, 70, -26, -22],
         [-157, 52, 36, 2, 49, -99, 16, -78],
         [125, -12, -29, 63, -15, -89, -3, -41],
         [49, 24, 55, -3, 74, 48, -14, 22],
         [-145, -118, 83, 125, 49, 75, -16, -6]],
        dtype=np.int64,
    )
    e02 = np.array(
        [[-2, 88, 142, 87, 70, -26],
         [52, 36, 2, 49, -99, 16],
         [-12, -29, 63, -15, -89, -3],
         [24, 55, -3, 74, 48, -14],
         [-118, 83, 125, 49, 75, -16]],
        dtype=np.int64,
    )

    deconvolve(1, d0, w0, e00, pad=0, output_pad=1)
    deconvolve(1, d0, w0, e01, pad=1, output_pad=1)
    deconvolve(1, d0, w0, e02, pad=2, output_pad=1)
    deconvolve(1, d0, w0, e00[::, :-1], pad=0, output_pad=0)
    deconvolve(1, d0, w0, e01[::, :-1], pad=1, output_pad=0)
    deconvolve(1, d0, w0, e02[::, :-1], pad=2, output_pad=0)

    d1 = np.array(
        [[-41, -98],
         [49, 73]],
        dtype=np.int64,
    )

    w1 = np.array(
        [[[-54, -64, 14],
          [33, 90, 123]],
         [[-104, -48, 34],
          [-81, 96, -89]],
         [[-67, 51, 43],
          [110, 105, 56]]],
        dtype=np.int64,
    )

    e10 = np.array(
        [[43, 55, 89, 100, 60, 0],
         [-45, 52, -74, 92, 33, 0],
         [8, 24, 63, 21, 114, 0]],
        dtype=np.int64,
    )

    e11 = np.array(
        [[55, 89, 100, 60],
         [52, -74, 92, 33],
         [24, 63, 21, 114]],
        dtype=np.int64,
    )

    e12 = np.array(
        [[89, 100],
         [-74, 92],
         [63, 21]],
        dtype=np.int64,
    )

    deconvolve(1, d1, w1, e10, pad=0, output_pad=1)
    deconvolve(1, d1, w1, e11, pad=1, output_pad=1)
    deconvolve(1, d1, w1, e12, pad=2, output_pad=1)
    deconvolve(1, d1, w1, e10[::, :-1], pad=0, output_pad=0)
    deconvolve(1, d1, w1, e11[::, :-1], pad=1, output_pad=0)
    deconvolve(1, d1, w1, e12[::, :-1], pad=2, output_pad=0)

    d2 = np.array(
        [[-41, -98],
         [49, 73]],
        dtype=np.int64,
    )

    w2 = np.array(
        [[[-54, -64, 14],
          [-67, 51, 43]]],
        dtype=np.int64,
    )

    e20 = np.array(
        [[-4, 21, 7, 49, 41, 0],
         [16, 20, -1, 29, -38, 0]],
        dtype=np.int64,
    )
    e21 = np.array(
        [[21, 7, 49, 41],
         [20, -1, 29, -38]],
        dtype=np.int64,
    )
    e22 = np.array(
        [[7, 49],
         [-1, 29]],
        dtype=np.int64,
    )

    deconvolve(2, d2, w2, e20, pad=0, output_pad=1)
    deconvolve(2, d2, w2, e21, pad=1, output_pad=1)
    deconvolve(2, d2, w2, e22, pad=2, output_pad=1)
    deconvolve(2, d2, w2, e20[::, :-1], pad=0, output_pad=0)
    deconvolve(2, d2, w2, e21[::, :-1], pad=1, output_pad=0)
    deconvolve(2, d2, w2, e22[::, :-1], pad=2, output_pad=0)

    d3 = np.array(
        [[-41, -98]],
        dtype=np.int64,
    )

    w3 = np.array(
        [[[-54, -64, 14]]],
        dtype=np.int64,
    )

    e30 = np.array(
        [[-4, 21, 7, 49, 41, 0]],
        dtype=np.int64,
    )
    e31 = np.array(
        [[21, 7, 49, 41]],
        dtype=np.int64,
    )
    e32 = np.array(
        [[7, 49]],
        dtype=np.int64,
    )

    deconvolve(1, d3, w3, e30, pad=0, output_pad=1)
    deconvolve(1, d3, w3, e31, pad=1, output_pad=1)
    deconvolve(1, d3, w3, e32, pad=2, output_pad=1)
    deconvolve(1, d3, w3, e30[::, :-1], pad=0, output_pad=0)
    deconvolve(1, d3, w3, e31[::, :-1], pad=1, output_pad=0)
    deconvolve(1, d3, w3, e32[::, :-1], pad=2, output_pad=0)

    d4 = np.array(
        [[-41, -98, 16, 73]],
        dtype=np.int64,
    )

    w4 = np.array(
        [[[-54, -64, 14]]],
        dtype=np.int64,
    )

    e40 = np.array(
        [[-4, 21, 7, 49, 43, -8, 1, -36, -31, 0]],
        dtype=np.int64,
    )
    e41 = np.array(
        [[21, 7, 49, 43, -8, 1, -36, -31]],
        dtype=np.int64,
    )
    e42 = np.array(
        [[7, 49, 43, -8, 1, -36]],
        dtype=np.int64,
    )

    deconvolve(1, d4, w4, e40, pad=0, output_pad=1)
    deconvolve(1, d4, w4, e41, pad=1, output_pad=1)
    deconvolve(1, d4, w4, e42, pad=2, output_pad=1)
    deconvolve(1, d4, w4, e40[::, :-1], pad=0, output_pad=0)
    deconvolve(1, d4, w4, e41[::, :-1], pad=1, output_pad=0)
    deconvolve(1, d4, w4, e42[::, :-1], pad=2, output_pad=0)


if __name__ == '__main__':
    test_convtranspose1d()
