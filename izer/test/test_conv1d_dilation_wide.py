#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Test dilation on Conv1d via Conv2d operator.
"""
import os
import sys

import numpy as np
import torch

# Allow test to run outside of pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from izer import compute, state  # noqa: E402 pylint: disable=wrong-import-position


def convolve1d(data, weight, pad=1, stride=1, dilation=1, expected=None):
    """Convolve data in 1D"""
    print('Input:\n', data)

    t = torch.nn.functional.conv1d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=stride,
        padding=pad,  # Keep data dimensions
        groups=1,
        dilation=dilation,
    ).int().squeeze().numpy()

    # print(t.shape)
    # print('PyTorch:\n', (t + 64) // 128)

    output = compute.conv1d(
        data,
        weight,
        None,
        data.shape,
        t.shape,
        kernel_size=weight.shape[1],
        stride=stride,
        pad=pad,
        dilation=dilation,
        fractional_stride=1,
        output_pad=0,
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


def convolve1d_via2d(data, weight, pad=1, stride=1, dilation=1, expected=None):
    """Convolve 1D data with dilation in 2D"""
    print('Input:\n', data)
    assert 0 <= pad <= 1  # Must be 0 or 1 so we can use 1-pad in 2D without edge issues
    assert stride == 1  # This is generally true for AI85
    assert 1 <= weight.shape[2] <= 3

    t = torch.nn.functional.conv1d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),  # Add batch dimension
        torch.as_tensor(weight, dtype=torch.float),
        bias=None,
        stride=stride,
        padding=pad,
        groups=1,
        dilation=dilation,
    ).int().squeeze().numpy()

    # print(t.shape)
    # print('PyTorch 1D:\n', (t + 64) // 128)

    # Compute expected output so we can discard anything extra that was created by padding
    expected_length = (data.shape[1] - dilation * (weight.shape[2] - 1) + 2 * pad) // stride
    assert t.shape[1] == expected_length

    # If pad = 1, insert a 0 on left and right. On MAX78000, this would be achieved by manipulating
    # the in_offset.
    if pad == 1:
        d_padded = np.insert(data, [0, data.shape[1]], 0, axis=1)
    else:
        d_padded = data

    # Pad out the input data so it can be folded
    if data.shape[1] % dilation != 0:
        d_padded = np.append(d_padded,
                             np.zeros((d_padded.shape[0], dilation - d_padded.shape[1] % dilation),
                                      dtype=np.int64), axis=1)
    # Fold the data and create the new weights
    d_folded = d_padded.reshape(d_padded.shape[0], d_padded.shape[1] // dilation, dilation)
    w_folded = np.insert(weight.reshape(weight.shape[0], weight.shape[1], weight.shape[2], -1),
                         [0, 1], 0, axis=3)
    skip = dilation
    if weight.shape[2] == 2:
        skip = 0
        w_folded = np.insert(w_folded, 0, 0, axis=2)  # Insert at top - throw away the padding
    elif weight.shape[2] == 1:
        skip = 0
        w_folded = np.insert(w_folded, [0, 1], 0, axis=2)  # Use center
    assert w_folded.shape[2] == w_folded.shape[3] == 3

    # Use the standard Conv2d method available in MAX78000 and MAX78002
    output = compute.conv2d(
        d_folded,
        w_folded,
        None,
        d_folded.shape,
        (weight.shape[0], d_folded.shape[1], d_folded.shape[2]),
        kernel_size=[w_folded.shape[2], w_folded.shape[3]],
        stride=[stride, stride],
        pad=[1, 1],
        dilation=[1, 1],
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=1,
    )
    # Discard extra data at beginning and end (this can be done by increasing in_offset on the
    # next layer or reducing out_offset on the current layer and by specifying in_dim on the next
    # layer). The amount to skip is 0 for kernel lengths 0 or 1 and 'dilation' for kernel length 3.
    output = output.reshape(output.shape[0], -1)[:, skip:skip + expected_length]

    print("PYTORCH OK" if np.array_equal(output, t) else "*** FAILURE ***")
    assert np.array_equal(output, t)

    print('Output before division:\n', output)
    output += 64
    output //= 128
    print('Output:\n', output)

    print('Expected:\n', expected)
    print("SUCCESS" if np.array_equal(output, expected) else "*** FAILURE ***")
    assert np.array_equal(output, expected)


def test_conv1d_dilation():
    """Main program to test compute.conv1d."""
    state.debug = True

    # 3x16 (CL)
    d0 = np.array(
        [[-41, -98, 16, 73,
          49, 73, 28, 25,
          35, 104, -27, -107,
          111, 42, -46, -10],
         [-114, -28, -31, 21,
          103, -76, 27, 78,
          -51, -74, 57, -76,
          -126, -71, 17, -40],
         [-98, 31, 109, 33,
          -59, 86, -51, 69,
          1, 85, -95, 121,
          -93, 8, -103, 73]],
        dtype=np.int64,
    )

    # 5x3x3
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

    # 5x4
    e0 = np.array(
        [[137, 74, 79, -141, 152, -119, 39, -79, 130, -237, 142, -154, 154, -131],
         [22, 136, 71, -29, -188, 16, -4, -15, -171, 113, -38, -83, -101, -87],
         [-20, -120, 120, -166, -43, 164, -116, 15, 133, 71, -108, 262, 45, -91],
         [89, 134, -21, -109, 159, -75, 69, 14, 46, -80, 212, -42, 39, 68],
         [96, -124, 107, 37, -43, -47, 128, -167, 75, -83, 92, -352, 71, -270]],
        dtype=np.int64,
    )
    convolve1d(d0, w0, pad=1, stride=1, dilation=2, expected=e0)

    # 5x3x5 (for dilation = 2)
    w_dilated = np.insert(w0, [1, 2], 0, axis=2)
    convolve1d(d0, w_dilated, pad=1, stride=1, dilation=1, expected=e0)

    e4 = np.array(
        [[78, -155, 178, -113, 37, -156, 23, -209, 243, -60],
         [-65, 147, 35, -80, -42, 108, -118, 51, -25, -254],
         [-62, 8, 188, -167, -71, 297, 4, -29, 25, 134],
         [35, -78, 88, -63, -12, -23, 141, -123, 119, 75],
         [4, -125, 67, -30, 158, -107, -93, -99, 255, -335]],
        dtype=np.int64,
    )
    convolve1d(d0, w0, pad=1, stride=1, dilation=4, expected=e4)

    # 5x3x9 (for dilation = 4)
    w_dilated = np.insert(w0, [1, 1, 1, 2, 2, 2], 0, axis=2)
    convolve1d(d0, w_dilated, pad=1, stride=1, dilation=1, expected=e4)

    e4_nopad = e4[:, 1:-1]
    convolve1d(d0, w_dilated, pad=0, stride=1, dilation=1, expected=e4_nopad)

    # USE CONV2D AND FOLDING

    # Dilation 4
    # 3x16 input
    convolve1d_via2d(d0, w0, pad=0, stride=1, dilation=4, expected=e4_nopad)

    # 3x15 input (which does not cleanly fold)
    d41 = d0[:, :-1]
    e41 = e4_nopad[:, :-1]
    convolve1d_via2d(d41, w0, pad=0, stride=1, dilation=4, expected=e41)

    # Dilation 7
    e7 = np.array(
        [[-116, 114],
         [246, -32],
         [-146, 115],
         [-31, 10],
         [-79, 38]],
        dtype=np.int64,
    )
    convolve1d_via2d(d0, w0, pad=0, stride=1, dilation=7, expected=e7)

    # Shorter kernel, dilation 5
    w2 = w0[:, :, :2]
    e5 = np.array(
        [[61, -17, 10, -8, 17, -97, 160, -37, 12, -82, 68],
         [57, 32, 25, -135, -33, -132, 77, -278, -105, -190, 111],
         [46, 21, -185, 16, 159, -168, 7, 201, 15, -152, 114],
         [132, -87, -19, 56, 59, -21, 142, 2, 93, 24, 36],
         [-190, 31, 166, -79, -95, -32, 79, -183, -147, -77, 8]],
        dtype=np.int64,
    )
    convolve1d_via2d(d0, w2, pad=0, stride=1, dilation=5, expected=e5)

    # Force to multiple rows
    e11 = np.array(
        [[175, 4, 48, -82, 45],
         [151, -163, -156, -124, 30],
         [-41, 204, -24, -29, 90],
         [181, -3, 65, -33, 47],
         [-11, -205, -28, -37, 20]],
        dtype=np.int64,
    )
    convolve1d_via2d(d0, w2, pad=0, stride=1, dilation=11, expected=e11)

    # Kernel length 1 (edge case)
    w1 = w0[:, :, :1]
    e11a = np.array(
        [[-9, 49, 21, -22, -35, -9, -25, 8, -15, -23, -13, 76, -72, -16, -7, 23],
         [86, 58, -85, -78, 6, -120, 12, -57, -33, -145, 87, 4, -42, -45, 104, -42],
         [-11, 11, -98, -41, 91, -150, 37, 8, -60, -164, 116, -76, -103, -83, 99, -70],
         [44, -37, 13, 22, -21, 65, 5, -32, 42, 79, -33, -25, 123, 54, -22, 8],
         [-134, 73, 62, -6, -14, -26, -44, 95, -57, -46, -25, 124, -238, -69, -47, 43]],
        dtype=np.int64,
    )
    convolve1d_via2d(d0, w1, pad=0, stride=1, dilation=11, expected=e11a)

    # Longer input data
    d1 = np.array(
        [[-31, 45, 119, 29, 103, 127, -92, -42, 13, 127, 127, 105, -128, 40, -128, 25, -34,
          -41, -98, 16, 73, 49, 73, 28, 25, 35, 104, -27, -107, 111, 42, -46, -10],
         [-81, 127, 54, -25, -23, 49, 19, 96, 127, 67, -128, -8, -128, 108, 80, 127,
          -90, -114, -28, -31, 21, 103, -76, 27, 78, -51, -74, 57, -76, -126, -71, 17, -40],
         [-128, -128, 64, 25, 127, 26, 127, -112, -128, -62, -60, 127, -47, 61, -128, -67,
          -33, -98, 31, 109, 33, -59, 86, -51, 69, 1, 85, -95, 121, -93, 8, -103, 73]],
        dtype=np.int64
    )

    # Dilation 6
    e6 = np.array(
        [[-14, -55, -307, -157, -116, -155, 186, 40, -62, -210, 34, -25, 233, 69,
          55, -223, 145, -209, 18, -61, 81],
         [309, -16, -109, -89, -310, 49, -136, 182, 89, -6, -168, -294, 78, 22,
          214, 90, -62, 218, 209, -90, -151],
         [-4, 19, -247, -157, 112, 35, 67, 114, 115, -141, 58, -37, -94, 77,
          324, 5, 80, 148, 9, -88, 74],
         [20, -94, -193, -67, 57, 13, 16, -85, -112, -85, 218, 89, 107, 32,
          -67, -187, 115, -43, -73, 5, 116],
         [48, -16, -64, -160, -244, -47, 156, 159, 51, -97, -249, -136, 61, 156,
          63, -39, -62, -196, 191, -59, -50]],
        dtype=np.int64,
    )
    convolve1d_via2d(d1, w0, pad=0, stride=1, dilation=6, expected=e6)

    # Dilation 9
    e9 = np.array(
        [[-162, 13, 58, 97, 3, -154, 40, 19, 42, -50, 87, -172, -107, -27, 6],
         [30, -204, -98, -138, -43, -65, -34, 58, -70, 7, -101, 7, 191, -40, 133],
         [47, 393, -56, -65, -139, -132, -141, 306, 416, -85, -90, 8, 35, 38, 264],
         [-90, 53, 109, 141, 17, -70, -152, 0, 41, 64, 200, -5, -118, 7, -128],
         [-228, -269, 35, -96, 186, 17, 237, -72, -128, -60, -112, -2, -27, -6, 24]],
        dtype=np.int64,
    )
    convolve1d_via2d(d1, w0, pad=0, stride=1, dilation=9, expected=e9)

    # Dilation 11
    e11 = np.array(
        [[77, 23, 29, -98, -87, -67, 162, -57, 30, -191, -147],
         [146, -14, -89, 14, -109, -176, -235, 307, 172, 8, -83],
         [134, 208, -220, -51, -97, -44, 170, 285, 241, -27, -129],
         [113, 67, 13, -81, -50, 71, -18, -38, 26, -22, 24],
         [-84, -123, 142, 65, 110, -156, 99, 59, 30, -157, -185]],
        dtype=np.int64,
    )
    convolve1d_via2d(d1, w0, pad=0, stride=1, dilation=11, expected=e11)

    # Padding 1
    e8 = np.array(
        [[-120, -270, -349, -56, 128, 146, -93, 54, -180, 115, -67, 77, 7,
          85, -263, 71, -189, 45, 5],
         [4, 168, 134, -370, -48, -278, 11, 16, 193, -90, -181, -34, -88, 64,
          123, 205, 58, 118, 29],
         [-178, 53, 259, 129, -4, -59, -241, 37, 70, 190, 211, -40, -231, 47,
          134, 349, 91, -74, 46],
         [-108, -170, -205, 54, 107, 184, -10, -137, -198, 65, 80, 220, 93,
          -12, -192, -6, -137, 11, 78],
         [-20, -152, -181, -239, 47, -4, 42, 304, 5, -90, -229, -89, 33, 4, 78,
          -13, -79, 74, -206]],
        dtype=np.int64,
    )
    convolve1d_via2d(d1, w0, pad=1, stride=1, dilation=8, expected=e8)


if __name__ == '__main__':
    test_conv1d_dilation()
