###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Pure Python implementation of Conv1d, Conv2d, Pool2d, and Linear.
Allows debug of individual accumulations.
NumPy implementation of Conv2d, Pool2d.
Compatible with PyTorch.
"""
import sys

import numpy as np
from numpy.lib.stride_tricks import as_strided

import op
import stats


def conv2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        out_channels,
        kernel_size,
        stride,
        pad,
        dilation,
        debug=False,
):
    """
    Compute a 2D convolution.

    SIMPLIFIED TO REMOVE GROUPS

    Note that all PyTorch numbers are ordered (C, H, W)
    """
    assert data.shape == tuple(input_size)
    in_channels = input_size[0]

    if debug:
        # Slow route using pure Python
        ref = np.full(shape=output_size, fill_value=np.nan, dtype=np.int64)

        for k in range(out_channels):
            for y in range(-pad[0],
                           input_size[1] - dilation[0] * (kernel_size[0] - 1) + pad[0],
                           stride[0]):
                for x in range(-pad[1],
                               input_size[2] - dilation[1] * (kernel_size[1] - 1) + pad[1],
                               stride[1]):
                    val = np.int64(0) if bias is None else bias[k]
                    for c in range(in_channels):
                        for h in range(kernel_size[0]):
                            for w in range(kernel_size[1]):
                                ypos, xpos = y + h * dilation[0], x + w * dilation[1]
                                if ypos >= 0 and ypos < input_size[1] and \
                                   xpos >= 0 and xpos < input_size[2]:
                                    val += weight[k][c][h][w] * data[c][ypos][xpos]
                                    stats.true_macc += 1
                                    if debug:
                                        print(f'k={k}, c={c}, x={x}, y={y}: '
                                              f'weight*data={weight[k][c][h][w]}'
                                              f'*{data[c][ypos][xpos]} -> accumulator = {val}')

                    ref[k][(y + pad[0]) // stride[0]][(x + pad[1]) // stride[1]] = val

    # Fast computation using NumPy

    # Create zero padding around data and stretch weights for dilation.
    if pad[0] or pad[1]:
        data = np.pad(data, pad_width=((0, 0), (pad[0], pad[0]), (pad[1], pad[1])),
                      mode='constant', constant_values=0)

    if dilation[0] > 1 or dilation[1] > 1:
        nweight = np.zeros((weight.shape[0], weight.shape[1],
                            (kernel_size[0] - 1) * dilation[0] + 1,
                            (kernel_size[1] - 1) * dilation[1] + 1),
                           dtype=weight.dtype)
        nweight[:, :, 0::dilation[0], 0::dilation[1]] = weight
        weight = nweight

    h = (data.shape[1] - weight.shape[3] + 1) // stride[0]  # Resulting output height
    w = (data.shape[2] - weight.shape[2] + 1) // stride[1]  # Resulting output width

    view = as_strided(data,
                      shape=(h, w, data.shape[0], weight.shape[2], weight.shape[3]),
                      strides=((data.strides[1] * stride[0], data.strides[2] * stride[1],
                                data.strides[0], data.strides[1], data.strides[2])),
                      writeable=False)
    output = np.tensordot(view, weight, axes=((2, 3, 4), (1, 2, 3))).transpose(2, 0, 1)

    # Apply bias
    if bias is not None:
        for k in range(out_channels):
            output[k] += bias[k]

    if debug:
        match = (ref == output).all()
        if not match:
            print('NumPy <-> Python mismatch in compute.conv2d')
            sys.exit(1)

    assert output.shape == tuple(output_size)

    return output


def conv1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        out_channels,
        kernel_size,
        stride,
        pad,
        dilation,
        debug=False,
):
    """
    Compute a 1D convolution.

    SIMPLIFIED TO REMOVE GROUPS

    Note that all PyTorch numbers are ordered (C, L)
    """
    in_channels = input_size[0]

    weight = weight.reshape(out_channels, input_size[0], -1)
    data = data.reshape(input_size[0], -1)

    output = np.full(shape=(output_size[0], output_size[1]),
                     fill_value=np.nan, dtype=np.int64)

    # Compute 1D convolution
    for k in range(out_channels):
        out_offs = 0
        for x in range(-pad, input_size[1] - dilation * (kernel_size - 1) + pad, stride):
            val = np.int64(0)
            for c in range(in_channels):
                for w in range(kernel_size):
                    src_offs = x + w * dilation
                    if src_offs >= 0 and src_offs < input_size[1]:
                        val += weight[k][c][w] * data[c][src_offs]
                        stats.true_macc += 1
                        if debug:
                            print(f'k={k}, c={c}, x={x}, src_offs={src_offs}, '
                                  f'wt_offs={w}: weight*data={weight[k][c][w]}'
                                  f'*{data[c][src_offs]} -> accumulator = {val}')

            if bias is not None:
                val += bias[k]
                if debug:
                    print(f'+bias {bias[k]} --> output[{k}][{out_offs}] = {val}')
            output[k][out_offs] = val
            out_offs += 1

    return output.reshape((output_size))


def linear(
        data,
        weight,
        bias,
        in_features,
        out_features,
        debug=False,
):
    """
    Compute a fully connected layer.
    """
    output = np.empty(out_features, dtype=np.int64)

    for w in range(out_features):
        val = np.int64(0)
        for n in range(in_features):
            val += data[n] * weight[w][n]
            stats.true_sw_macc += 1
            if debug:
                print(f'w={w}, n={n}, weight={weight[w][n]}, data={data[n]} '
                      f'-> accumulator = {val} ')
        if bias is not None:
            val += bias[w]
            if debug:
                print(f'+bias {bias[w]} --> output[{w}] = {val}')
        output[w] = val

    return output


def pool2d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        floor=True,
        debug=False,
):
    """
    Compute 2D Pooling (Average or Max)
    """
    assert data.shape == tuple(input_size)

    if debug:
        # Slow using pure Python
        ref = np.empty(shape=output_size, dtype=np.int64)

        for c in range(input_size[0]):
            for row in range(0, output_size[1]*stride[0], stride[0]):
                for col in range(0, output_size[2]*stride[1], stride[1]):
                    if average:
                        avg = np.average(data[c][row:row+pool[0], col:col+pool[1]])
                        if floor:
                            if avg < 0:
                                val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                            else:
                                val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            val = np.floor(avg + 0.5).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.amax(data[c][row:row+pool[0], col:col+pool[1]])
                    ref[c][row//stride[0]][col//stride[1]] = val

    # Fast computation using NumPy
    data_pad = data[:, :(data.shape[1] - pool[0]) // stride[0] * stride[0] + pool[0],
                    :(data.shape[2] - pool[1]) // stride[1] * stride[1] + pool[1], ...]
    h, w = data_pad.strides[1:]

    view = as_strided(data_pad,
                      shape=(data_pad.shape[0],
                             1 + (data_pad.shape[1]-pool[0]) // stride[0],
                             1 + (data_pad.shape[2]-pool[1]) // stride[1],
                             pool[0], pool[1]),
                      strides=(data_pad.strides[0], stride[0] * h, stride[1] * w, h, w),
                      writeable=False)

    if average:
        if floor:
            pooled = np.nanmean(view, dtype=np.int64, axis=(3, 4))
        else:
            pooled = np.round(np.nanmean(view, axis=(3, 4))).astype(np.int64)
    else:
        pooled = np.nanmax(view, axis=(3, 4))

    if debug:
        match = (ref == pooled).all()
        if not match:
            print('NumPy <-> Python mismatch in compute.pool2d')
            sys.exit(1)

    assert pooled.shape == tuple(output_size)

    return pooled


def pool1d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        floor=True,
        debug=False,
):  # pylint: disable=unused-argument
    """
    Compute 1D Pooling (Average or Max)
    """
    assert data.shape == tuple(input_size)

    pooled = np.empty(shape=output_size, dtype=np.int64)
    for c in range(input_size[0]):
        for x in range(0, output_size[1]*stride, stride):
            if average:
                avg = np.average(data[c][x:x+pool])
                if avg < 0:
                    val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                else:
                    val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
            else:
                val = np.amax(data[c][x:x+pool])
            pooled[c][x//stride] = val

    return pooled


def eltwise(
        operator,
        data,
        input_size,
        debug=False,
):  # pylint: disable=unused-argument
    """
    Compute element-wise operation.
    """
    assert data[0].shape == tuple(input_size)
    operands = len(data)

    output = data[0]
    for i in range(1, operands):
        if operator == op.ELTWISE_ADD:
            output = np.add(output, data[i])
        elif operator == op.ELTWISE_MUL:
            output = np.multiply(output, data[i])
        elif operator == op.ELTWISE_OR:
            output = np.bitwise_or(output, data[i])
        elif operator == op.ELTWISE_SUB:
            output = np.subtract(output, data[i])
        elif operator == op.ELTWISE_XOR:
            output = np.bitwise_xor(output, data[i])
        else:
            print(f"Unknown operator `{op.string(operator)}`")
            raise NotImplementedError

    assert output.shape == tuple(input_size)
    return output
