###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Pure Python implementation of Conv1d, Conv2d, ConvTranspose2d, Pool1d, Pool2d, Eltwise, and Linear.
Allows debug of individual accumulations.
NumPy implementation of Conv2d, ConvTranspose2d, Pool2d.
Compatible with PyTorch.
"""
import os

import numpy as np
from numpy.lib.stride_tricks import as_strided

from . import op, stats
from .eprint import eprint

debug_log = None


def debug_open(
        layer,
        base_directory,
        test_name,
        log_filename,  # pylint: disable=unused-argument
):
    """
    Create debug log for a layer
    """
    global debug_log  # pylint: disable=global-statement
    debug_log = open(os.path.join(base_directory, test_name,
                                  f'compute-{layer}.csv'), 'w')


def debug_print(
        t,
):
    """
    Print to the compute debug log
    """
    global debug_log  # pylint: disable=global-statement
    print(t, file=debug_log)


def debug_close():
    """
    Close the compute debug log
    """
    global debug_log  # pylint: disable=global-statement
    debug_log.close()


def conv2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride,
        output_pad,
        groups=1,
        debug=False,  # pylint: disable=unused-argument
):
    """
    Compute a 2D convolution.

    Note that all PyTorch numbers are ordered (C, H, W)
    """
    assert data.shape == tuple(input_size)
    in_channels = input_size[0]
    out_channels = output_size[0]

    # Stretch data for fractionally-strided convolution
    if fractional_stride[0] > 1 or fractional_stride[1] > 1:
        ndata = np.zeros((data.shape[0],
                          data.shape[1] * fractional_stride[0] - 1,
                          data.shape[2] * fractional_stride[1] - 1),
                         dtype=data.dtype)
        ndata[:, 0::fractional_stride[0], 0::fractional_stride[1]] = data
        data = ndata

    # Create zero padding around data
    if pad[0] or pad[1] or output_pad[0] or output_pad[1]:
        data = np.pad(data, pad_width=((0, 0),
                                       (pad[0], pad[0] + output_pad[0]),
                                       (pad[1], pad[1] + output_pad[1])),
                      mode='constant', constant_values=0)

    if dilation[0] > 1 or dilation[1] > 1:
        # Stretch weights for dilation
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

    if groups > 1:
        nweight = np.zeros((weight.shape[0], in_channels, weight.shape[2], weight.shape[3]),
                           dtype=weight.dtype)
        for i in range(weight.shape[0]):
            for j in range(in_channels // groups):
                nweight[i, i * (in_channels // groups) + j, :, :] = weight[i, j, :, :]
        weight = nweight

    output = np.tensordot(view, weight, axes=((2, 3, 4), (1, 2, 3))).transpose(2, 0, 1)

    # Apply bias
    if bias is not None:
        for k in range(out_channels):
            output[k] += bias[k]

    assert output.shape == tuple(output_size), \
        f'Shape mismatch: NumPy result {output.shape} vs expected {output_size}'

    return output


def convtranspose2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride,
        output_pad,
        groups=1,
        debug=False,
):
    """
    Compute a transposed 2D convolution.
    """

    return conv2d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        (
            dilation[0] * (kernel_size[0] - 1) - pad[0],
            dilation[1] * (kernel_size[1] - 1) - pad[1]
        ),
        dilation,
        fractional_stride,
        output_pad,
        groups,
        debug,
    )


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
        groups=1,
        debug=False,
):
    """
    Compute a 1D convolution.

    Note that all PyTorch numbers are ordered (C, L)
    """
    in_channels = input_size[0]

    weight = weight.reshape(out_channels, input_size[0] // groups, -1)
    data = data.reshape(input_size[0], -1)

    output = np.full(shape=(output_size[0], output_size[1]),
                     fill_value=np.nan, dtype=np.int64)

    # Compute 1D convolution
    if debug:
        debug_print('k,c,x,src_offs,wt_offs,weight,data,acc')
    for k in range(out_channels):
        out_offs = 0
        for x in range(-pad, input_size[1] - dilation * (kernel_size - 1) + pad, stride):
            val = np.int64(0)
            for c in range(in_channels // groups):
                dc = c if groups == 1 else c + k * (in_channels // groups)
                for w in range(kernel_size):
                    src_offs = x + w * dilation
                    if 0 <= src_offs < input_size[1]:
                        val += weight[k][c][w] * data[dc][src_offs]
                        stats.true_macc += 1
                        if debug:
                            debug_print(
                                f'{k},{c},{x},{src_offs},{w},{weight[k][c][w]},'
                                f'{data[dc][src_offs]},{val}'
                            )

            if bias is not None:
                val += bias[k]
                if debug:
                    debug_print(
                        f'+bias {bias[k]} --> output[{k}][{out_offs}] = {val}',
                    )
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
                debug_print(
                    f'w={w}, n={n}, weight={weight[w][n]}, data={data[n]} '
                    f'-> accumulator = {val} '
                )
        if bias is not None:
            val += bias[w]
            if debug:
                debug_print(f'+bias {bias[w]} --> output[{w}] = {val}')
        output[w] = val

    return output


def pool2d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        dilation=(1, 1),
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
                        avg = np.average(data[c][row:row+pool[0]*dilation[0]:dilation[0],
                                                 col:col+pool[1]*dilation[1]:dilation[1]])
                        if floor:
                            if avg < 0:
                                val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                            else:
                                val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            val = np.floor(avg + 0.5).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.amax(data[c][row:row+pool[0]*dilation[0]:dilation[0],
                                              col:col+pool[1]*dilation[1]:dilation[1]])
                    ref[c][row//stride[0]][col//stride[1]] = val

    # Fast computation using NumPy
    data_pad = data[
        :,
        :(data.shape[1] - pool[0] + dilation[0] - 1) // stride[0] * stride[0] + pool[0],
        :(data.shape[2] - pool[1] + dilation[1] - 1) // stride[1] * stride[1] + pool[1],
        ...
    ]
    h, w = data_pad.strides[1:]

    view = as_strided(data_pad,
                      shape=(data_pad.shape[0],
                             1 + (data_pad.shape[1] - pool[0] - dilation[0] + 1) // stride[0],
                             1 + (data_pad.shape[2] - pool[1] - dilation[1] + 1) // stride[1],
                             pool[0], pool[1]),
                      strides=(data_pad.strides[0], stride[0] * h,
                               stride[1] * w, h * dilation[0], w * dilation[1]),
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
            eprint('NumPy <-> Python mismatch in compute.pool2d')

    assert pooled.shape == tuple(output_size), f'shape mismatch {pooled.shape} vs {output_size}'

    return pooled


def pool1d(
        data,
        input_size,
        output_size,
        pool,
        stride,
        average,
        dilation=1,
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
                avg = np.average(data[c][x:x+pool*dilation:dilation])
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
