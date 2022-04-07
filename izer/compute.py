###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
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
from numpy.typing import ArrayLike

from . import op, state, stats
from .eprint import eprint


def debug_open(
        layer: int,
        base_directory: str,
        test_name: str,
        log_filename: str,  # pylint: disable=unused-argument
) -> None:
    """
    Create debug log for a layer
    """
    if not state.debug_computation:
        return
    state.debug_log = open(
        os.path.join(base_directory, test_name, f'compute-{layer}.csv'),
        mode='w',
        encoding='utf-8',
    )


def debug_print(
        t,
) -> None:
    """
    Print to the compute debug log
    """
    if not state.debug_computation:
        return
    print(t, file=state.debug_log)


def debug_close() -> None:
    """
    Close the compute debug log
    """
    if not state.debug_computation:
        return
    assert state.debug_log is not None
    state.debug_log.close()
    state.debug_log = None


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
) -> ArrayLike:
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

    h = (data.shape[1] - weight.shape[3]) // stride[0] + 1  # Resulting output height
    w = (data.shape[2] - weight.shape[2]) // stride[1] + 1  # Resulting output width

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
) -> ArrayLike:
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
    )


def conv1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        pad,
        dilation,
        fractional_stride=1,
        output_pad=0,
        groups=1,
) -> ArrayLike:
    """
    Compute a 1D convolution.

    Note that all PyTorch numbers are ordered (C, L)
    """
    assert data.shape == tuple(input_size)
    in_channels = input_size[0]
    out_channels = output_size[0]

    weight = weight.reshape(out_channels, input_size[0] // groups, -1)
    data = data.reshape(input_size[0], -1)

    output = np.full(shape=(output_size[0], output_size[1]),
                     fill_value=np.nan, dtype=np.int64)

    # Stretch data for fractionally-strided convolution
    if fractional_stride > 1:
        ndata = np.zeros((data.shape[0],
                          data.shape[1] * fractional_stride - 1),
                         dtype=data.dtype)
        ndata[:, 0::fractional_stride] = data
        data = ndata

    # Create zero padding around data
    if pad or output_pad:
        data = np.pad(data, pad_width=((0, 0), (pad, pad + output_pad)),
                      mode='constant', constant_values=0)

    if dilation > 1:
        # Stretch weights for dilation
        nweight = np.zeros((weight.shape[0], weight.shape[1],
                            (kernel_size - 1) * dilation + 1),
                           dtype=weight.dtype)
        nweight[:, :, 0::dilation] = weight
        weight = nweight

    ll = (data.shape[1] - weight.shape[2]) // stride + 1  # Resulting output length

    view = as_strided(data,
                      shape=(ll, data.shape[0], weight.shape[2]),
                      strides=((data.strides[1] * stride,
                                data.strides[0], data.strides[1])),
                      writeable=False)

    if groups > 1:
        nweight = np.zeros((weight.shape[0], in_channels, weight.shape[2]),
                           dtype=weight.dtype)
        for i in range(weight.shape[0]):
            for j in range(in_channels // groups):
                nweight[i, i * (in_channels // groups) + j, :] = weight[i, j, :]
        weight = nweight

    output = np.tensordot(view, weight, axes=((1, 2), (1, 2))).transpose(1, 0)

    # Apply bias
    if bias is not None:
        for k in range(out_channels):
            output[k] += bias[k]

    assert output.shape == tuple(output_size[:2]), \
        f'Shape mismatch: NumPy result {output.shape} vs expected {tuple(output_size[:2])}.'

    return output


def convtranspose1d(
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
) -> ArrayLike:
    """
    Compute a transposed 1D convolution.
    """

    return conv1d(
        data,
        weight,
        bias,
        input_size,
        output_size,
        kernel_size,
        stride,
        dilation * (kernel_size - 1) - pad,
        dilation,
        fractional_stride=fractional_stride,
        output_pad=output_pad,
        groups=groups,
    )


def linear(
        layer,
        data,
        weight,
        bias,
        in_features,
        out_features,
) -> ArrayLike:
    """
    Compute a fully connected layer.
    """
    output = np.empty(out_features, dtype=np.int64)

    for w in range(out_features):
        val = np.int64(0)
        for n in range(in_features):
            val += data[n] * weight[w][n]
            stats.account(
                layer,
                "true_sw_macc",
                1,
            )
            debug_print(
                f'w={w}, n={n}, weight={weight[w][n]}, data={data[n]} '
                f'-> accumulator = {val} '
            )
        if bias is not None:
            val += bias[w]
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
):
    """
    Compute 2D Pooling (Average or Max)
    """
    assert data.shape == tuple(input_size)

    if state.debug:
        # Slow using pure Python
        ref = np.empty(shape=output_size, dtype=np.int64)

        for c in range(input_size[0]):
            for row in range(0, output_size[1]*stride[0], stride[0]):
                for col in range(0, output_size[2]*stride[1], stride[1]):
                    if average:
                        avg = np.mean(data[c][row:row+pool[0]*dilation[0]:dilation[0],
                                              col:col+pool[1]*dilation[1]:dilation[1]])
                        if floor:
                            if avg < 0.:
                                val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                            else:
                                val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            if avg < 0.:
                                val = np.ceil(avg - 0.5).astype(np.int64).clip(min=-128, max=127)
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
            pooled = np.mean(view, axis=(3, 4))
            pooled = np.ceil(pooled - 0.5, where=pooled < 0., out=pooled)
            pooled = np.floor(pooled + 0.5, where=pooled >= 0., out=pooled) \
                .astype(np.int64).clip(min=-128, max=127)
    else:
        pooled = np.nanmax(view, axis=(3, 4))

    if state.debug:
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
        floor=True,  # pylint: disable=unused-argument
) -> ArrayLike:
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
) -> ArrayLike:
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
