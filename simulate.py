###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Simulate a single CNN layer
"""
import numpy as np

import op
import stats
import tornadocnn as tc
from compute import conv1d, conv2d, linear, pool1d, pool2d, eltwise


def print_data(
        header,
        data,
        input_size,
        expand,
        expand_thresh,
):
    """
    Print `data` of dimensions `input_size` with `expand` and `expand_thresh`,
    prefixed by `header`.
    """
    int8_format = '{0:4}' if np.any(data < 0) else '{0:3}'

    print(header)

    with np.printoptions(formatter={'int': int8_format.format}):
        if input_size[1] == input_size[2] == 1:
            for i in range(0, input_size[0], expand_thresh):
                last = min(i + expand_thresh, input_size[0])
                if last - 1 > i:
                    print(f'Channels #{i} to #{last-1}', end='')
                else:
                    print(f'Channel #{i}', end='')
                if expand and expand > 1:
                    print(f' (expansion: {(i // expand_thresh) + 1} of {expand})')
                else:
                    print('')
                print(np.squeeze(data[i:last]))
        else:
            for i in range(input_size[0]):
                print(f'Channel #{i}', end='')
                if expand and expand > 1:
                    print(f' (expansion: {(i // expand_thresh) + 1} of {expand})')
                else:
                    print('')
                print(data[i])
    print('')


def cnn2d_layer(
        layer,  # pylint: disable=unused-argument
        verbose,
        input_size,
        kernel_size,
        quantization,
        output_channels,
        padding,
        dilation,
        stride,
        do_activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        device=84,  # pylint: disable=unused-argument
        debug=False,
):
    """
    Perform 2D convolution for one layer.
    """
    if verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S):")
        with np.printoptions(formatter={'int': '{0:4}'.format}):
            for i in range(output_channels):
                print(f'Output channel #{i}')
                if kernel_size[0] == kernel_size[1] == 1:
                    print(np.squeeze(kernel[i]))
                else:
                    print(kernel[i])
        print(f"BIAS: {bias}\n")

    out_size = [output_channels,
                (input_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (input_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv2d(data=data,
                     weight=kernel,
                     bias=bias,
                     input_size=input_size,
                     output_size=out_size,
                     out_channels=output_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     pad=padding,
                     dilation=dilation,
                     debug=debug)

    if verbose:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        if out_size[1] == out_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    stats.macc += input_size[0] * kernel_size[0] * kernel_size[1] * out_size[0] \
        * out_size[1] * out_size[2]

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (16*quantization)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if do_activation else '(NO ACTIVATION)'}:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.comp += out_size[0] * out_size[1] * out_size[2]

    return out_buf, out_size


def cnn1d_layer(
        layer,  # pylint: disable=unused-argument
        verbose,
        input_size,
        kernel_size,
        quantization,
        output_channels,
        padding,
        dilation,
        stride,
        do_activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        device=84,  # pylint: disable=unused-argument
        debug=False,
):
    """
    Perform 1D convolution for one layer.
    """
    if verbose:
        print(f"KERNEL SIZE {kernel_size}:")
        print(kernel)
        print(f"BIAS: {bias}\n")

    out_size = [output_channels,
                (input_size[1] - dilation * (kernel_size - 1) - 1 +
                 2 * padding) // stride + 1,
                1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv1d(data=data,
                     weight=kernel,
                     bias=bias,
                     input_size=input_size,
                     output_size=out_size,
                     out_channels=output_channels,
                     kernel_size=kernel_size,
                     stride=stride,
                     pad=padding,
                     dilation=dilation,
                     debug=debug)

    if verbose:
        print(f"{out_size[0]}x{out_size[1]} FULL-RES OUTPUT:")
        print(out_buf.squeeze(axis=-1))
        print('')

    stats.macc += input_size[0] * kernel_size * out_size[0] \
        * out_size[1]

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (16*quantization)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if do_activation else '(NO ACTIVATION)'}:")
            print(out_buf.squeeze(axis=-1))
            print('')

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"{out_size[0]}x{out_size[1]} ACTIVATED OUTPUT:")
            print(out_buf.squeeze(axis=-1))
            print('')

        stats.comp += out_size[0] * out_size[1]

    return out_buf, out_size


def linear_layer(
        verbose,
        do_activation,
        weight,
        bias,
        data,
        bits=16,
        debug=False,
):
    """
    Perform one linear layer.
    """
    in_features = data.shape[0]
    out_features = weight.shape[0]

    if verbose:
        print("CLASSIFICATION LAYER (LINEAR)...\n")
        print(f"INPUT DATA (size {in_features}):")
        print(data)
        print('')

        print(f"WEIGHTS (size {in_features * out_features}):")
        print(weight)
        print(f"BIAS: {bias}\n")

    out_buf = linear(data=data, weight=weight, bias=bias,
                     in_features=in_features, out_features=out_features,
                     debug=debug)
    out_buf = np.floor(0.5 + out_buf / 128).astype(np.int64). \
        clip(-(2**(bits-1)), 2**(bits-1)-1)

    if verbose:
        print(f"OUTPUT (size {out_features}):")
        print(out_buf)
        print('')

    stats.sw_macc += in_features * out_features

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"ACTIVATED OUTPUT (size {out_features}):")
            print(out_buf)
            print('')

        stats.sw_comp += out_features

    return out_buf, out_features


def passthrough_layer(
        layer,  # pylint: disable=unused-argument
        verbose,  # pylint: disable=unused-argument
        input_size,
        data,
        device=84,  # pylint: disable=unused-argument
        debug=False,  # pylint: disable=unused-argument
):
    """
    2D passthrough for one layer.
    """

    return data, input_size


def eltwise_layer(
        operator,
        layer,  # pylint: disable=unused-argument
        verbose,
        input_size,
        data,
        output_width=8,
        device=84,  # pylint: disable=unused-argument
        debug=False,
        operands=1,
):
    """
    Element-wise operators for one layer.
    """
    quantization = bits = 8
    assert operands == len(data)

    if verbose:
        print(f"{operands}-OPERAND {op.string(operator, elt=True).upper()}:\n")

    out_buf = eltwise(operator=operator,
                      data=data,
                      input_size=input_size,
                      debug=debug)

    if verbose:
        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} FULL-RES OUTPUT:")
        if input_size[1] == input_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    if operator in [op.ELTWISE_ADD, op.ELTWISE_SUB]:
        stats.add += (operands - 1) * out_buf.size
    elif operator == op.ELTWISE_MUL:
        stats.mul += (operands - 1) * out_buf.size
    elif operator in [op.ELTWISE_OR, op.ELTWISE_XOR]:
        stats.bitwise += (operands - 1) * out_buf.size

    if output_width != 32:
        if operator == op.ELTWISE_MUL:
            out_buf = np.floor(0.5 + out_buf / (16*quantization)).astype(np.int64). \
                clip(-(2**(bits-1)), 2**(bits-1)-1)
        else:
            np.clip(out_buf, -(2**(bits-1)), 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT:")
            if input_size[1] == input_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    return out_buf, input_size


def pooling_layer(
        layer,  # pylint: disable=unused-argument
        verbose,
        input_size,
        pool,
        pool_stride,
        pool_average,
        data,
        debug=False,
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
        rounding=False,
):
    """
    Perform pooling for one layer.
    """
    if pool[0] > 1 or pool[1] > 1:
        if operation != op.CONV1D:
            in_chan = input_size[0] // operands

            pooled_size = [input_size[0],
                           (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0],
                           (input_size[2] + pool_stride[1] - pool[1]) // pool_stride[1]]
            pooled = pool2d(data, input_size, pooled_size, pool, pool_stride, pool_average,
                            floor=not rounding, debug=debug)
            if verbose:
                if operands == 1:
                    print_data(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                               f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                               f"{input_size} -> {pooled_size}:",
                               pooled,
                               pooled_size,
                               expand,
                               expand_thresh)
                else:
                    d = np.split(pooled,
                                 operands,
                                 axis=0)
                    for i in range(operands):
                        print_data(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                                   f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                                   f"{[in_chan, input_size[1], input_size[2]]} -> "
                                   f"{[in_chan, pooled_size[1], pooled_size[2]]}, "
                                   f"POOLED DATA {i}:",
                                   d[i],
                                   [in_chan, pooled_size[1], pooled_size[2]],
                                   expand,
                                   expand_thresh)

            if pool_average:
                stats.add += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
            else:
                stats.comp += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
        else:
            pooled_size = [input_size[0],
                           (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0]]
            pooled = pool1d(data, input_size, pooled_size, pool[0], pool_stride[0],
                            pool_average, floor=not rounding, debug=debug)
            if verbose:
                print(f"{pool[0]} {'AVERAGE' if pool_average else 'MAX'} "
                      f"POOLING, STRIDE {pool_stride[0]} "
                      f"{input_size} -> {pooled_size}:")
                print(pooled)
                print('')

            if pool_average:
                stats.add += pool[0] * pooled_size[0] * pooled_size[1]
            else:
                stats.comp += pool[0] * pooled_size[0] * pooled_size[1]

    else:
        pooled_size = input_size
        pooled = data

    return pooled, pooled_size


def show_data(
        layer,
        verbose,
        input_size,
        data,
        debug=False,  # pylint: disable=unused-argument
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
):
    """
    Show input data.
    """
    if verbose:
        if expand_thresh is None:
            expand_thresh = input_size[0]

        if operation != op.CONV1D:
            if operands == 1:
                op_string = f"LAYER {layer} ({op.string(operation).upper()})...\n"
                in_chan = input_size[0]
            else:
                op_string = f"LAYER {layer} ({op.string(operation).upper()}, " \
                            f"{operands} OPERANDS)...\n"
                in_chan = input_size[0] // operands
            print(op_string)

            if operands == 1:
                print_data(f"{in_chan}x{input_size[1]}x{input_size[2]} INPUT DATA:",
                           data,
                           [in_chan, input_size[1], input_size[2]],
                           expand,
                           expand_thresh)
            else:
                d = np.split(data.reshape(input_size[0], input_size[1], input_size[2]),
                             operands,
                             axis=0)
                for i in range(operands):
                    print_data(f"{in_chan}x{input_size[1]}x{input_size[2]} INPUT DATA {i}:",
                               d[i],
                               [in_chan, input_size[1], input_size[2]],
                               expand,
                               expand_thresh)
        else:
            print(f"LAYER {layer} ({op.string(operation).upper()})...\n")
            print(f"{input_size[0]}x{input_size[1]} INPUT DATA:")
            print(np.squeeze(data))
            print('')
