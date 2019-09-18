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


def print_data(header,
               data,
               input_size,
               expand,
               expand_thresh):
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


def cnn2d_layer(layer,
                verbose,
                input_size,
                kernel_size,
                quantization,
                output_channels,
                padding,
                dilation,
                stride,
                pool,
                pool_stride,
                pool_average,
                do_activation,
                kernel,
                bias,
                data,
                bits=8,
                output_width=8,
                device=84,  # pylint: disable=unused-argument
                debug=False,
                expand=None,
                expand_thresh=None):
    """
    Perform 2D pooling and 2D convolution for one layer.
    """
    if verbose:
        if expand_thresh is None:
            expand_thresh = input_size[0]
        print_data(f"LAYER {layer} (CONV2D)...\n\n"
                   f"{input_size[0]}x{input_size[1]}x{input_size[2]} INPUT DATA:",
                   data,
                   input_size,
                   expand,
                   expand_thresh)

    if pool[0] > 1 or pool[1] > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0],
                       (input_size[2] + pool_stride[1] - pool[1]) // pool_stride[1]]
        pooled = pool2d(data, input_size, pooled_size, pool, pool_stride, pool_average,
                        floor=True, debug=debug)  # FIXME: Fix rounding for AI85?
        if verbose:
            print_data(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                       f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                       f"{input_size} -> {pooled_size}:",
                       pooled,
                       pooled_size,
                       expand,
                       expand_thresh)

        if pool_average:
            stats.add += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
        else:
            stats.comp += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
    else:
        pooled_size = input_size
        pooled = data

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
                (pooled_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (pooled_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv2d(data=pooled,
                     weight=kernel,
                     bias=bias,
                     input_size=pooled_size,
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

    stats.macc += pooled_size[0] * kernel_size[0] * kernel_size[1] * out_size[0] \
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


def cnn1d_layer(layer,
                verbose,
                input_size,
                kernel_size,
                quantization,
                output_channels,
                padding,
                dilation,
                stride,
                pool,
                pool_stride,
                pool_average,
                do_activation,
                kernel,
                bias,
                data,
                bits=8,
                output_width=8,
                device=84,  # pylint: disable=unused-argument
                debug=False):
    """
    Perform 1D pooling and 1D convolution for one layer.
    """
    if verbose:
        print(f"LAYER {layer} (CONV1D)...\n")

        print(f"{input_size[0]}x{input_size[1]} INPUT DATA:")
        print(np.squeeze(data))
        print('')

    if pool > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride - pool) // pool_stride]
        pooled = pool1d(data, input_size, pooled_size, pool, pool_stride, pool_average,
                        floor=True, debug=debug)  # FIXME: Fix rounding for AI85?
        if verbose:
            print(f"{pool} {'AVERAGE' if pool_average else 'MAX'} "
                  f"POOLING, STRIDE {pool_stride} "
                  f"{input_size} -> {pooled_size}:")
            print(pooled)
            print('')

        if pool_average:
            stats.add += pool * pooled_size[0] * pooled_size[1]
        else:
            stats.comp += pool * pooled_size[0] * pooled_size[1]
    else:
        pooled_size = input_size
        pooled = data

    if verbose:
        print(f"KERNEL SIZE {kernel_size}:")
        print(kernel)
        print(f"BIAS: {bias}\n")

    out_size = [output_channels,
                (pooled_size[1] - dilation * (kernel_size - 1) - 1 +
                 2 * padding) // stride + 1,
                1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv1d(data=pooled,
                     weight=kernel,
                     bias=bias,
                     input_size=pooled_size,
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

    stats.macc += pooled_size[0] * kernel_size * out_size[0] \
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


def linear_layer(verbose,
                 do_activation,
                 weight,
                 bias,
                 data,
                 bits=16,
                 debug=False):
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


def passthrough_layer(layer,
                      verbose,
                      input_size,
                      pool,
                      pool_stride,
                      pool_average,
                      data,
                      device=84,  # pylint: disable=unused-argument
                      debug=False,
                      expand=None,
                      expand_thresh=None):
    """
    2D pooling or passthrough for one layer.
    """
    if verbose:
        if expand_thresh is None:
            expand_thresh = input_size[0]
        print_data(f"LAYER {layer} (PASSTHROUGH)...\n\n"
                   f"{input_size[0]}x{input_size[1]}x{input_size[2]} INPUT DATA:",
                   data,
                   input_size,
                   expand,
                   expand_thresh)

    if pool[0] > 1 or pool[1] > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0],
                       (input_size[2] + pool_stride[1] - pool[1]) // pool_stride[1]]
        pooled = pool2d(data, input_size, pooled_size, pool, pool_stride, pool_average,
                        floor=True, debug=debug)  # FIXME: Fix rounding for AI85?
        if verbose:
            print_data(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                       f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                       f"{input_size} -> {pooled_size}:",
                       pooled,
                       pooled_size,
                       expand,
                       expand_thresh)

        if pool_average:
            stats.add += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
        else:
            stats.comp += pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2]
    else:
        pooled_size = input_size
        pooled = data

    return pooled, pooled_size


def eltwise_layer(operator,
                  layer,
                  verbose,
                  input_size,
                  do_activation,
                  data,
                  output_width=8,
                  device=84,  # pylint: disable=unused-argument
                  debug=False,
                  expand=None,
                  operands=1,
                  expand_thresh=None):
    """
    Element-wise operators for one layer.
    """
    quantization = bits = 8
    assert operands == len(data)

    if verbose:
        if expand_thresh is None:
            expand_thresh = input_size[0]
        print(f"LAYER {layer} ({operands}-OPERAND {op.string(operator).upper()})...\n")
        for i in range(operands):
            print_data(f"{input_size[0]}x{input_size[1]}x{input_size[2]} INPUT DATA {i}:",
                       data[i],
                       input_size,
                       expand,
                       expand_thresh)

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
            print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if do_activation else '(NO ACTIVATION)'}:")
            if input_size[1] == input_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if do_activation:
        np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)

        if verbose:
            print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} ACTIVATED OUTPUT:")
            if input_size[1] == input_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.comp += input_size[0] * input_size[1] * input_size[2]

    return out_buf, input_size
