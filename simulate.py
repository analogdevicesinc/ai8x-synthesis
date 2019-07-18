###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Simulate a single CNN layer
"""
import numpy as np

import stats
import tornadocnn as tc
from compute import conv1d, conv2d, linear


def cnn2d_layer(layer, verbose,
                input_size, kernel_size, quantization,
                output_channels, padding, dilation, stride,
                pool, pool_stride, pool_average, do_activation,
                kernel, bias, data, bits=8, output_width=8,
                device=84, debug=False,  # pylint: disable=unused-argument
                expand=None, expand_thresh=None):
    """
    Perform 2D pooling and 2D convolution for one layer.
    """
    int8_format = '{0:4}' if np.any(data < 0) else '{0:3}'
    if verbose:
        if expand_thresh is None:
            expand_thresh = input_size[0]
        print(f"LAYER {layer}...\n")

        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} INPUT DATA:")
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

    if pool[0] > 1 or pool[1] > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]) // pool_stride[0],
                       (input_size[2] + pool_stride[1] - pool[1]) // pool_stride[1]]
        pooled = np.empty(shape=(pooled_size[0], pooled_size[1], pooled_size[2]),
                          dtype=np.int64)
        for c in range(input_size[0]):
            for row in range(0, pooled_size[1]*pool_stride[0], pool_stride[0]):
                for col in range(0, pooled_size[2]*pool_stride[1], pool_stride[1]):
                    if pool_average:
                        avg = np.average(data[c][row:row+pool[0], col:col+pool[1]])
                        if avg < 0:
                            val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                        else:
                            val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.amax(data[c][row:row+pool[0], col:col+pool[1]])
                    pooled[c][row//pool_stride[0]][col//pool_stride[1]] = val
        if verbose:
            print(f"{pool[0]}x{pool[1]} {'AVERAGE' if pool_average else 'MAX'} "
                  f"POOLING, STRIDE {pool_stride[0]}/{pool_stride[1]} "
                  f"{input_size} -> {pooled_size}:")
            with np.printoptions(formatter={'int': int8_format.format}):
                for i in range(input_size[0]):
                    print(f'Channel #{i}', end='')
                    if expand and expand > 1:
                        print(f' (expansion: {(i // expand_thresh) + 1} of {expand})')
                    else:
                        print('')
                    print(pooled[i])
            print('')

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

    kernel = kernel.reshape((output_channels, input_size[0], -1))
    pooled = pooled.reshape((pooled_size[0], -1))

    out_size = [output_channels,
                (pooled_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (pooled_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]
    out_buf = np.full(shape=(out_size[0], out_size[1]*out_size[2]),
                      fill_value=np.nan, dtype=np.int64)

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    conv2d(data=pooled,
           weight=kernel,
           bias=bias,
           input_size=pooled_size,
           out_channels=output_channels,
           kernel_size=kernel_size,
           stride=stride,
           pad=padding,
           dilation=dilation,
           output=out_buf,
           debug=debug)

    out_buf = out_buf.reshape((out_size))

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


def cnn1d_layer(layer, verbose,
                input_size, kernel_size, quantization,
                output_channels, padding, dilation, stride,
                pool, pool_stride, pool_average, do_activation,
                kernel, bias, data, bits=8, output_width=8,
                device=84, debug=False):  # pylint: disable=unused-argument
    """
    Perform 1D pooling and 1D convolution for one layer.
    """
    if verbose:
        print(f"LAYER {layer}...\n")

        print(f"{input_size[0]}x{input_size[1]} INPUT DATA:")
        print(np.squeeze(data))
        print('')

    if pool > 1:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride - pool) // pool_stride]
        pooled = np.empty(shape=(pooled_size),
                          dtype=np.int64)
        for c in range(input_size[0]):
            for x in range(0, pooled_size[1]*pool_stride, pool_stride):
                if pool_average:
                    avg = np.average(data[c][x:x+pool])
                    if avg < 0:
                        val = np.ceil(avg).astype(np.int64).clip(min=-128, max=127)
                    else:
                        val = np.floor(avg).astype(np.int64).clip(min=-128, max=127)
                else:
                    val = np.amax(data[c][x:x+pool])
                pooled[c][x//pool_stride] = val
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

    kernel = kernel.reshape((output_channels, input_size[0], -1))
    pooled = pooled.reshape((pooled_size[0], -1))

    out_size = [output_channels,
                (pooled_size[1] - dilation * (kernel_size - 1) - 1 +
                 2 * padding) // stride + 1,
                1]
    out_buf = np.full(shape=(out_size[0], out_size[1]),
                      fill_value=np.nan, dtype=np.int64)

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    conv1d(data=pooled,
           weight=kernel,
           bias=bias,
           input_size=pooled_size,
           out_channels=output_channels,
           kernel_size=kernel_size,
           stride=stride,
           pad=padding,
           dilation=dilation,
           output=out_buf,
           debug=debug)

    out_buf = out_buf.reshape((out_size))

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


def linear_layer(verbose, do_activation,
                 weight, bias, data, bits=16, debug=False):
    """
    Perform one linear layer.
    """
    in_features = data.shape[0]
    out_features = weight.shape[0]

    if verbose:
        print("CLASSIFICATION LAYER...\n")
        print(f"INPUT DATA (size {in_features}):")
        print(data)
        print('')

        print(f"WEIGHTS (size {in_features * out_features}):")
        print(weight)
        print(f"BIAS: {bias}\n")

    out_buf = np.empty(out_features, dtype=np.int64)
    linear(data=data, weight=weight, bias=bias,
           in_features=in_features, out_features=out_features,
           output=out_buf, debug=debug)
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
