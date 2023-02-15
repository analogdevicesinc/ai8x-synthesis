###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Simulate a single CNN layer
"""
import os

import numpy as np

from . import op, state, stats
from . import tornadocnn as tc
from .compute import conv1d, conv2d, convtranspose2d, eltwise, linear, pool1d, pool2d
from .names import layer_str


def print_data(
        verbose_data,
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

    print(header, end='')
    if verbose_data:
        print(':')
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


def print_data1d(
        verbose_data,
        header,
        data,
        step=16,
):
    """
    Print 1-dimensional `data` `step`-elements at a time, prefixed by `header`.
    This function is intended for bias data.
    """
    size = len(data) if data is not None else 0

    if verbose_data:
        if size <= step:
            print(f'{header}:', data)
        else:
            int8_format = '{0:4}' if np.any(data < 0) else '{0:3}'

            print(f'{header}:')
            with np.printoptions(formatter={'int': int8_format.format}):
                for i in range(0, size, step):
                    last = min(i + step, size)
                    if last - 1 > i:
                        print(f'Output channels #{i} to #{last-1}:')
                    else:
                        print(f'Output channel #{i}"')
                    print(np.squeeze(data[i:last]))
        print('')
    elif size > 0:
        print(f"\n{header} SIZE: {size}")
    else:
        print('')


def conv2d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        stride,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform 2D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S)", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(":")
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                for i in range(output_channels):
                    if kernel_size[0] == kernel_size[1] == 1:
                        print(f'Output channel #{i}')
                        print(np.squeeze(kernel[i]))
                    else:
                        if kernel[i].shape[0] < 8:
                            print(f'Output channel #{i}')
                            print(kernel[i])
                        else:
                            for j in range(0, kernel[i].shape[0], 8):
                                print(f'Output channel #{i} (input channels {j}-'
                                      f'{min(kernel[i].shape[0], j+8) - 1})')
                                print(kernel[i][j:j+8])
        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - dilation[0] * (kernel_size[0] - 1) - 1 +
                 2 * padding[0]) // stride[0] + 1,
                (input_size[2] - dilation[1] * (kernel_size[1] - 1) - 1 +
                 2 * padding[1]) // stride[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv2d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
        fractional_stride=[1, 1],
        output_pad=[0, 0],
        groups=groups,
    )

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        if out_size[1] == out_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size[0] * kernel_size[1] * out_size[0]
        * out_size[1] * out_size[2],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1] * out_size[2],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


def convtranspose2d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        fractional_stride,
        output_padding,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform a fractionally strided 2D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"{kernel_size[0]}x{kernel_size[1]} KERNEL(S)", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(':')
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                for i in range(output_channels):
                    if kernel_size[0] == kernel_size[1] == 1:
                        print(f'Output channel #{i}')
                        print(np.squeeze(kernel[i]))
                    else:
                        if kernel[i].shape[0] < 8:
                            print(f'Output channel #{i}')
                            print(kernel[i])
                        else:
                            for j in range(0, kernel[i].shape[0], 8):
                                print(f'Output channel #{i} (input channels {j}-'
                                      f'{min(kernel[i].shape[0], j+8) - 1})')
                                print(kernel[i][j:j+8])

        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - 1) * fractional_stride[0] - 2 * padding[0]
                + dilation[0] * (kernel_size[0] - 1)
                + output_padding[0] + 1,
                (input_size[2] - 1) * fractional_stride[1] - 2 * padding[1]
                + dilation[1] * (kernel_size[1] - 1)
                + output_padding[1] + 1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = convtranspose2d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=[1, 1],
        pad=padding,
        dilation=dilation,
        fractional_stride=fractional_stride,
        output_pad=output_padding,
        groups=groups,
    )

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} FULL-RES OUTPUT:")
        if out_size[1] == out_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size[0] * kernel_size[1] * out_size[0]
        * out_size[1] * out_size[2],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            if out_size[1] == out_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1] * out_size[2],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]}x{out_size[2]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


def conv1d_layer(
        layer,
        input_size,
        kernel_size,
        output_shift,
        output_channels,
        padding,
        dilation,
        stride,
        activation,
        kernel,
        bias,
        data,
        bits=8,
        output_width=8,
        groups=1,
        bypass=False,
        datafile=None,
):
    """
    Perform 1D convolution for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    if state.verbose:
        print(f"KERNEL SIZE {kernel_size}", end='')
        if bypass:
            print(' (BYPASS)')
        if state.verbose_all and not bypass:
            print(':')
            with np.printoptions(formatter={'int': state.kernel_format.format}):
                print(kernel)
        print_data1d(state.verbose_all, "BIAS", bias)

    out_size = [output_channels,
                (input_size[1] - dilation * (kernel_size - 1) - 1 +
                 2 * padding) // stride + 1,
                1]

    if bias is not None:
        bias = bias * tc.dev.BIAS_DIV

    out_buf = conv1d(
        data=data,
        weight=kernel,
        bias=bias,
        input_size=input_size,
        output_size=out_size,
        kernel_size=kernel_size,
        stride=stride,
        pad=padding,
        dilation=dilation,
        fractional_stride=1,
        output_pad=0,
        groups=groups,
    )[:, :, np.newaxis]

    if datafile is not None:
        np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

    if state.verbose and verbose_data:
        print(f"{out_size[0]}x{out_size[1]} FULL-RES OUTPUT:")
        print(out_buf.squeeze(axis=-1))
        print('')

    stats.account(
        layer,
        "macc",
        (input_size[0] // groups) * kernel_size * out_size[0] * out_size[1],
    )

    if output_width != 32:
        out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
            clip(-(2**(bits-1)), 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]} OUTPUT "
                  f"{'BEFORE ACTIVATION' if activation is not None else '(NO ACTIVATION)'}:")
            print(out_buf.squeeze(axis=-1))
            print('')

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"{out_size[0]}x{out_size[1]} ACTIVATED OUTPUT"
                  f" ({op.act_string(activation).upper()}):")
            print(out_buf.squeeze(axis=-1))
            print('')

        stats.account(
            layer,
            "comp",
            out_size[0] * out_size[1],
        )

    if state.verbose and not verbose_data:
        print(f"{out_size[0]}x{out_size[1]} OUTPUT"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_size


def linear_layer(
        layer,
        activation,
        weight,
        bias,
        data,
        bits=16,
):
    """
    Perform one software linear layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]
    verbose_input = state.verbose_all or layer == state.start_layer \
        or state.in_sequences[layer] is not None and -1 in state.in_sequences[layer]

    in_features = data.shape[0]
    out_features = weight.shape[0]

    if state.verbose_all or verbose_input:
        print("CLASSIFICATION LAYER (LINEAR)...\n")
        print(f"INPUT DATA (size {in_features})", end='')
        if verbose_input:
            print(':')
            print(data)
        print('')

    if state.verbose_all:
        print(f"WEIGHTS (size {in_features * out_features})", end='')
        print(':')
        print(weight)
        print_data1d(state.verbose_all, "BIAS", bias)

    out_buf = linear(
        layer=layer,
        data=data,
        weight=weight,
        bias=bias,
        in_features=in_features,
        out_features=out_features,
    )
    out_buf = np.floor(0.5 + out_buf / 128).astype(np.int64). \
        clip(-(2**(bits-1)), 2**(bits-1)-1)

    if state.verbose and verbose_data:
        print(f"OUTPUT (size {out_features}):")
        print(out_buf)
        print('')

    stats.account(
        layer,
        "sw_macc",
        in_features * out_features,
    )

    if activation is not None:
        if activation == op.ACT_RELU:
            np.clip(out_buf, 0, 2**(bits-1)-1, out_buf)
        elif activation == op.ACT_ABS:
            out_buf = np.abs(out_buf).clip(0, 2**(bits-1)-1)

        if state.verbose and verbose_data:
            print(f"ACTIVATED OUTPUT (size {out_features})"
                  f" ({op.act_string(activation).upper()}):")
            print(out_buf)
            print('')

        stats.account(
            layer,
            "sw_comp",
            out_features,
        )

    if state.verbose and not verbose_data:
        print(f"OUTPUT (size {out_features})"
              f" ({op.act_string(activation).upper()})\n")

    return out_buf, out_features


def passthrough_layer(
        layer,  # pylint: disable=unused-argument
        input_size,
        data,
        datafile=None,
):
    """
    2D passthrough for one layer.
    """
    if datafile is not None:
        np.save(datafile, np.empty((0)), allow_pickle=False, fix_imports=False)

    return data, input_size


def eltwise_layer(
        operator,
        layer,
        input_size,
        output_shift,
        data,
        output_width=8,
        operands=1,
):
    """
    Element-wise operators for one layer.
    """
    verbose_data = state.verbose_all or state.output_layer[layer]

    bits = 8
    assert operands == len(data)

    if state.verbose:
        print(f"{operands}-OPERAND {op.string(operator, elt=True).upper()}:\n")

    out_buf = eltwise(
        operator=operator,
        data=data,
        input_size=input_size,
    )

    if state.verbose and verbose_data:
        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} FULL-RES OUTPUT:")
        if input_size[1] == input_size[2] == 1:
            print(np.squeeze(out_buf))
        else:
            print(out_buf)
        print('')

    if operator in [op.ELTWISE_ADD, op.ELTWISE_SUB]:
        stats.account(
            layer,
            "add",
            (operands - 1) * out_buf.size,
        )
    elif operator == op.ELTWISE_MUL:
        stats.account(
            layer,
            "mul",
            (operands - 1) * out_buf.size,
        )
    elif operator in [op.ELTWISE_OR, op.ELTWISE_XOR]:
        stats.account(
            layer,
            "bitwise",
            (operands - 1) * out_buf.size,
        )

    if output_width != 32:
        if operator == op.ELTWISE_MUL:
            out_buf = np.floor(0.5 + out_buf / (128 / 2.0**output_shift)).astype(np.int64). \
                clip(-(2**(bits-1)), 2**(bits-1)-1)
        else:
            np.clip(out_buf, -(2**(bits-1)), 2**(bits-1)-1, out_buf)

        if state.verbose and verbose_data:
            print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT:")
            if input_size[1] == input_size[2] == 1:
                print(np.squeeze(out_buf))
            else:
                print(out_buf)
            print('')

    if state.verbose and not verbose_data:
        print(f"{input_size[0]}x{input_size[1]}x{input_size[2]} OUTPUT")

    return out_buf, input_size


def pooling_layer(
        layer,
        input_size,
        pool,
        pool_stride,
        pool_average,
        data,
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
        rounding=False,
        debug_data=None,
        dilation=(1, 1),
):
    """
    Perform pooling for one layer.
    """
    # Always apply stride
    if operation != op.CONV1D:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0]
                        - pool[0] - dilation[0] + 1) // pool_stride[0],
                       (input_size[2] + pool_stride[1]
                        - pool[1] - dilation[1] + 1) // pool_stride[1]]
    else:
        pooled_size = [input_size[0],
                       (input_size[1] + pool_stride[0] - pool[0]
                        - dilation[0] + 1) // pool_stride[0]]

    # Actual pooling operation?
    if pool[0] > 1 or pool[1] > 1:
        if operation != op.CONV1D:
            pooled = np.empty((operands, pooled_size[0], pooled_size[1], pooled_size[2]),
                              dtype=np.int64)
            for i in range(operands):
                if debug_data is not None:
                    for j in range(input_size[0]):
                        np.savetxt(os.path.join(debug_data, f"unpooled-{i}-L{layer}-ch{j}.csv"),
                                   data[i][j, :, :], delimiter=",")
                pooled[i] = pool2d(
                    data[i],
                    input_size,
                    pooled_size,
                    pool,
                    pool_stride,
                    pool_average,
                    dilation=dilation,
                    floor=not rounding,
                )
                if state.verbose:
                    if dilation[0] > 1 or dilation[1] > 1:
                        dilation_str = f", DILATION {dilation[0]}/{dilation[1]}"
                    else:
                        dilation_str = ''
                    print_data(
                        state.verbose_all,
                        f"{'AVERAGE' if pool_average else 'MAX'} "
                        f"POOL {pool[0]}x{pool[1]} WITH STRIDE {pool_stride[0]}/{pool_stride[1]}"
                        + dilation_str +
                        f" {input_size} -> {pooled_size}"
                        + (f", POOLED DATA {i}" if operands > 1 else ""),
                        pooled[i],
                        pooled_size,
                        expand,
                        expand_thresh,
                    )
                if debug_data is not None:
                    for j in range(pooled_size[0]):
                        np.savetxt(os.path.join(debug_data, f"pooled-{i}-L{layer}-ch{j}.csv"),
                                   pooled[i][j, :, :], delimiter=",")

            st = pool[0] * pool[1] * pooled_size[0] * pooled_size[1] * pooled_size[2] * operands
            if pool_average:
                stats.account(
                    layer,
                    "add",
                    st,
                )
            else:
                stats.account(
                    layer,
                    "comp",
                    st,
                )
        else:
            pooled = pool1d(
                data[0],
                input_size,
                pooled_size,
                pool[0],
                pool_stride[0],
                pool_average,
                dilation=dilation[0],
                floor=not rounding,
            )
            if state.verbose:
                print(f"{'AVERAGE' if pool_average else 'MAX'} "
                      f"POOL {pool[0]} WITH STRIDE {pool_stride[0]} ", end='')
                if dilation[0] > 1:
                    print(f", DILATION {dilation[0]} ", end='')
                print(f"{input_size} -> {pooled_size}", end='')
                if state.verbose_all:
                    print(':')
                    print(pooled)
                print('')

            if pool_average:
                stats.account(
                    layer,
                    "add",
                    pool[0] * pooled_size[0] * pooled_size[1],
                )
            else:
                stats.account(
                    layer,
                    "comp",
                    pool[0] * pooled_size[0] * pooled_size[1],
                )

            pooled = np.expand_dims(pooled, axis=0)

    else:
        # Use pool_stride only
        if operation != op.CONV1D:
            pooled = data[:, :, ::pool_stride[0], ::pool_stride[1]]
            if pool_stride[0] > 1 or pool_stride[1] > 1:
                if state.verbose:
                    print(f"{'AVERAGE' if pool_average else 'MAX'} "
                          f"POOL {pool[0]}x{pool[1]} WITH STRIDE {pool_stride[0]}/{pool_stride[1]}"
                          f" {input_size} -> {pooled_size}", end='')
                    if state.verbose_all:
                        print(':')
                        print(pooled)
                    print('')
        else:
            pooled = data[:, :, ::pool_stride[0]]
            if pool_stride[0] > 1:
                if state.verbose:
                    print(f"{'AVERAGE' if pool_average else 'MAX'} "
                          f"POOL {pool[0]} WITH STRIDE {pool_stride[0]} "
                          f"{input_size} -> {pooled_size}", end='')
                    if state.verbose_all:
                        print(':')
                        print(pooled)
                    print('')

    return pooled, pooled_size


def show_data(
        layer,
        input_size,
        data,
        expand=None,
        expand_thresh=None,
        operation=None,
        operands=1,
):
    """
    Show input data.
    """
    if state.verbose:
        verbose_input = state.verbose_all or layer == state.start_layer \
            or state.in_sequences[layer] is not None and -1 in state.in_sequences[layer]

        if expand_thresh is None:
            expand_thresh = input_size[0]

        if operation != op.CONV1D:
            if operands == 1:
                op_string = f"LAYER {layer_str(layer)} ({op.string(operation).upper()})...\n"
            else:
                op_string = f"LAYER {layer_str(layer)} ({op.string(operation).upper()}, " \
                            f"{operands} OPERANDS)...\n"
            print(op_string)

            if operands == 1:
                print_data(verbose_input,
                           f"{data.shape[1]}x{data.shape[2]}x{data.shape[3]} INPUT DATA",
                           data[0],
                           [data.shape[1], data.shape[2], data.shape[3]],
                           expand,
                           expand_thresh)
            else:
                for i in range(operands):
                    print_data(verbose_input,
                               f"{data.shape[1]}x{data.shape[2]}x{data.shape[3]} INPUT DATA {i}",
                               data[i],
                               [data.shape[1], data.shape[2], data.shape[3]],
                               expand,
                               expand_thresh)
        else:
            print(f"LAYER {layer_str(layer)} ({op.string(operation).upper()})...\n")
            print(f"{input_size[1]}x{input_size[2]} INPUT DATA", end='')
            if verbose_input:
                print(':')
                print(np.squeeze(data))
            print('')
