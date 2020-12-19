###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Bias related functions
"""
import numpy as np

import tornadocnn as tc
from eprint import eprint, wprint
from utils import argmin

_INVALID_VALUE = -(2**63)


def combine(
        b,
        quantization,
        start,
        out_chan,
):
    """
    When quantizing, combine multiple bias values `b` based on `quantization`. The first kernel
    index is `start`. `out_chan` is used to determine whether to pad the result with zeros,
    if necessary.
    Returns the combined bias values.
    """
    val = 0
    for i in range(8 // quantization):
        if start + i < out_chan:
            this_bias = b[start + i] & (2**quantization-1)
            val |= this_bias << (i * quantization)

    return val


def load(
        verbose,  # pylint: disable=unused-argument
        embedded_code,
        apb,
        layers,
        bias,
        quantization,  # pylint: disable=unused-argument
        group_map,
        output_chan,
        streaming,
        debug,  # pylint: disable=unused-argument
):
    """
    Write `bias` values for the network to C code.
    """
    # Bias: Each group has one bias memory (size BIAS_SIZE bytes). Use only the bias memory in
    # one selected group for the layer, and only if the layer uses a bias. Keep track of the
    # offsets so they can be programmed into the mask count register later.

    if embedded_code:
        bias_values = np.zeros((tc.dev.P_NUMGROUPS, tc.dev.BIAS_SIZE), dtype=np.int64)

    if not embedded_code:
        apb.function_header(function='load_bias')

    group_bias_max = [0] * tc.dev.P_NUMGROUPS
    bias_offs = [None] * layers
    bias_group = [None] * layers
    for ll in range(layers):
        if bias[ll] is None:
            continue
        if len(bias[ll]) != output_chan[ll]:
            eprint(f'Layer {ll}: output channel count {output_chan[ll]} does not match the number '
                   f'of bias values {len(bias[ll])}.')
        if not np.any(bias[ll] != 0):
            wprint(f'Layer {ll}: All bias values are zero. Ignoring the input.')
            continue

        q = 8  # Fixed to 8 bits instead of quantization[ll]
        qfactor = 8 // q
        # Round up the divided length of bias values
        # FIXME: Is it necessary to handle gaps in the next layer?
        bias_len = (output_chan[ll] + qfactor-1) // qfactor

        if ll == 0 and streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            # Work around a problem on AI85
            bias_len += 1
        if streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            wprint(f'Layer {ll} uses streaming and a bias. '
                   'THIS COMBINATION MIGHT NOT BE FUNCTIONING CORRECTLY!!!')

        # Pick the group with the least amount of data in it
        group = argmin(group_bias_max[t] for t in group_map[ll])
        if group_bias_max[group] + bias_len > tc.dev.BIAS_SIZE:
            eprint(f'Layer {ll}: bias memory capacity exceeded - available groups: '
                   f'{group_map[ll]}, used so far: {group_bias_max}, needed: {bias_len}.')
        bias_group[ll] = group
        bias_offs[ll] = group_bias_max[group]
        # Each layer has output_channel number of bias values
        i = 0
        target_offs = 0
        if ll == 0 and streaming[ll] and tc.dev.FIX_STREAM_BIAS:
            # Work around a problem on AI85
            if not embedded_code:
                apb.write_bias(group, bias_offs[ll], 0)
            else:
                # Store for later
                bias_values[group][bias_offs[ll]] = 0
            target_offs += 1
        while i < output_chan[ll]:
            b = combine(bias[ll], q, i, output_chan[ll])
            if not embedded_code:
                apb.write_bias(group, bias_offs[ll] + target_offs, b)
            else:
                # Store for later
                bias_values[group][bias_offs[ll] + target_offs] = b & 0xff
            i += qfactor
            target_offs += 1
        group_bias_max[group] += bias_len

    if embedded_code:
        if max(group_bias_max) > 0:
            # At least one bias value exists, output defines
            for group in range(tc.dev.P_NUMGROUPS):
                if group_bias_max[group] == 0:
                    continue  # but not for this group
                apb.output_define(bias_values[group][:group_bias_max[group]], f'BIAS_{group}',
                                  '0x%02x', 16)
            # Output variables
            for group in range(tc.dev.P_NUMGROUPS):
                if group_bias_max[group] == 0:
                    continue
                apb.output(f'static const uint8_t bias_{group}[] = BIAS_{group};\n', embedded_code)
            apb.output('\n', embedded_code)

            # Finally, create function and do memcpy()
            apb.function_header(prefix='', function='memcpy_8to32', return_type='static void',
                                arguments='uint32_t *dst, const uint8_t *src, int n')
            apb.output('  while (n-- > 0) {\n    *dst++ = *src++;\n  }\n', embedded_code)
            apb.function_footer(return_value='void')

            apb.function_header(function='load_bias')
            for group in range(tc.dev.P_NUMGROUPS):
                if group_bias_max[group] == 0:
                    continue
                addr = apb.apb_base + tc.dev.C_GROUP_OFFS*group + tc.dev.C_BRAM_BASE
                apb.output(f'  memcpy_8to32((uint32_t *) 0x{addr:08x}, bias_{group}, '
                           f'sizeof(uint8_t) * {group_bias_max[group]});\n', embedded_code)
        else:
            apb.function_header(function='load_bias')
            apb.output('  // Not used in this network', embedded_code)

    apb.function_footer()  # load_bias()

    return bias_offs, bias_group, group_bias_max
