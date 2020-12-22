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
from utils import argmin, ffs, nthone, popcount

_INVALID_VALUE = -(2**63)


def load(
        verbose,  # pylint: disable=unused-argument
        embedded_code,
        apb,
        start_layer,
        layers,
        bias,
        group_map,
        output_chan,
        streaming,
        conv_groups,
        broadcast_mode,
        output_processor_map,
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
    bias_offs = [[None] * tc.dev.P_NUMGROUPS for _ in range(layers)]
    bias_group = [None] * layers
    for ll in range(start_layer, layers):
        if bias[ll] is None or group_map[ll] is None:
            continue
        if len(bias[ll]) != output_chan[ll]:
            eprint(f'Layer {ll}: output channel count {output_chan[ll]} does not match the number '
                   f'of bias values {len(bias[ll])}.')
        if not np.any(bias[ll] != 0):
            wprint(f'Layer {ll}: All bias values are zero. Ignoring the input.')
            continue

        # Round up the divided length of bias values
        # FIXME: Is it necessary to handle gaps in the next layer?
        bias_len = output_chan[ll]

        if ll == 0 and streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            # Work around a problem on AI85
            bias_len += 1
        if streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            wprint(f'Layer {ll} uses streaming and a bias. '
                   'THIS COMBINATION MIGHT NOT BE FUNCTIONING CORRECTLY!!!')

        if conv_groups[ll] == 1:
            # Pick the group with the least amount of data in it
            group = argmin(group_bias_max[t] for t in group_map[ll])
            if group_bias_max[group] + bias_len > tc.dev.BIAS_SIZE:
                eprint(f'Layer {ll}: bias memory capacity exceeded - available groups: '
                       f'{group_map[ll]}, used so far: {group_bias_max}, needed: {bias_len}.')
            bias_group[ll] = group
            for i in range(tc.dev.P_NUMGROUPS):
                bias_offs[ll][i] = group_bias_max[group]
            # Each layer has output_channel number of bias values
            i = 0
            target_offs = 0
            if ll == 0 and streaming[ll] and tc.dev.FIX_STREAM_BIAS:
                # Work around a problem on AI85
                if not embedded_code:
                    apb.write_bias(group, bias_offs[ll][group], 0)
                else:
                    # Store for later
                    bias_values[group][bias_offs[ll][group]] = 0
                target_offs += 1
            while i < output_chan[ll]:
                if not embedded_code:
                    apb.write_bias(group, bias_offs[ll][group] + target_offs, bias[ll][i] & 0xff)
                else:
                    # Store for later
                    bias_values[group][bias_offs[ll][group] + target_offs] = bias[ll][i] & 0xff
                i += 1
                target_offs += 1
            group_bias_max[group] += bias_len
        else:
            # Each group needs to have 'its own' bias values
            bias_group[ll] = 'all'
            chan = 0

            def bias_add_byte(layer, group, val):
                """
                Add a single bias byte `val` to the memory in `group`.
                """
                if group_bias_max[group] >= tc.dev.BIAS_SIZE:
                    eprint(f'Layer {layer}: bias memory capacity for group {group} exceeded, '
                           f'used so far: {group_bias_max[group]}.')

                if not embedded_code:
                    apb.write_bias(group, group_bias_max[group], val)
                else:
                    # Store for later
                    bias_values[group][group_bias_max[group]] = val
                group_bias_max[group] += 1

            for group in range(tc.dev.P_NUMGROUPS):
                if output_processor_map[ll] >> group * tc.dev.P_NUMPRO & (2**tc.dev.P_NUMPRO - 1):
                    bias_offs[ll][group] = group_bias_max[group]

            pop = popcount(output_processor_map[ll])
            if broadcast_mode[ll]:
                # Pad out to allow for parallel read from the 8-bit memories in broadcast mode
                extend = ffs(output_processor_map[ll]) % tc.dev.P_NUMPRO
                popall = pop + extend
            else:
                extend = 0
            passes = (output_chan[ll] + pop - 1) // pop

            for chan in range(passes * extend + output_chan[ll]):
                if broadcast_mode[ll]:
                    p = chan % popall
                    group = p // tc.dev.P_NUMPRO
                    src = (p & ~0x0f) + (p % 4) * 4 + (p % 16) // 4 - extend
                    if src >= 0:
                        src += chan // popall * pop
                else:
                    p = chan % pop
                    group = nthone(p + 1, output_processor_map[ll]) // tc.dev.P_NUMPRO
                    src = (p & 0x03) + (p >> 2) % passes * pop + (chan // (4 * passes)) * 4

                val = bias[ll][src] & 0xff if src >= 0 and src < len(bias[ll]) else 0
                bias_add_byte(ll, group, val)

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
