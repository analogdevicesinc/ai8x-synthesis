###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Bias related functions
"""
import numpy as np

from . import state
from . import tornadocnn as tc
from .eprint import eprint, wprint
from .names import layer_pfx
from .utils import argmin, ffs, fls, popcount

_INVALID_VALUE = -(2**63)


def load(
        embedded_code,
        apb,
        layers,
        bias,
        group_map,
        bias_group_map,
        output_chan,
        streaming,
        conv_groups,
        broadcast_mode,
        processor_map,
        output_processor_map,
        out_expand,
        out_expand_thresh,
        groups_used,
        flatten,
):
    """
    Write `bias` values for the network to C code.
    """
    # Bias: Each group has one bias memory (size BIAS_SIZE bytes). Use only the bias memory in
    # one selected group for the layer, and only if the layer uses a bias. Keep track of the
    # offsets so they can be programmed into the mask count register later.

    # Cache for faster access
    calcx4 = state.calcx4
    start_layer = state.first_layer_used

    # Initialize with known value
    bias_values = np.full((tc.dev.P_NUMGROUPS, tc.dev.BIAS_SIZE), _INVALID_VALUE, dtype=np.int64)

    if not embedded_code:
        apb.function_header(function='load_bias')

    group_bias_max = [0] * tc.dev.P_NUMGROUPS
    bias_offs = [[None] * tc.dev.P_NUMGROUPS for _ in range(layers)]
    bias_group = [None] * layers

    # Allocate bias groups
    bias_map = []
    bias_len = [0] * layers

    # Allocate all depth-wise and qupac layers first since they are bound to a single group
    for ll in range(start_layer, layers):
        if bias[ll] is None or group_map[ll] is None:
            continue
        # Check all layers, including non-depth-wise
        if len(bias[ll]) != output_chan[ll]:
            eprint(f'{layer_pfx(ll)}Output channel count {output_chan[ll]} does not match the '
                   f'number of bias values {len(bias[ll])}.')
        if not np.any(bias[ll] != 0):
            wprint(f'{layer_pfx(ll)}All bias values are zero. Ignoring the input.')
            continue
        if conv_groups[ll] == 1 and not (state.fast_fifo_quad and ll == start_layer):
            # For regular convolutions, collect length data
            # Round up the divided length of bias values
            bias_len[ll] = output_chan[ll] \
                + ffs(output_processor_map[ll]) % tc.dev.P_SHARED * out_expand[ll]
            if ll == 0 and streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
                bias_len[ll] += 1  # Work around a problem on AI85

            bias_map += [(ll, groups_used if bias_group_map[ll] is None else bias_group_map[ll],
                          bias_len[ll], flatten[ll])]
            continue

        # For depth-wise convolutions, and qupac, each group needs to have 'its own' bias values
        bias_group[ll] = 'all'

        def bias_add_byte(layer, group, val):
            """
            Add a single bias byte `val` to the memory in `group`.
            """
            assert 0 <= group < tc.dev.P_NUMGROUPS_ALL
            if group_bias_max[group] >= tc.dev.BIAS_SIZE:
                eprint(f'{layer_pfx(layer)}Bias memory capacity for group {group} exhausted, '
                       f'used so far: {group_bias_max[group]}.')

            if val is not None and val != _INVALID_VALUE:  # else just consume the space
                # Store for later
                bias_values[group][group_bias_max[group]] = val & 0xff
            group_bias_max[group] += 1

        used_groups = 0
        for group in range(tc.dev.P_NUMGROUPS):
            if processor_map[ll] >> group * tc.dev.P_NUMPRO & (2**tc.dev.P_NUMPRO - 1):
                if broadcast_mode[ll]:  # Round up so we can use all groups in parallel
                    group_bias_max[group] = (group_bias_max[group] + 3) & ~3
                bias_offs[ll][group] = group_bias_max[group]
                used_groups += 1
            else:
                bias_offs[ll][group] = None

        if conv_groups[ll] == 1:
            # Qupac mode
            for i, e in enumerate(bias[ll]):
                bias_add_byte(ll, i % 4, e)
            continue

        map_used = processor_map[ll]
        if not broadcast_mode[ll]:
            # The first gap is 'moved' to the end of the group
            start_proc = ffs(map_used)
            first_group = start_proc // tc.dev.P_NUMPRO
            map_used &= ~((2**tc.dev.P_NUMPRO - 1) << first_group * tc.dev.P_NUMPRO)
            map_used |= (processor_map[ll] & (((2**tc.dev.P_NUMPRO - 1) <<
                         (first_group * tc.dev.P_NUMPRO)))) >> start_proc % tc.dev.P_NUMPRO

        def rearrange_processor(p, bc_mode):
            """
            Rearrange processor `p` for broadcast mode if needed.
            """
            return (p & ~0x0f) + (p % 4) * 4 + (p % 16) // 4 if bc_mode else p

        start_proc = ffs(map_used)
        last_proc = fls(map_used)
        if broadcast_mode[ll]:
            last_proc = ((last_proc + 16) & ~0x0f) - 1
        if broadcast_mode[ll] or used_groups > 1:
            # Pad out to allow for parallel read from the 8-bit memories
            start_proc &= ~(2**tc.dev.P_NUMPRO - 1)

        # Break bias into multiple passes
        bias_pad = bias[ll].copy()
        leftover = out_expand[ll] * out_expand_thresh[ll] - len(bias_pad)
        if leftover != 0:
            # Odd length with leftover unused values
            bias_pad = np.append(bias_pad, np.array([_INVALID_VALUE] * leftover))
        bias_pad = bias_pad.reshape(out_expand[ll], -1)

        assert bias_pad.shape[1] == popcount(processor_map[ll])

        # Insert 'None' elements where processors are not used. This is not needed at the end,
        # and additionally, not needed at the start but only when only one group is used and
        # when not in broadcast mode.

        # Start inserting from the left so the `proc` index will match the bias array
        # Skip last_proc since we already know it is used (do not skip start since it could be
        # unused)
        for proc in range(start_proc, last_proc + 1):
            if map_used >> proc & 1 == 0:
                bias_pad = np.insert(bias_pad, proc,
                                     np.array([_INVALID_VALUE] * out_expand[ll]), axis=1)

        for expand in range(out_expand[ll]):
            for p in range(start_proc, last_proc + 1):
                group = p // tc.dev.P_NUMPRO
                src = rearrange_processor(p, broadcast_mode[ll])
                val = bias_pad[expand][src - start_proc] \
                    if src - start_proc < bias_pad.shape[1] else None
                # Add value, even if it's None
                bias_add_byte(ll, group, val)
        while (group_bias_max[group] > 0
               and bias_values[group][group_bias_max[group] - 1] == _INVALID_VALUE):
            group_bias_max[group] -= 1

    # For regular convolutions, bias allocation is a version of the off-line bin packing problem

    # Reverse-sort the data structure for a Fit-First Descending (FFD) algorithm
    def bias_sort(e):
        """Order bias tuples"""
        # Highest priority: flatten
        if tc.dev.REQUIRE_FLATTEN_BIAS_FIRST and e[3]:  # flatten
            return -1
        # Next highest priority: If only one group is allowed
        if len(e[1]) == 1:
            return 0
        # For all other cases, reverse sort by size; sorted() will keep the original order
        # when keys are identical (i.e., no need to take the layer number into account).
        # This will also automatically give the highest priority to layers with the maximum bias
        # size (and even higher when exceeding the maximum size so the error message will appear
        # early).
        return tc.dev.BIAS_SIZE - e[2]

    bias_map = sorted(bias_map, key=bias_sort)

    for (ll, gmap, blen, _) in bias_map:
        if not calcx4[ll]:
            group = gmap[argmin(group_bias_max[t] for t in gmap)]
        else:
            group = gmap[argmin((group_bias_max[t] + 3) & ~3 for t in gmap)]
            group_bias_max[group] = (group_bias_max[group] + 3) & ~3  # Round up for x4 mode

        if group_bias_max[group] + blen > tc.dev.BIAS_SIZE:
            eprint(f'{layer_pfx(ll)}bias memory capacity exhausted - available groups: '
                   f'{gmap}, used so far: {group_bias_max}, needed: {blen} bytes, '
                   f'best available: group {group} with '
                   f'{tc.dev.BIAS_SIZE - group_bias_max[group]} bytes available.')
        bias_group[ll] = group
        for i in range(tc.dev.P_NUMGROUPS):
            bias_offs[ll][i] = group_bias_max[group]
        group_bias_max[group] += blen

    for ll in range(start_layer, layers):
        if bias[ll] is None or group_map[ll] is None or conv_groups[ll] > 1 \
           or not np.any(bias[ll] != 0) or state.fast_fifo_quad and ll == start_layer:
            continue

        # Round up the divided length of bias values
        target_offs = ffs(output_processor_map[ll]) % tc.dev.P_SHARED * out_expand[ll]

        if streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            wprint(f'{layer_pfx(ll)}Combining streaming and a bias. '
                   'THIS COMBINATION MIGHT NOT FUNCTION CORRECTLY!!!')

        group = bias_group[ll]

        # Each layer has output_channel number of bias values
        i = 0
        if ll == 0 and streaming[ll] and not tc.dev.SUPPORT_STREAM_BIAS:
            # Work around a problem on AI85
            # Store for later
            bias_values[group][bias_offs[ll][group]] = 0
            target_offs += 1
        while i < output_chan[ll]:
            # Store for later
            bias_values[group][bias_offs[ll][group] + target_offs] = bias[ll][i] & 0xff
            i += 1
            target_offs += 1

    # Replace placeholders with zero
    for group in range(tc.dev.P_NUMGROUPS):
        for i in range(group_bias_max[group]):
            if bias_values[group][i] == _INVALID_VALUE:
                bias_values[group][i] = 0

    if embedded_code:
        if max(group_bias_max) > 0:
            # At least one bias value exists, output #defines
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
                addr = state.apb_base + tc.dev.C_GROUP_OFFS*group + tc.dev.C_BRAM_BASE
                apb.output(f'  memcpy_8to32((uint32_t *) 0x{addr:08x}, bias_{group}, '
                           f'sizeof(uint8_t) * {group_bias_max[group]});\n', embedded_code)
        else:
            apb.function_header(function='load_bias')
            apb.output('  // Not used in this network', embedded_code)
    else:
        for group in range(tc.dev.P_NUMGROUPS):
            if group_bias_max[group] == 0:
                continue  # Nothing in this group
            for i in range(group_bias_max[group]):
                apb.write_bias(group, i, bias_values[group][i])

    apb.function_footer()  # load_bias()

    return bias_offs, bias_group, group_bias_max
