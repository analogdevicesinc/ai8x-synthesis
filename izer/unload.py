###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Unload AI8X HWC memory into standard representation.
"""
import os
from typing import List, Optional, TextIO

import numpy as np

from . import datamem, state, toplevel
from . import tornadocnn as tc
from .eprint import eprint, nprint, wprint
from .names import layer_pfx, layer_str
from .utils import ffs, popcount


def unload(
        *,
        memfile: TextIO,
        output_layer: List[bool],
        processor_map: List[int],
        input_shape: List[List[int]],
        out_offset: List[int],
        out_expand: List[int],
        out_expand_thresh: List[int],
        output_width: List[int],
        write_gap: List[int],
):
    """
    Unload HWC memory from hardware, writing C code to the `memfile` handle.
    The generated C code is specific to the network configuration passed in in `processor_map`,
    and `input_shape`. Additionally, the generated addresses are offset by `apb_base` and
    `out_offset`. The C code function takes a pointer to a memory array, and the depth of
    the array does not matter (flattened or not flattened) as long as the size is correct.
    When `mlator` is set, use the hardware mechanism to rearrange 4-channel data into single
    channels.
    """
    assert tc.dev is not None

    def mlator_write_one(
            prefix: str = '',
            comment: str = '',
            out_size: int = 8,
    ) -> None:
        """
        Print a single mlator unload line
        """
        return f'{prefix}  out_buf{"32" if out_size != 32 else ""}' \
               f'[offs++] = *mlat;{comment}\n'

    # Cache for faster access
    apb_base = state.apb_base
    mlator = state.mlator
    mlator_chunk = state.mlator_chunk if state.embedded_code else 1
    narrow_chunk = state.narrow_chunk if state.embedded_code else 0
    wide_chunk = state.wide_chunk if state.embedded_code else 0
    unload_custom = state.unload_custom
    mlator_warning = state.mlator_warning

    assert not state.block_mode or not mlator

    mlator_layers = []

    # If 'unload' is specified in the YAML file, create synthetic versions of
    # output_layer[], output_width[], input_shape[], processor_map[], out_expand[],
    # out_expand_thresh[], out_offset[], write_gap[].
    if unload_custom is not None:
        layers = len(unload_custom)
        # Create synthetic variables
        output_layer = [False] * layers
        processor_map = [None] * layers
        input_shape = [None] * layers
        out_offset = [None] * layers
        output_width = [None] * layers
        write_gap = [None] * layers
        out_expand = [None] * layers
        out_expand_thresh = [None] * layers
        # Fill them
        for ll, e in enumerate(unload_custom):
            output_layer[ll] = True
            processor_map[ll] = e['proc']
            input_shape[ll] = e['dim']
            out_offset[ll] = e['offset']
            output_width[ll] = e['width']
            write_gap[ll] = e['write_gap']
            output_chan = input_shape[ll][0]
            out_expand[ll] = (output_chan + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
            out_expand_thresh[ll] = (output_chan + out_expand[ll]-1) // out_expand[ll]
            if output_chan > tc.dev.MAX_PROC:
                out_expand_thresh[ll] = \
                    min((out_expand_thresh[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                        tc.dev.MAX_PROC)

    o_widths = set()
    for ll, e in enumerate(output_layer):
        if not e:
            continue

        lname = layer_pfx(ll) if unload_custom is None else f"Unload sequence #{ll}: "
        o_widths.add(output_width[ll])

        if output_width[ll] != 8:
            if mlator and mlator_warning:
                wprint(f'{lname}Ignoring --mlator for 32-bit output.')
        elif input_shape[ll][0] > 1 and input_shape[ll][1] * input_shape[ll][2] % 4 != 0:
            if mlator and mlator_warning:
                wprint(f'{lname}Ignoring --mlator for '
                       f'{input_shape[ll][1]}x{input_shape[ll][2]} frame size '
                       f'({input_shape[ll][1] * input_shape[ll][2]} bytes) that is '
                       f'not divisible by 4.')
        elif input_shape[ll][1] * input_shape[ll][2] < 4:
            if mlator and mlator_warning:
                wprint(f'{lname}--mlator should only be used with 4 or more 8-bit outputs '
                       ' per channel; ignoring.')
        elif mlator:
            mlator_layers.append(ll)
        elif state.embedded_code and mlator_warning:
            nprint(f'{lname}Use --mlator to optimize cnn_unload() for 8-bit output values.')

    # If ANY output is 8-bit, use the 8-bit unload since there is a chance of non-word aligned
    # writes.
    o_width = 32 if 8 not in o_widths else 8
    toplevel.function_header(memfile, function='unload',
                             arguments=f'uint32_t *out_buf{"32" if o_width != 32 else ""}')
    out_text = ''
    need_i = False
    need_offs = False
    read_addr = None
    write_addr = None
    written = 0
    out_addr = 0
    out_size = 1
    have_non_mlator = False
    first_output = True
    prev_out_size = 4

    for ll, e in enumerate(output_layer):
        if not e:
            continue

        lname = f"layer {layer_str(ll)}" if unload_custom is None else f"unload sequence #{ll}"
        out_text += f'\n  // Custom unload for this network, {lname}: ' \
                    f'{output_width[ll]}-bit data, shape: {input_shape[ll]}\n'
        if o_width != 32 and input_shape[ll][1] * input_shape[ll][2] != 1:
            need_offs = True

        out_size = output_width[ll] // 8

        coffs_start = ffs(processor_map[ll]) & ~(tc.dev.P_SHARED-1)
        coffs = coffs_start
        poffs = coffs_start
        next_layer_map_init = processor_map[ll] >> coffs
        next_layer_map = next_layer_map_init

        # Output expansion for channels and/or wide output
        width = out_expand[ll] * out_size

        mlat_addr = None
        emit_list = []
        c = 0
        while c < input_shape[ll][0]:
            if c % out_expand_thresh[ll] == 0:
                poffs = coffs_start
                next_layer_map = next_layer_map_init

            expand = c // out_expand_thresh[ll]  # Channels 64+ handled by processors 0+
            proc = poffs & ~(tc.dev.P_SHARED-1)

            if ll not in mlator_layers:
                have_non_mlator = True
                for doffs in range(input_shape[ll][1] * input_shape[ll][2]):
                    row, col = divmod(doffs, input_shape[ll][2])
                    this_map = next_layer_map
                    this_c = c

                    # Source
                    offs = out_offset[ll] + \
                        (((proc % tc.dev.P_NUMPRO) * tc.dev.INSTANCE_SIZE |
                            (proc // tc.dev.P_NUMPRO) * tc.dev.C_GROUP_OFFS // 4) +
                            doffs * width + expand * out_size) * (write_gap[ll] + 1) * 4

                    for shift in range(4):
                        if this_map & 1:
                            if o_width == 32:  # out_size == 4 (implied)
                                emit_list.append(offs)
                                offs += 4
                            elif out_size == 4:  # o_width == 8
                                emit_list.append((offs + 4 * shift, 0))
                                emit_list.append((offs + 4 * shift, 1))
                                emit_list.append((offs + 4 * shift, 2))
                                emit_list.append((offs + 4 * shift, 3))
                            else:  # out_size == 1, o_width == 8
                                emit_list.append((offs, shift))
                            this_c += 1
                        this_map >>= 1
            else:  # mlator
                def mlator_loop(
                        num: int = 1,
                ) -> None:
                    """
                    Print multiple mlator unload lines using a partially unrolled loop
                    """
                    if mlator_chunk == 1:
                        return ''

                    result = ''
                    # Gather several statements in a partially unrolled loop.
                    # The for() statement is only useful when the for loop runs at least twice.
                    if num >= 2 * mlator_chunk:
                        result += f'  for (i = 0; i < {num // mlator_chunk}; i++) {{\n'
                        for _ in range(mlator_chunk):
                            result += mlator_write_one('  ', '', out_size)
                        result += '  }\n'
                        num = num % mlator_chunk

                    # Emit single lines for all remaining statements
                    while num > 0:
                        result += mlator_write_one('', '', out_size)
                        num -= 1
                    return result

                assert out_size == 1
                this_map = next_layer_map
                mlat = apb_base + tc.ctl_addr(proc // tc.dev.P_NUMPRO, tc.dev.REG_MLAT)
                ctrl = apb_base + tc.ctl_addr(proc // tc.dev.P_NUMPRO, tc.dev.REG_CTL)
                if mlat_addr != mlat:
                    mlat_addr = mlat
                    out_text += f'  ctrl = (volatile uint32_t *) 0x{ctrl:08x};\n' \
                                f'  mlat = (volatile uint32_t *) 0x{mlat:08x};\n'

                this_c = c
                loop_count = 0
                for shift in range(4):
                    if this_map & 1:
                        out_text += f'  // Channel {this_c}\n'

                        for doffs in range(0, input_shape[ll][1] * input_shape[ll][2], 4):
                            if input_shape[ll][1] == 1 or input_shape[ll][2] == 1:
                                row, col = 0, doffs
                            else:
                                row, col = divmod(doffs, input_shape[ll][2])

                            # Get four bytes from memory
                            source = out_offset[ll] + \
                                (((proc % tc.dev.P_NUMPRO) * tc.dev.INSTANCE_SIZE |
                                  (proc // tc.dev.P_NUMPRO) * tc.dev.C_GROUP_OFFS // 4) +
                                 (doffs >> 2) * width + expand * out_size) \
                                * (write_gap[ll] + 1) * 4
                            target = this_c * input_shape[ll][1] * input_shape[ll][2] \
                                + row * input_shape[ll][1] + col + written
                            assert target & 3 == 0

                            if target != write_addr:
                                out_text += f'  offs = 0x{target >> 2:04x};\n'
                            if source != read_addr:
                                if loop_count > 0:
                                    out_text += mlator_loop(loop_count)
                                    loop_count = 0
                                if doffs != 0:
                                    out_text += \
                                        f'  *ctrl = 0x{tc.dev.READY_SEL << 1 | 1 << 3:08x}; ' \
                                        '// Disable mlator\n'
                                # Set wptr to start address
                                val = apb_base + tc.lreg_addr(proc // tc.dev.P_NUMPRO,
                                                              tc.dev.LREG_WPTR_BASE)
                                out_text += f'  *((volatile uint32_t *) 0x{val:08x}) = ' \
                                            f'0x{doffs:08x}; // Set SRAM address\n'
                                # Set wptr_inc to set increment value (default: 1)
                                val = apb_base + tc.lreg_addr(proc // tc.dev.P_NUMPRO,
                                                              tc.dev.LREG_LCTL2)
                                out_text += f'  *((volatile uint32_t *) 0x{val:08x}) = ' \
                                            f'0x{expand:08x}; // Set pointer increment\n'
                                # Set mlatorld enable bit to load write ptr; select byte 0..3
                                val = tc.dev.READY_SEL << 1 | 1 << 16 | shift << 17 | 1 << 3
                                out_text += f'  *ctrl = 0x{val:08x}; ' \
                                            f'// Enable mlator, byte {shift}\n'
                                # out_text += '  val = *mlat; // Prime\n'
                                out_text += '  asm volatile ("" : "=m" (*mlat) : "r" (*mlat));' \
                                            ' // Prime\n'

                            # FIXME: Do not write more than
                            # `num_bytes = min(4, input_shape[2] - col)`
                            if mlator_chunk == 1:
                                out_text += mlator_write_one('',
                                                             f' // {this_c},{row},{col}-{col+3}',
                                                             out_size)
                            loop_count += 1
                            read_addr = source + 4
                            write_addr = target + 4

                        if loop_count > 0:
                            out_text += mlator_loop(loop_count)
                            loop_count = 0
                        # Disable mlator
                        out_text += f'  *ctrl = 0x{tc.dev.READY_SEL << 1 | 1 << 3:08x}; ' \
                                    '// Disable mlator\n'
                    this_c += 1

                    this_map >>= 1

            coffs += 4
            poffs += 4
            c += popcount(next_layer_map & 0x0f)
            next_layer_map >>= 4

        if len(emit_list) > 0:
            if o_width == 32:
                if prev_out_size != 4:
                    # TODO: Resync output pointer
                    pass
                idx = 0
                chunk = max(1, wide_chunk)
                while idx < len(emit_list):
                    # Collect runs of same-delta sources
                    run = 0
                    if idx + 1 < len(emit_list):
                        delta_r = emit_list[idx + 1] - emit_list[idx]
                        assert delta_r % 4 == 0
                    else:
                        delta_r = 4
                    while (idx + run + 1 < len(emit_list)
                           and emit_list[run + 1] - emit_list[run] == delta_r):
                        run += 1

                    # Output as a loop
                    if out_addr == 0 or out_addr != apb_base + tc.dev.C_SRAM_BASE + emit_list[idx]:
                        out_text += '  addr = (volatile uint32_t *) ' \
                                    f'0x{apb_base + tc.dev.C_SRAM_BASE + emit_list[idx]:08x};\n'
                        out_addr = apb_base + tc.dev.C_SRAM_BASE + emit_list[idx]

                    remaining = run + 1
                    while remaining > 0:
                        loop_runs = max(
                            1,
                            remaining // chunk if wide_chunk > 0 else 1,
                        )
                        if loop_runs > 1:
                            need_i = True
                            out_text += f'  for (i = 0; i < {loop_runs}; i++) {{\n'
                            prefix = '  '
                        else:
                            prefix = ''
                        for _ in range(min(remaining, chunk)):
                            if delta_r == 4:
                                out_text += f'{prefix}  *out_buf++ = *addr++;\n'
                            else:
                                out_text += f'{prefix}  *out_buf++ = *addr;\n' \
                                            f'{prefix}  addr {"+" if delta_r >= 0 else "-"}= ' \
                                            f'0x{abs(delta_r) // 4:04x};\n'
                        if loop_runs > 1:
                            out_text += '  }\n'
                        remaining -= loop_runs * chunk
                        out_addr += loop_runs * chunk
                    idx += run + 1
            else:  # o_width == 8
                idx = 0
                xy_dim = input_shape[ll][1] * input_shape[ll][2]
                short_write = xy_dim == 1
                chunk = max(1, narrow_chunk)
                if not short_write:
                    out_text += '  offs = 0x0000;\n'
                if not first_output:
                    out_text += f'  out_buf = ((uint8_t *) out_buf32) + 0x{written:04x};\n'
                while idx < len(emit_list):
                    # Find how many have the same r/w addresses with different shift,
                    # then how many the same deltas between rs and ws with the same set of shifts.
                    shift_list = []
                    shift_count = 0
                    read_addr = emit_list[idx][0]
                    while (idx + shift_count < len(emit_list)
                           and emit_list[idx + shift_count][0] == read_addr):
                        shift_list.append(emit_list[idx + shift_count][1])
                        shift_count += 1
                    run = 0
                    if idx + shift_count < len(emit_list):
                        delta_r = emit_list[idx + shift_count][0] - emit_list[idx][0]
                        assert delta_r % 4 == 0
                    else:
                        # delta_w = 1
                        delta_r = 4
                    while (idx + shift_count * (run + 1) < len(emit_list)
                           and emit_list[idx + shift_count * (run + 1)][0]
                           - emit_list[idx + shift_count * run][0] == delta_r):
                        run += 1

                    # Output as a loop
                    if out_addr == 0 or out_addr != apb_base + tc.dev.C_SRAM_BASE + read_addr:
                        out_text += '  addr = (volatile uint32_t *) ' \
                                    f'0x{apb_base + tc.dev.C_SRAM_BASE + read_addr:08x};\n'
                        out_addr = apb_base + tc.dev.C_SRAM_BASE + read_addr

                    remaining = run + 1
                    while remaining > 0:
                        loop_runs = max(
                            1,
                            remaining // chunk if narrow_chunk > 0 else 1,
                        )
                        if loop_runs > 1:
                            need_i = True
                            out_text += f'  for (i = 0; i < {loop_runs}; i++) {{\n'
                            prefix = '  '
                        else:
                            prefix = ''
                        for _ in range(min(remaining, chunk)):
                            if delta_r == 4:
                                out_text += f'{prefix}  val = *addr++;\n'
                            else:
                                out_text += f'{prefix}  val = *addr;\n' \
                                            f'{prefix}  addr {"+" if delta_r >= 0 else "-"}= ' \
                                            f'0x{abs(delta_r) // 4:04x};\n'
                            for shift in shift_list:
                                if not short_write:
                                    out_text += f'{prefix}  out_buf[offs'
                                    if shift > 0:
                                        out_text += f'+0x{xy_dim * shift:02x}'
                                    out_text += '] = '
                                else:
                                    out_text += f'{prefix}  *out_buf++ = '
                                if shift == 0:
                                    out_text += 'val'
                                else:
                                    out_text += f'(val >> {shift * 8})'
                                out_text += ' & 0xff;\n'

                            if not short_write:
                                out_text += f'{prefix}  offs++;\n'
                        if loop_runs > 1:
                            out_text += '  }\n'
                        remaining -= loop_runs * chunk
                        out_addr += 4 * loop_runs * chunk

                    idx += (run + 1) * shift_count
                    if not short_write and idx < len(emit_list) and shift_count > 1:
                        out_text += f'  offs += 0x{xy_dim * (shift_count - 1):04x};\n'

        # Always a byte counter
        written += input_shape[ll][0] * input_shape[ll][1] * input_shape[ll][2] \
            * output_width[ll] // 8

        first_output = False
        prev_out_size = out_size

    if o_width != 32 and have_non_mlator:
        memfile.write(f'  uint{o_width}_t *out_buf = (uint{o_width}_t *) out_buf32;\n')
        memfile.write('  uint32_t val;\n')
    if o_width == 32 or have_non_mlator:
        memfile.write('  volatile uint32_t *addr;\n')
    if mlator_layers:
        memfile.write('  volatile uint32_t *mlat, *ctrl;\n')
    if need_i or mlator_layers and mlator_chunk > 1:
        memfile.write('  int i;\n')
    if need_offs:
        memfile.write('  uint32_t offs;\n')
    if out_text != '':
        memfile.write(out_text)
    toplevel.function_footer(memfile)  # unload()


def verify(
        verify_fn,
        ll,
        in_map,
        out_map,
        out_buf,
        processor_map,
        input_shape,
        out_offset,
        out_expand,
        out_expand_thresh,
        output_width: int = 8,
        overwrite_ok: bool = False,
        mlator: bool = False,
        body: Optional[List] = None,
        write_gap: int = 0,
        unload_layer: bool = False,
        embedded: bool = False,
        test_name: str = '',
        streaming: bool = False,
):
    """
    Verify HWC memory from AI8X, writing C or mem code using the `verify_fn` function.
    The generated code is specific to the network configuration passed in in `processor_map`,
    and `input_shape`. Additionally, the generated addresses are offset by
    `out_offset`. The function takes a pointer to a memory array, and the depth of
    the array does not matter (flattened or not flattened) as long as the size is correct.
    `in_map` and `out_map` are used to optionally prevent overwriting data
    (controlled by `overwrite_ok` and `no_error_stop`).
    When `mlator` is set, use the hardware mechanism to rearrange 4-channel data into single
    channels.
    """
    assert tc.dev is not None

    # Cache for faster access
    apb_base = state.apb_base
    max_count = state.max_count
    no_error_stop = state.no_error_stop

    if state.result_numpy is not None:
        # Also save as a NumPy pickle
        np.save(os.path.join(state.base_directory, test_name, state.result_numpy),
                out_buf, allow_pickle=False, fix_imports=False)

    if embedded:
        # check_output() does not use mlator for embedded code (it is used for cnn_unload(),
        # and for RTL sims)
        mlator = False
    if mlator and output_width != 8:
        wprint(f'{layer_pfx(ll)}Ignoring --mlator for 32-bit output.')
        mlator = False

    count = 0

    def check_overwrite(
            p,
            target_offs,
            in_map,
            out_map,
            c,
            row,
            col,
    ):
        if not overwrite_ok:
            # If using single layer, make sure we're not overwriting the input
            old_ll, old_c, old_row, old_col = datamem.unpack(in_map, target_offs)
            if old_ll is not None:
                old_layer = \
                    f'layer {layer_str(old_ll)}, CHW={old_c},{old_row},{old_col}' if old_ll >= 0 \
                    else 'the input loader'
                eprint(f'Processor {p}: '
                       f'Layer {layer_str(ll)} output for CHW={c},{row},{col} is overwriting '
                       f'input at offset 0x{target_offs:08x} that was created by '
                       f'{old_layer}.',
                       error=not no_error_stop)
            # Check we're not overflowing the data memory
            if out_map is not None:
                old_ll, old_c, old_row, old_col = datamem.unpack(out_map, target_offs)
                if old_ll is not None:
                    eprint(f'Processor {p}: '
                           f'Layer {layer_str(ll)} output for CHW={c},{row},{col} is overwriting '
                           f'offset 0x{target_offs:08x}. Previous write by '
                           f'layer {layer_str(old_ll)}, CHW={old_c},{old_row},{old_col}.',
                           error=not no_error_stop)

    # Start at the instance of the first active output processor/channel
    coffs_start = ffs(processor_map) & ~(tc.dev.P_SHARED-1)
    next_layer_map = processor_map >> coffs_start
    # Output expansion for channels and/or wide output
    out_size = output_width // 8
    width = out_expand * out_size

    if unload_layer and not embedded:
        body.append(f'  // Layer {layer_str(ll)}\n')

    for doffs in range(input_shape[1] * input_shape[2]):
        row, col = divmod(doffs, input_shape[2])
        this_map = next_layer_map
        coffs = coffs_start
        poffs = coffs_start
        c = 0
        while c < input_shape[0]:
            if c % out_expand_thresh == 0:
                poffs = coffs_start
                this_map = next_layer_map  # Wrap around for AI85 channel expansion

            this_c = c
            expand = c // out_expand_thresh  # Channels 64+ handled by processors 0+
            # Physical offset into instance and group
            proc = poffs & ~(tc.dev.P_SHARED-1)

            # Get four bytes or words either from output or zeros and construct HWC word
            no_data = True
            if out_size == 1:
                val = 0
                for _ in range(4):
                    val >>= 8
                    if this_map & 1:
                        no_data = False
                        if c < input_shape[0]:
                            val |= (out_buf[c][row][col] & 0xff) << 24
                        c += 1
                    this_map >>= 1
            else:
                val = [0] * 4
                for i in range(4):
                    if this_map & 1:
                        no_data = False
                        if c < input_shape[0]:
                            val[i] = out_buf[c][row][col] & 0xffffffff
                        c += 1
                    this_map >>= 1

            # Get the offset of the first output byte/word of 4
            offs = tc.dev.C_SRAM_BASE + out_offset + \
                (((proc % tc.dev.P_NUMPRO) * tc.dev.INSTANCE_SIZE |
                  (proc // tc.dev.P_NUMPRO) * tc.dev.C_GROUP_OFFS // 4) +
                 (doffs * width + expand * out_size) * (write_gap + 1)) * 4

            if not no_data:
                num_bytes = min(c - this_c, input_shape[0] - this_c)
                if out_size == 1:
                    if not streaming:
                        check_overwrite(
                            proc,
                            offs,
                            in_map,
                            out_map,
                            this_c,
                            row,
                            col,
                        )
                        if out_map is not None:
                            datamem.store(out_map, offs, (ll, this_c, row, col))
                    if not mlator:
                        verify_fn(
                            offs,
                            val,
                            rv=False,
                            comment=f' // {this_c}-{this_c+num_bytes-1},{row},{col}',
                            num_bytes=num_bytes,
                            first_proc=ffs(processor_map >> proc) % 4,
                            data=unload_layer,
                        )
                else:
                    for i in range(min(num_bytes, out_size)):
                        if not streaming:
                            check_overwrite(
                                proc,
                                offs,
                                in_map,
                                out_map,
                                this_c,
                                row,
                                col,
                            )
                            if out_map is not None:
                                datamem.store(out_map, offs, (ll, this_c, row, col))
                        if not mlator:
                            verify_fn(
                                offs,
                                val[i],
                                rv=False,
                                comment=f' // {this_c+i},{row},{col}',
                                data=unload_layer,
                            )
                        offs += out_size
                count += 1
                if count == max_count:
                    body.append('  // Truncated further checks...\n')

            coffs += 4
            poffs += 4

    if mlator:
        # This path is used for RTL sims to emit the verification code.
        # Overwrite checks have already happened above.
        assert out_size == 1
        c = 0
        poffs = coffs_start
        this_map = next_layer_map
        read_addr = None

        while c < input_shape[0]:
            if c % out_expand_thresh == 0:
                poffs = coffs_start  # Wrap around for AI85 channel expansion
                this_map = next_layer_map

            expand = c // out_expand_thresh  # Channels 64+ handled by processors 0+
            # Physical offset into instance and group
            proc = poffs & ~(tc.dev.P_SHARED-1)

            mlat = tc.ctl_addr(proc // tc.dev.P_NUMPRO, tc.dev.REG_MLAT)
            ctrl = tc.ctl_addr(proc // tc.dev.P_NUMPRO, tc.dev.REG_CTL)

            for shift in range(4):
                if this_map & 1:
                    for doffs in range(0, input_shape[1] * input_shape[2], 4):
                        row, col = divmod(doffs, input_shape[2])

                        # Get four bytes or words either from output or zeros and
                        # construct HWC word
                        val = 0
                        for i in range(4):
                            val >>= 8
                            if col+i < input_shape[2]:
                                val |= (out_buf[c][row][col+i] & 0xff) << 24

                        # Get the offset of the first output byte/word of 4
                        source = out_offset + \
                            (((proc % tc.dev.P_NUMPRO) * tc.dev.INSTANCE_SIZE |
                              (proc // tc.dev.P_NUMPRO) * tc.dev.C_GROUP_OFFS // 4) +
                             (doffs >> 2) * width) * 4

                        if source != read_addr:
                            if doffs != 0:
                                body.append(f'  *((volatile uint32_t *) '
                                            f'0x{apb_base + ctrl:08x}) = '
                                            f'0x{tc.dev.READY_SEL << 1 | 1 << 3:08x}; '
                                            '// Disable mlator\n')
                            # Set wptr to start address
                            w = apb_base + tc.lreg_addr(proc // tc.dev.P_NUMPRO,
                                                        tc.dev.LREG_WPTR_BASE)
                            body.append(f'  *((volatile uint32_t *) 0x{w:08x}) = '
                                        f'0x{source >> 2:08x}; // Set SRAM address\n')
                            # Set wptr_inc to set increment value (default: 1)
                            w = apb_base + tc.lreg_addr(proc // tc.dev.P_NUMPRO,
                                                        tc.dev.LREG_LCTL2)
                            body.append(f'  *((volatile uint32_t *) 0x{w:08x}) = '
                                        f'0x{expand:08x}; // Set pointer increment\n')
                            # Set mlatorld enable bit to load write ptr; select byte 0..3
                            w = tc.dev.READY_SEL << 1 | 1 << 16 | shift << 17 | 1 << 3
                            body.append(f'  *((volatile uint32_t *) 0x{apb_base + ctrl:08x}) ='
                                        f' 0x{w:08x}; '
                                        f'// Enable mlator, byte {shift}\n')
                            body.append('  asm volatile ("" : "=m" (*((volatile uint32_t *) '
                                        f'0x{apb_base + mlat:08x})) : "r" '
                                        f'(*((volatile uint32_t *) 0x{apb_base + mlat:08x})));'
                                        ' // Prime\n')

                        num_bytes = min(4, input_shape[2] - col)
                        first_proc = (out_offset >> 2) & 0x03
                        if first_proc != 0:
                            val <<= first_proc * 8
                        # No overwrite checks here since mlator happens on read of the memory,
                        # not on write TO the memory.
                        verify_fn(
                            tc.dev.C_SRAM_BASE + source,
                            val,
                            rv=False,
                            comment=f' // {c},{row},{col}-{col+num_bytes-1}',
                            num_bytes=num_bytes,
                            first_proc=first_proc,
                            data=unload_layer,
                        )

                        read_addr = source + 4
                    # Disable mlator
                    body.append(f'  *((volatile uint32_t *) '
                                f'0x{apb_base + ctrl:08x}) = '
                                f'0x{tc.dev.READY_SEL << 1 | 1 << 3:08x}; '
                                '// Disable mlator\n')

                this_map >>= 1
                c += 1

            poffs += 4
