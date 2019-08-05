###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Load Tornado CNN data memory
"""
import sys
import numpy as np
import tornadocnn as tc
from utils import s2u, popcount


def load(embedded_code, apb, chw, processor_map, input_offset, input_size,
         in_expand, in_expand_thresh,
         data, padding, split=1, debug=False):
    """
    Create C code to load data input to offset `input_offset` in CHW format (if `chw` is `True`)
    or HWC format for the `processor_map`. Data `data` is organized in `input_size` channels and
    and dimensions. Channel expansion is configured in
    `in_expand` and `in_expand_thresh`.
    The code performs optional `padding`, can `split` the input into more than one chunk
    and has optional `debug` output.
    The code is target for simulation (`embedded_code` == `False`) or embedded hardware (`True`).
    Output is written to the `apb` object.
    """
    input_list = []
    chan = input_size[0]
    out_map = apb.get_mem()

    if not embedded_code:
        apb.output('\n\n  ')
    apb.output(f'// {chan}-channel {input_size[1]}x{input_size[2]} data input:\n')
    c = 0
    data_offs = None
    step = 1 if chw else 4
    for ch in range(0, tc.dev.MAX_CHANNELS, step):
        instance_map = (processor_map >> (ch % tc.dev.MAX_PROC)) % 2**step
        if not instance_map:
            # Channel or block of four channels not used for input
            continue
        num_ch = popcount(instance_map)

        # Load channel into shared memory
        group = (ch % tc.dev.MAX_PROC) // tc.dev.P_NUMPRO
        expand = c // in_expand_thresh  # Channels 64+ handled by processors 0+
        instance = (ch % tc.dev.P_NUMPRO) // tc.dev.P_SHARED
        new_data_offs = tc.dev.C_SRAM_BASE + tc.dev.C_GROUP_OFFS*group \
            + tc.dev.INSTANCE_SIZE*16*instance + expand*4

        if expand == 0:
            new_data_offs += input_offset
        if new_data_offs == data_offs:
            print('Layer 0 processor map is misconfigured for data input. '
                  f'There is data overlap between processors {ch-1} and {ch}')
            sys.exit(1)
        data_offs = new_data_offs

        if debug:
            print(f'G{group} L0 data_offs:      {data_offs:08x}')

        if input_size[2] == 1:
            data_len = 3 * ((input_size[1] + 2) // 3)
        else:
            data_len = input_size[1] * input_size[2]

        if chw:
            assert split > 0

            # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
            if embedded_code and split == 1:
                # Create optimized code when we're not splitting the input
                apb.output(f'// CHW (big data): {input_size[1]}x{input_size[2]}, channel {c}\n')
                offs = 0
                code_buffer = np.zeros(data_len // 4, dtype=np.int64)
                addr = data_offs

                val = 0
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        shift = (row * input_size[2] + col) % 4
                        val |= (s2u(data[c][row][col]) & 0xff) << (shift * 8)
                        if shift == 3:
                            apb.check_overwrite(data_offs & ~3)
                            out_map[(data_offs & ~3) >> 2] = (c, row, col, val)
                            code_buffer[offs] = val
                            offs += in_expand
                            val = 0
                        data_offs += 1
                        if data_offs & ~3 == 0:
                            data_offs += 4 * (in_expand - 1)

                if shift != 3:
                    apb.check_overwrite(data_offs & ~3)
                    out_map[(data_offs & ~3) >> 2] = (c, row, col, val)
                    code_buffer[offs] = val
                    offs += in_expand

                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8, weights=False)
                apb.output(f'static const uint32_t input_{ch}[] = INPUT_{ch};\n\n')
                input_list.append((addr, ch, offs))

                apb.data_offs = data_offs  # For mixed HWC/CHW operation
            else:
                if embedded_code:
                    apb.output('void load_input(void)\n{\n')

                apb.output(f'  // CHW (big data): {input_size[1]}x{input_size[2]}, channel {c}\n')

                chunk = input_size[1] // split
                # (Note: We do not need to flush here, since that is done at the
                # end of each channel's output below)
                if split > 1:
                    # Add top pad
                    for _ in range(padding[0]):
                        for _ in range(input_size[2]):
                            apb.write_byte(data_offs, 0)
                            data_offs += 1
                            if data_offs & ~3 == 0:
                                data_offs += 4 * (in_expand - 1)
                row = 0
                for s in range(split):
                    if split > 1 and s + 1 < split:
                        overlap = padding[0]
                    else:
                        overlap = 0
                    while row < (s + 1) * chunk + overlap:
                        for col in range(input_size[2]):
                            apb.write_byte(data_offs, s2u(data[c][row][col]))
                            data_offs += 1
                            if data_offs & ~3 == 0:
                                data_offs += 4 * (in_expand - 1)
                        row += 1
                    row -= 2*overlap  # Rewind
                    # Switch to next memory instance
                    if split > 1 and s + 1 < split:
                        new_data_offs = ((data_offs + tc.dev.INSTANCE_SIZE - 1) //
                                         tc.dev.INSTANCE_SIZE) * tc.dev.INSTANCE_SIZE
                        if new_data_offs != data_offs:
                            apb.write_byte_flush(0)
                        data_offs = new_data_offs
                if split > 1:
                    # Add bottom pad
                    for _ in range(padding[0]):
                        for _ in range(input_size[2]):
                            apb.write_byte(data_offs, 0)
                            data_offs += 1
                            if data_offs & ~3 == 0:
                                data_offs += 4 * (in_expand - 1)
            c += 1
        else:
            # HWC ("Little Data") - (Up to) four channels packed into a word
            # (0BGR0BGR0BGR0BGR0BGR....)
            if not embedded_code:
                apb.output('  ')
            apb.output(f'// HWC (little data): {input_size[1]}x{input_size[2]}, '
                       f'channels {c} to {c+num_ch-1}\n')

            if embedded_code:
                offs = 0
                code_buffer = np.zeros(data_len, dtype=np.int64)
                addr = data_offs

            for row in range(input_size[1]):
                for col in range(input_size[2]):
                    # Always write multiple of four bytes even for last input
                    # Handle gaps and fill with 0
                    val = 0
                    this_c = c
                    for i in range(4):
                        if instance_map & 2**i:
                            if this_c < len(data):
                                val |= (s2u(data[this_c][row][col]) & 0xff) << (i * 8)
                            this_c += 1

                    apb.check_overwrite(data_offs)
                    out_map[data_offs >> 2] = (this_c, row, col, val)
                    if not embedded_code:
                        apb.write(data_offs, val)
                    else:
                        code_buffer[offs] = val
                        offs += in_expand
                    apb.data_offs = data_offs  # For mixed HWC/CHW operation
                    data_offs += 4 * in_expand

            if embedded_code:
                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8, weights=False)
                apb.output(f'static const uint32_t input_{ch}[] = INPUT_{ch};\n\n')
                input_list.append((addr, ch, offs))

            c += num_ch

        apb.write_byte_flush(0)
        if c >= chan:
            # Consumed all available channels
            break

    if embedded_code:
        if input_list:
            apb.output('void load_input(void)\n{\n')
            for _, (addr, ch, offs) in enumerate(input_list):
                apb.output(f'  memcpy((uint32_t *) 0x{apb.apb_base + addr:08x}, input_{ch}, '
                           f'sizeof(uint32_t) * {offs});\n')
        apb.output('}\n\n')

    if not embedded_code:
        apb.output(f'  // End of data input\n\n')
