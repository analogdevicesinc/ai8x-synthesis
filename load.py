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
import camera
import rv
import tornadocnn as tc
from utils import popcount, s2u


def load(
        embedded_code,
        apb,
        chw,
        processor_map,
        input_offset,
        input_size,
        in_expand,
        operands,
        in_expand_thresh,
        data,
        padding,
        split=1,
        fifo=False,
        slowdown=0,
        synthesize=None,
        riscv_flash=False,
        csv_file=None,
        debug=False,
):
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

    if csv_file is not None:
        return loadcsv(
            embedded_code,
            apb,
            input_size,
            operands,
            data,
            slowdown,
            csv_file,
            debug,
        )
    elif fifo:
        return loadfifo(
            embedded_code,
            apb,
            chw,
            processor_map,
            input_size,
            operands,
            data,
            slowdown,
            synthesize,
            riscv_flash,
            debug,
        )

    input_list = []
    chan = input_size[0]
    out_map = apb.get_mem()

    if not embedded_code:
        apb.output('\n\n  ')
    apb.output(f'// {chan}-channel {input_size[1]}x{input_size[2]} data input:\n')
    c = 0
    data_offs = None
    step = 1 if chw else 4
    assert operands == data.shape[0] // input_size[0]

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
            + tc.dev.INSTANCE_SIZE*16*instance + expand*4 * operands

        if expand == 0:
            new_data_offs += input_offset
        if new_data_offs == data_offs:
            print('Layer 0 processor map is misconfigured for data input. '
                  f'There is data overlap between processors {ch-1} and {ch}')
            sys.exit(1)
        data_offs = new_data_offs

        if debug:
            print(f'G{group} L0 data_offs:      {data_offs:08x}')

        if chw:
            assert split > 0
            assert operands == 1  # We don't support multiple operands here (yet)
            # FIXME: Support multiple operands for CHW data

            # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
            if embedded_code and split == 1:
                # Create optimized code when we're not splitting the input
                apb.output(f'// CHW (big data): {input_size[1]}x{input_size[2]}, channel {c}\n')
                offs = 0
                code_buffer = np.zeros(input_size[1] * input_size[2] // 4, dtype=np.int64)
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
                            offs += 1
                            val = 0
                        data_offs += 1
                        if data_offs & ~3 == 0:
                            data_offs += 4 * (in_expand - 1)

                if shift != 3:
                    apb.check_overwrite(data_offs & ~3)
                    out_map[(data_offs & ~3) >> 2] = (c, row, col, val)
                    code_buffer[offs] = val
                    offs += 1

                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8, weights=False)
                if riscv_flash:
                    apb.output(rv.RISCV_FLASH)
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
                code_buffer = np.zeros(operands * input_size[1] * input_size[2], dtype=np.int64)
                addr = data_offs

            for row in range(input_size[1]):
                for col in range(input_size[2]):
                    for op in range(operands):
                        # Always write multiple of four bytes even for last input
                        # Handle gaps and fill with 0
                        val = 0
                        this_c = c
                        for i in range(4):
                            if instance_map & 2**i:
                                if this_c < len(data) // operands:
                                    val |= (s2u(data[this_c + op*input_size[0]][row][col])
                                            & 0xff) << (i * 8)
                                this_c += 1

                        apb.check_overwrite(data_offs)
                        out_map[data_offs >> 2] = (this_c, row, col, val)
                        if not embedded_code:
                            apb.write(data_offs, val)
                        else:
                            code_buffer[offs] = val
                            offs += 1
                        apb.data_offs = data_offs  # For mixed HWC/CHW operation
                        data_offs += 4
                    data_offs += 4 * (in_expand - 1) * operands

            if embedded_code:
                apb.output_define(code_buffer, f'INPUT_{ch}', '0x%08x', 8, weights=False)
                if riscv_flash:
                    apb.output(rv.RISCV_FLASH)
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
                apb.output(f'  memcpy32((uint32_t *) 0x{apb.apb_base + addr:08x}, input_{ch}, '
                           f'{offs});\n')
        apb.output('}\n\n')
    else:
        apb.output(f'  // End of data input\n\n')


def loadfifo(
        embedded_code,
        apb,
        chw,
        processor_map,
        input_size,
        operands,
        data,
        slowdown=0,
        synthesize=None,
        riscv_flash=False,
        debug=False,  # pylint: disable=unused-argument
):
    """
    Create C code to load data into FIFO(s) in CHW format (if `chw` is `True`)
    or HWC format for the `processor_map`. Data `data` is organized in `input_size` channels and
    and dimensions. The code has optional `debug` output.
    The code is target for simulation (`embedded_code` == `False`) or embedded hardware (`True`).
    Output is written to the `apb` object.
    """
    assert operands == 1  # We don't support multiple operands here
    # FIXME: Support multiple operands

    if not embedded_code:
        apb.output('\n\n  ')

    if chw:
        # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
        apb.output('// Data input: CHW (big data): '
                   f'{input_size[0]}x{input_size[1]}x{input_size[2]}\n')

        if embedded_code:
            code_buffer = np.zeros((input_size[0], (input_size[1] * input_size[2] + 3) // 4),
                                   dtype=np.int64)

        for row_col in range(0, input_size[1] * input_size[2], 4):
            pmap = 0
            for c in range(input_size[0]):
                if pmap == 0:
                    pmap = processor_map
                    fifo = 0
                while pmap & 1 == 0:
                    pmap >>= 16
                    fifo += 1
                val = 0
                for b in range(4):
                    if row_col + b < input_size[1] * input_size[2]:
                        row, col = divmod(row_col + b, input_size[2])
                        val |= (s2u(data[c][row][col]) & 0xff) << b * 8
                if not embedded_code:
                    apb.write(0, val, '', fifo=fifo)
                    for _ in range(slowdown):
                        apb.output('  asm volatile("nop");\n')
                else:
                    code_buffer[fifo][row_col // 4] = val

                pmap >>= 16
                fifo += 1

        if embedded_code:
            fifos = input_size[0]
    else:
        # HWC ("Little Data") - (Up to) four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
        apb.output('// Data input: HWC (little data): '
                   f'{input_size[0]}x{input_size[1]}x{input_size[2]}\n')

        if embedded_code:
            code_buffer = np.zeros(((input_size[0] + 3) // 4, input_size[1] * input_size[2]),
                                   dtype=np.int64)

        for row in range(input_size[1]):
            for col in range(input_size[2]):
                pmap = 0
                for c in range(0, input_size[0], 4):
                    if pmap == 0:
                        pmap = processor_map
                        fifo = 0
                    while pmap & 0x0f == 0:
                        pmap >>= 16
                        fifo += 1
                    val = 0
                    for b in range(4):
                        if pmap & 1 != 0 and c + b < input_size[0]:
                            val |= (s2u(data[c + b][row][col]) & 0xff) << b * 8
                        pmap >>= 1
                    if not embedded_code:
                        apb.write(0, val, '', fifo=fifo)
                        for _ in range(slowdown):
                            apb.output('  asm volatile("nop");\n')
                    else:
                        code_buffer[fifo][row * col] = val
                    pmap >>= 12
                    fifo += 1

        if embedded_code:
            fifos = (input_size[0] + 3) // 4

    if embedded_code:
        for c in range(fifos):
            apb.output_define(code_buffer[c], f'INPUT_{c}', '0x%08x', 8, weights=False)
            if riscv_flash:
                apb.output(rv.RISCV_FLASH)
            apb.output(f'static const uint32_t input_{c}[] = INPUT_{c};\n')

        apb.output('\nvoid load_input(void)\n{\n')
        apb.output('  int i;\n')
        if synthesize is not None:
            apb.output(f'  uint32_t add = 0;\n')
        max_len = 0
        const_len = True
        for c in range(fifos):
            if c > 1 and max_len != len(code_buffer[c]):
                const_len = False
            max_len = max(max_len, len(code_buffer[c]))
            apb.output(f'  const uint32_t *in{c} = input_{c};\n')
        apb.output(f'\n  for (i = 0; i < {max_len}; i++) {{\n')
        for c in range(fifos):
            if fifos > 1 and not const_len:
                apb.output(f'    if (i < {len(code_buffer[c])})\n  ')
            if synthesize is None:
                apb.write(0, f'*in{c}++', fifo=c,
                          comment=f' // Write FIFO {c}', indent='    ')
            else:
                apb.write(0, f'*in{c}++ + add', fifo=c,
                          comment=f' // Write FIFO {c}', indent='    ')
        if synthesize is not None:
            apb.output('    if ((i % 8 == 0) && (i != 0)) {\n')
            apb.output(f'      add += 0x{synthesize:x};\n    ')
            for c in range(fifos):
                apb.output(f'  in{c} = input_{c};\n')
            apb.output('    }\n')
        apb.output('  }\n')
        apb.output('}\n\n')
    else:
        apb.output(f'  // End of data input\n\n')


def loadcsv(
        embedded_code,
        apb,
        input_size,
        operands,
        data,
        slowdown=0,  # pylint: disable=unused-argument
        csv_file=None,
        debug=False,  # pylint: disable=unused-argument
):
    """
    Create C code to load data into FIFO(s) from the camera interface.
    The code is target for simulation (`embedded_code` == `False`) or embedded hardware (`True`).
    Output is written to the `apb` object.
    Additionally, the code creates a CSV file with input data for simulation.
    """
    assert operands == 1  # We don't support multiple operands here
    # FIXME: Support multiple operands
    assert csv_file is not None

    if not embedded_code:
        apb.output('\n\n  ')

    # HWC ("Little Data") - (Up to) four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
    apb.output('// Data input: HWC (little data): '
               f'{input_size[0]}x{input_size[1]}x{input_size[2]}\n')

    fifos = (input_size[0] + 3) // 4

    if embedded_code:
        apb.output('\n#ifdef USE_FIFO\n')
        apb.output('#define FIFO_SZ 1024\n')
        apb.output('uint32_t fifo[FIFO_SZ];\n')
        apb.output('#endif\n\n')
        apb.output('void load_input(void)\n{\n')
        apb.output('#ifndef USE_FIFO\n')
        apb.output('  int i;\n')
        apb.output('  uint32_t d;\n')
        apb.output('#else\n')
        apb.output('  int i = 0;\n')
        apb.output('  register int head = 0;\n')
        apb.output('  register int tail = 0;\n\n')
        apb.output('#endif\n')
        apb.output('  // Tell tb to start sending pcif data\n')
        apb.output('  sim->trig = 0;\n\n')
        max_len = input_size[1] * input_size[2]

        apb.output('#ifndef USE_FIFO\n')
        apb.output(f'  for (i = 0; i < {max_len}; i++) {{\n')
        for c in range(fifos):
            apb.output('    while ((MXC_CAMERAIF0->int_fl & 0x80) == 0); '
                       '// Wait for camera FIFO not empty\n')
            apb.output('    d = MXC_CAMERAIF0->dma_data; // Read camera\n')
            apb.write(0, 'd', fifo=c, comment=f' // Write FIFO {c}', indent='    ')
        apb.output('  }\n')
        apb.output('#else\n')
        apb.output(f'  while (i < {max_len}) {{\n')
        apb.output('    if (((MXC_CAMERAIF0->int_fl & 0x80) != 0) && (head + 1 != tail) && '
                   '((head + 1 != FIFO_SZ) || (tail != 0))) {\n')
        apb.output('      // Camera FIFO not empty and software FIFO not full\n')
        apb.output('      fifo[head++] = MXC_CAMERAIF0->dma_data; // Read camera\n')
        apb.output('      if (head == FIFO_SZ)\n')
        apb.output('        head = 0;\n')
        apb.output('    }\n\n')
        apb.output('    if ((head != tail) && (((*((volatile uint32_t *) 0x400c0404) '
                   '& 2)) == 0)) {\n')
        apb.output('      // Software FIFO not empty, and room in CNN FIFO\n')
        apb.output('      *((volatile uint32_t *) 0x400c0410) = fifo[tail++];\n')
        apb.output('      if (tail == FIFO_SZ)\n')
        apb.output('        tail = 0;\n')
        apb.output('      i++;\n')
        apb.output('    }\n')
        apb.output('  }\n')
        apb.output('#endif\n')

        apb.output('}\n\n')

        with open(csv_file, mode='w') as f:
            camera.header(f)

            for row in range(input_size[1]):
                for col in range(input_size[2]):
                    for c in range(0, input_size[0]):
                        camera.pixel(f, s2u(data[c][row][col]) & 0xff)
                camera.finish_row(f)

            camera.finish_image(f)
