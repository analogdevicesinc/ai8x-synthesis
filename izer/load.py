###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Load Tornado CNN data memory
"""
from typing import List

import numpy as np

from . import camera, datamem, rv, state
from . import tornadocnn as tc
from .eprint import eprint
from .utils import popcount, s2u


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
        csv_file=None,
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
    # Cache for faster access
    fixed_input = state.fixed_input
    split = state.split
    synthesize = state.synthesize_input

    if fixed_input and not embedded_code:
        eprint('--fixed-input requires --embedded-code')

    if csv_file is not None:
        return loadcsv(
            embedded_code,
            apb,
            chw,
            input_size,
            operands,
            data,
            csv_file,
        )
    # else:
    if state.fifo:
        return loadfifo(
            embedded_code,
            apb,
            chw,
            processor_map,
            input_size,
            operands,
            data,
        )

    input_list = []
    chan = input_size[0]
    out_map = apb.get_mem()

    if not embedded_code:
        apb.output('\n\n  ')
    byte_size = input_size[0]*input_size[1]*input_size[2]
    apb.output(f'// {chan}-channel {input_size[1]}x{input_size[2]} data input '
               f'({byte_size} bytes')
    if input_size[0] > 1:
        apb.output(f' total / {input_size[1]*input_size[2]} bytes per channel):\n')
    else:
        apb.output(f' / {(byte_size + 3) // 4} 32-bit words):\n')

    c = 0
    data_offs = None
    step = 1 if chw else 4
    if operands != data.shape[0] // input_size[0]:
        eprint(f'Data input/weights mismatch: The sample data input has {data.shape[0]} channels '
               f'with {operands} operand(s), but there are weights for {input_size[0]} channels.')

    buffer_list = [[] for i in range(tc.dev.MAX_PROC)]

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
            eprint('Layer 0 processor map is misconfigured for data input. '
                   f'There is data overlap between processors {ch-1} and {ch}')
        data_offs = new_data_offs

        if state.debug:
            print(f'G{group} L0 data_offs:      {data_offs:08x}')

        if chw:
            assert split > 0
            assert operands == 1  # We don't support multiple operands here (yet)
            # FIXME: Support multiple operands for CHW data

            if embedded_code and in_expand > 1:
                # FIXME: This code does not handle multi-pass
                eprint('--compact-data does not currently support multi-pass CHW input')

            # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
            if embedded_code and split == 1:
                # Create optimized code when we're not splitting the input
                apb.output(f'// CHW {input_size[1]}x{input_size[2]}, channel {c}\n')
                offs = 0
                code_buffer = np.zeros((input_size[1] * input_size[2] + 3) // 4, dtype=np.int64)
                addr = data_offs

                val = 0
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        shift = (row * input_size[2] + col) % 4
                        val |= (s2u(data[c][row][col]) & 0xff) << (shift * 8)
                        if shift == 3:
                            datamem.store(out_map, data_offs & ~3, (-1, c, row, col),
                                          check_overwrite=True)
                            code_buffer[offs] = val
                            offs += 1
                            val = 0
                        data_offs += 1
                        if data_offs & ~3 == 0:
                            data_offs += 4 * (in_expand - 1)

                if shift != 3:
                    datamem.store(out_map, data_offs & ~3, (-1, c, row, col),
                                  check_overwrite=True)
                    code_buffer[offs] = val
                    offs += 1

                if not fixed_input:
                    b = code_buffer if synthesize is None else code_buffer[:state.synthesize_words]
                    apb.output_define(b, f'SAMPLE_INPUT_{ch}', '0x%08x', 8,
                                      weights=False)
                    apb.inc_writes(len(b))
                if state.riscv_flash:
                    apb.output(rv.RISCV_FLASH)
                if not fixed_input:
                    apb.output(f'static const uint32_t input_{ch}[] = SAMPLE_INPUT_{ch};\n\n')
                input_list.append((addr, ch, offs))

                apb.data_offs = data_offs  # For mixed HWC/CHW operation
            else:
                if embedded_code:
                    apb.function_header(dest='wrapper', prefix='', function='load_input',
                                        return_type='void')
                    apb.output('  // This function loads the sample data input -- '
                               'replace with actual data\n\n')

                apb.output(f'  // CHW {input_size[1]}x{input_size[2]}, channel {c}\n')

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
                apb.output(f'// HWC {input_size[1]}x{input_size[2]}, '
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

                        datamem.store(out_map, data_offs, (-1, this_c, row, col),
                                      check_overwrite=True)
                        if not embedded_code:
                            apb.write_data(data_offs, val)
                        else:
                            code_buffer[offs] = val
                            offs += 1
                        apb.data_offs = data_offs  # For mixed HWC/CHW operation
                        data_offs += 4
                    data_offs += 4 * (in_expand - 1) * operands

            if embedded_code:
                proc = ch % tc.dev.MAX_PROC

                # Save for merge
                buffer_list[proc].append((code_buffer, addr, c, c+num_ch-1))

                if expand == in_expand-1:
                    # Big buffer holds the multi-pass data
                    buf = np.zeros((expand + 1) * operands * input_size[1] * input_size[2],
                                   dtype=np.int64)

                    # Merge all buffers into big buffer
                    for i, e in enumerate(buffer_list[proc]):
                        apb.output(f'// HWC {input_size[1]}x{input_size[2]}, '
                                   f'channels {e[2]} to {e[3]}\n')
                        for j, val in enumerate(e[0]):
                            target = i * operands + (j // operands) * in_expand * operands \
                                + j % operands
                            buf[target] = val

                    if not fixed_input:
                        b = buf if synthesize is None else buf[:state.synthesize_words]
                        apb.output_define(b, f'SAMPLE_INPUT_{proc}', '0x%08x', 8, weights=False)
                        apb.inc_writes(len(b))
                    if state.riscv_flash:
                        apb.output(rv.RISCV_FLASH)
                    if not fixed_input:
                        apb.output(f'static const uint32_t input_{proc}[] = '
                                   f'SAMPLE_INPUT_{proc};\n\n')

                    # Append information using first address, processor number, and total length
                    input_list.append((buffer_list[proc][0][1], proc, offs * in_expand))

            c += num_ch

        apb.write_byte_flush(0)
        if c >= in_expand_thresh * in_expand:
            # Consumed all available channels
            break

    if embedded_code:
        if input_list:
            if fixed_input:
                apb.function_header(prefix='', function='memcpy32_const', return_type='void',
                                    arguments='uint32_t *dst, int n')
                apb.output('  while (n > 0) {\n'
                           '    *dst++ = 0x55555555;\n'
                           '    *dst++ = 0xaaaaaaaa;\n'
                           '    n -= 2;\n'
                           '  }\n', True)
                apb.function_footer(return_value='void')  # memcpy32_const()

            apb.function_header(dest='wrapper', prefix='', function='load_input',
                                return_type='void')
            apb.output('  // This function loads the sample data input -- '
                       'replace with actual data\n\n')
            for (addr, ch, offs) in input_list:
                if not fixed_input:
                    apb.output(f'  memcpy32((uint32_t *) 0x{state.apb_base + addr:08x}, '
                               f'input_{ch}, {offs});\n')
                else:
                    apb.output(f'  memcpy32_const((uint32_t *) 0x{state.apb_base + addr:08x}, '
                               f'{offs});\n')
        apb.function_footer(dest='wrapper', return_value='void')  # load_input()
    else:
        apb.output('  // End of data input\n\n')

    return None


def loadfifo(
        embedded_code,
        apb,
        chw,
        processor_map,
        input_size,
        operands,
        data,
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

    # Cache for faster access
    synthesize = state.synthesize_input

    if not embedded_code:
        apb.output('\n\n  ')

    byte_size = input_size[0]*input_size[1]*input_size[2]

    if chw:
        # CHW ("Big Data") - Separate channel sequences (BBBBB....GGGGG....RRRRR....)
        apb.output('// Data input: CHW '
                   f'{input_size[0]}x{input_size[1]}x{input_size[2]} '
                   f'({byte_size} bytes')
        if input_size[0] > 1:
            apb.output(f' total / {input_size[1]*input_size[2]} bytes per channel')
        apb.output('):\n')

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
                    apb.write(0, val, '', fifo=fifo, fifo_wait=state.fifo_wait)
                    for _ in range(state.slow_load):
                        apb.output('  asm volatile("nop");\n')
                else:
                    code_buffer[fifo][row_col // 4] = val

                pmap >>= 16
                fifo += 1

        if embedded_code:
            fifos = input_size[0]
    else:
        # HWC ("Little Data") - (Up to) four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
        apb.output('// Data input: HWC '
                   f'{input_size[0]}x{input_size[1]}x{input_size[2]} '
                   f'({byte_size} bytes')
        if input_size[0] > 1:
            apb.output(f' total / {input_size[1]*input_size[2]} bytes per channel')
        apb.output('):\n')

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
                        apb.write(0, val, '', fifo=fifo, fifo_wait=state.fifo_wait)
                        for _ in range(state.slow_load):
                            apb.output('  asm volatile("nop");\n')
                    else:
                        code_buffer[fifo][row * input_size[2] + col] = val
                    pmap >>= 12
                    fifo += 1

        if embedded_code:
            fifos = (input_size[0] + 3) // 4

    if embedded_code:
        for c in range(fifos):
            b = code_buffer[c] if synthesize is None else code_buffer[c][:state.synthesize_words]
            apb.output_define(b, f'SAMPLE_INPUT_{c}', '0x%08x', 8, weights=False)
            if state.riscv_flash:
                apb.output(rv.RISCV_FLASH)
            apb.output(f'static const uint32_t input_{c}[] = SAMPLE_INPUT_{c};\n')
            apb.inc_writes(len(b), fifo=c, fifo_wait=state.fifo_wait)

        apb.function_header(dest='wrapper', prefix='', function='load_input',
                            return_type='void')
        apb.output('  // This function loads the sample data input -- '
                   'replace with actual data\n\n'
                   '  int i;\n')
        if synthesize is not None:
            apb.output('  uint32_t add = 0;\n')
            mask = ''
            if input_size[0] == 1:
                mask = '0xff'
            elif input_size[0] == 2:
                mask = '0xffff'
            elif input_size[0] == 3:
                mask = '0xffffff'
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
                apb.write(0, f'*in{c}++', fifo=c, fifo_wait=state.fifo_wait,
                          comment=f' // Write FIFO {c}', indent='    ')
            else:
                s = f'*in{c}++ + add' if mask == '' else f'(*in{c}++ + add) & {mask}'
                apb.write(0, s, fifo=c, fifo_wait=state.fifo_wait,
                          comment=f' // Write FIFO {c}', indent='    ')
        if synthesize is not None:
            apb.output(f'    if (i % {state.synthesize_words} == '
                       f'{state.synthesize_words - 1}) {{\n')
            apb.output(f'      add += 0x{synthesize:x};\n    ')
            if mask != '':
                apb.output(f'  add &= {mask};\n    ')
            for c in range(fifos):
                apb.output(f'  in{c} = input_{c};\n')
            apb.output('    }\n')
        apb.output('  }\n')
        apb.function_footer(dest='wrapper', return_value='void')  # load_input()
    else:
        apb.output('  // End of data input\n\n')


def loadcsv(
        embedded_code: bool,
        apb,
        chw: bool,
        input_size: List[int],
        operands: int,
        data,
        csv_file: str = '',
):
    """
    Create C code to load data into FIFO(s) from the camera interface.
    The code is target for simulation (`embedded_code` == `False`) or embedded hardware (`True`).
    Output is written to the `apb` object.
    Additionally, the code creates a CSV file with input data for simulation.
    """
    assert tc.dev is not None
    assert operands == 1  # We don't support multiple operands here

    # FIXME: Support multiple operands
    assert csv_file is not None

    # Cache for faster access
    camera_format = state.input_csv_format

    if not embedded_code:
        apb.output('\n\n  ')

    # HWC ("Little Data") - (Up to) four channels packed into a word (0BGR0BGR0BGR0BGR0BGR....)
    # CHW ("Big Data") - One channel per word
    apb.output(f'// Data input: {"CHW" if chw else "HWC"} '
               f'{input_size[0]}x{input_size[1]}x{input_size[2]}\n')

    fifos = input_size[0] if chw else (input_size[0] + 3) // 4

    if embedded_code:
        apb.output('\n#ifdef USE_FIFO\n')
        apb.output('#define FIFO_SZ 1024\n')
        apb.output('uint32_t fifo[FIFO_SZ];\n')
        apb.output('#endif\n\n')
        apb.output('void load_input(void)\n{\n')
        apb.output('#ifndef USE_FIFO\n')
        apb.output('  int i;\n')
        if chw:
            apb.output('  int j;\n')
            apb.output('  uint32_t d[4], u;\n')
        apb.output('#else\n')
        apb.output('  int i = 0;\n')
        apb.output('  register int head = 0;\n')
        apb.output('  register int tail = 0;\n\n')
        apb.output('#endif\n')
        apb.output('  // Tell tb to start sending pcif data\n')
        apb.output('  sim->trig = 0;\n\n')
        if chw:
            max_len = input_size[1] * ((input_size[2] + 3) // 4)
        else:
            max_len = input_size[1] * input_size[2]
        apb.output('#ifndef USE_FIFO\n')
        apb.output(f'  for (i = 0; i < {max_len}; i++) {{\n')
        fd = 'MXC_CAMERAIF0->dma_data' if not tc.dev.MODERN_SIM else 'MXC_PCIF->fifo_data'
        if not chw:
            for c in range(fifos):
                if not tc.dev.MODERN_SIM:
                    apb.output('    while ((MXC_CAMERAIF0->int_fl & 0x80) == 0); '
                               '// Wait for camera FIFO not empty\n')
                else:
                    apb.output('    while ((MXC_PCIF->int_fl & 0x80) == 0); '
                               '// Wait for camera FIFO not empty\n')
                apb.write(0, fd, fifo=c, comment=f' // Read camera, write FIFO {c}',
                          indent='    ', fifo_wait=False)
        else:
            apb.output('    for (j = 0; j < 4; j++) {\n'
                       '      while ((MXC_PCIF->int_fl & 0x80) == 0); '
                       '// Wait for camera FIFO not empty\n'
                       f'      d[j] = {fd}; '
                       '// Read 24 bits RGB from camera\n'
                       '    }\n')
            for c in range(fifos):
                if c == 0:
                    apb.output('    u = (d[0] & 0xff) '
                               '| ((d[1] & 0xff) << 8) '
                               '| ((d[2] & 0xff) << 16) '
                               '| ((d[3] & 0xff) << 24);\n')
                else:
                    apb.output(f'    u = ((d[0] >> {c * 8}) & 0xff) '
                               f'| (((d[1] >> {c * 8}) & 0xff) << 8) '
                               f'| (((d[2] >> {c * 8}) & 0xff) << 16) '
                               f'| (((d[3] >> {c * 8}) & 0xff) << 24);\n')
                apb.write(0, 'u', fifo=c, comment=f' // Write FIFO {c}',
                          indent='    ', fifo_wait=False)

        apb.output('  }\n')
        apb.output('#else\n')
        apb.output(f'  while (i < {max_len}) {{\n')
        if not tc.dev.MODERN_SIM:
            apb.output('    if (((MXC_CAMERAIF0->int_fl & 0x80) != 0) && (head + 1 != tail) && ')
        else:
            apb.output('    if (((MXC_PCIF->int_fl & 0x80) != 0) && (head + 1 != tail) && ')
        apb.output('((head + 1 != FIFO_SZ) || (tail != 0))) {\n'
                   '      // Camera FIFO not empty and software FIFO not full\n')
        if not tc.dev.MODERN_SIM:
            apb.output('      fifo[head++] = MXC_CAMERAIF0->dma_data; // Read camera\n')
        else:
            apb.output('      fifo[head++] = MXC_PCIF->fifo_data; // Read camera\n')
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

        with open(csv_file, mode='w', encoding='utf-8') as f:
            camera.header(f)

            if camera_format == 888:
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        for c in range(0, input_size[0]):
                            camera.pixel(f, s2u(data[c][row][col]) & 0xff)
                    if chw:
                        # Round up so we have a full 4 bytes
                        for _ in range(input_size[2] % 4):
                            for _ in range(0, input_size[0]):
                                camera.pixel(f, 0)
                    camera.finish_row(f, retrace=state.input_csv_retrace)
            elif camera_format == 555:
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        w = (s2u(data[0][row][col]) & 0xf8) << 7 \
                            | (s2u(data[1][row][col]) & 0xf8) << 2 \
                            | (s2u(data[2][row][col]) & 0xf8) >> 3
                        camera.pixel(f, w >> 8 & 0xff)
                        camera.pixel(f, w & 0xff)
                    camera.finish_row(f, retrace=state.input_csv_retrace)
            elif camera_format == 565:
                for row in range(input_size[1]):
                    for col in range(input_size[2]):
                        w = (s2u(data[0][row][col]) & 0xf8) << 8 \
                            | (s2u(data[1][row][col]) & 0xfc) << 3 \
                            | (s2u(data[2][row][col]) & 0xf8) >> 3
                        camera.pixel(f, w >> 8 & 0xff)
                        camera.pixel(f, w & 0xff)
                    camera.finish_row(f, retrace=state.input_csv_retrace)
            else:
                raise RuntimeError(f'Unknown camera format {camera_format}')

            camera.finish_image(f)
