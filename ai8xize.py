#!/usr/bin/env python3
###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Embedded network and simulation test generator program for Tornado CNN
"""
import hashlib
import os
import signal
import sys

import numpy as np

import apbaccess
import assets
import checkpoint
import cmsisnn
import commandline
import compute
import kbias
import kernels
import load
import onnxcp
import op
import rtlsim
import sampledata
import sampleweight
import stats
import tornadocnn as tc
import yamlcfg
from eprint import eprint
from simulate import conv1d_layer, conv2d_layer, convtranspose2d_layer, \
    linear_layer, passthrough_layer, eltwise_layer, \
    pooling_layer, show_data
from utils import ffs, fls, popcount


def create_net(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        prefix,
        verbose,
        debug,
        debug_computation,
        no_error_stop,
        overwrite_ok,
        log,
        apb_base,
        layers,
        operator,
        input_dim,
        pooled_dim,
        output_dim,
        processor_map,
        output_processor_map,
        kernel_size,
        quantization,
        output_shift,
        input_chan,
        output_chan,
        conv_groups,
        output_width,
        padding,
        dilation,
        stride,
        pool,
        pool_stride,
        pool_average,
        activation,
        data,
        kernel,
        bias,
        big_data,
        fc_weights,
        fc_bias,
        split,
        in_offset,
        out_offset,
        streaming,
        flatten,
        operands,
        eltwise,
        pool_first,
        in_sequences,
        input_filename,
        output_filename,
        c_filename,
        base_directory,
        runtest_filename,
        log_filename,
        zero_unused,
        timeout,
        block_mode,
        verify_writes=False,
        verify_kernels=False,
        embedded_code=False,
        compact_weights=False,
        compact_data=False,
        write_zero_regs=False,
        weight_filename=None,
        sample_filename=None,
        device=84,
        init_tram=False,
        avg_pool_rounding=False,
        fifo=False,
        fast_fifo=False,
        fast_fifo_quad=False,
        zero_sram=False,
        mlator=False,
        oneshot=0,
        stopstart=False,
        mexpress=False,
        riscv=False,
        riscv_exclusive=False,
        riscv_flash=False,
        riscv_cache=False,
        riscv_debug=False,
        riscv_debugwait=True,
        override_start=None,
        increase_start=0,
        override_rollover=None,
        override_delta1=None,
        increase_delta1=0,
        override_delta2=None,
        increase_delta2=0,
        slow_load=False,
        synthesize_input=None,
        mlator_noverify=False,
        input_csv=None,
        input_csv_period=None,
        input_csv_format=None,
        input_csv_retrace=None,
        input_fifo=False,
        input_sync=False,
        sleep=False,
        powerdown=False,
        simple1b=False,
        legacy_test=True,
        log_intermediate=False,
        log_pooling=False,
        allow_streaming=False,
        softmax=False,
        unload=False,
        clock_trim=None,
        repeat_layers=1,
        fixed_input=False,
        max_count=None,
        boost=None,
        forever=False,
):
    """
    Chain multiple CNN layers, create and save input and output
    """
    in_expand = [0] * layers
    out_expand = [0] * layers
    in_expand_thresh = [0] * layers
    out_expand_thresh = [0] * layers
    tram_max = [0] * layers

    input_dim_str = [None] * layers
    output_dim_str = [None] * layers
    kernel_size_str = [None] * layers
    pool_str = [None] * layers
    padding_str = [None] * layers
    pool_stride_str = [None] * layers
    stride_str = [None] * layers

    if riscv_debug:
        riscv = True
    if riscv_cache:
        riscv = True
        riscv_flash = True
    if riscv_flash or riscv_exclusive:
        riscv = True

    # Check streaming and FIFO constraints
    if fast_fifo_quad:
        fast_fifo = True
    if fast_fifo:
        fifo = True
        fifo_group = True
    else:
        fifo_group = False
    if fifo:
        if input_chan[0] > 16 or big_data[0] and input_chan[0] > 4:
            eprint("Using the FIFO is restricted to a maximum of 4 input channels (CHW) or "
                   f"16 channels (HWC); this test is using {input_chan[0]} channels.")
            sys.exit(1)
        if big_data[0] and processor_map[0] & ~0x0001000100010001 != 0 \
           or not big_data[0] and processor_map[0] & ~0x000f000f000f000f != 0:
            eprint("The FIFO is restricted to processors 0, 16, 32, 48 (CHW) or "
                   "0-3, 16-19, 32-35, 48-51 (HWC).")
            sys.exit(1)
        if fast_fifo:
            if big_data[0] and input_chan[0] > 1:
                eprint("Fast FIFO supports only a single CHW input channel; "
                       f"this test is using {input_chan[0]} channels.")
                sys.exit(1)
            elif not big_data[0] and input_chan[0] > 4:
                eprint("Fast FIFO supports up to four HWC input channels; "
                       f"this test is using {input_chan[0]} channels.")
                sys.exit(1)
            if processor_map[0] & 0x0e == 0:
                fifo_group = False
            if output_width[0] != 8:
                eprint('Single-layer fast FIFO setup requires output width of 8.')
                sys.exit(1)
            if operator[0] not in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D]:
                eprint('Fast FIFO requies a convolution operation in the first layer.')
                sys.exit(1)
    elif streaming[0] and not allow_streaming:
        eprint('Streaming in the first layer requires use of a FIFO.')
        sys.exit(1)

    if mlator and (output_dim[-1][0] * output_dim[-1][1] < 4 or output_width[-1] > 8):
        eprint('--mlator should only be used with 4 or more 8-bit outputs per channel; ignoring.',
               error=False)
        mlator = False

    processor_map_0 = processor_map[0]
    if fast_fifo_quad:
        processor_map[0] = processor_map_0 << 48 | processor_map_0 << 32 \
            | processor_map_0 << 16 | processor_map_0

    # Check that input channels are in separate memory instances if CHW (big) data format is used,
    # and calculate input and output expansion
    for ll in range(layers):
        if big_data[ll]:
            p = processor_map[ll] >> (ffs(processor_map[ll]) & ~(tc.dev.P_SHARED-1))
            while p:
                if popcount(p & (tc.dev.P_SHARED-1)) > 1:
                    eprint(f"Layer {ll} uses CHW (big data) input format, but multiple channels "
                           "share the same memory instance. Modify the processor map for "
                           f"layer {ll}.")
                    sys.exit(1)
                p >>= tc.dev.P_SHARED

        out_expand[ll] = (output_chan[ll] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
        out_expand_thresh[ll] = (output_chan[ll] + out_expand[ll]-1) // out_expand[ll]
        if output_chan[ll] > tc.dev.MAX_PROC:
            out_expand_thresh[ll] = \
                min((out_expand_thresh[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                    tc.dev.MAX_PROC)
        in_expand[ll] = (input_chan[ll] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
        in_expand_thresh[ll] = (input_chan[ll] + in_expand[ll]-1) // in_expand[ll]

        if input_chan[ll] > tc.dev.MAX_PROC:
            in_expand_thresh[ll] = \
                min((in_expand_thresh[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                    tc.dev.MAX_PROC)

        assert input_dim[ll][0] * input_dim[ll][1] * in_expand[ll] < tc.dev.FRAME_SIZE_MAX

        # Data memory size check - 4 channels share one instance unless CHW format
        in_size = input_dim[ll][0] * input_dim[ll][1] * in_expand[ll] * operands[ll] \
            * (1 if big_data[ll] else 4)
        if not streaming[ll] and in_size + in_offset[ll] > tc.dev.INSTANCE_SIZE*16:
            eprint(f'Layer {ll}: {1 if big_data[ll] else 4}-channel input size {in_size} '
                   f'with input offset 0x{in_offset[ll]:04x} and expansion {in_expand[ll]}x '
                   f'exceeds data memory instance size of {tc.dev.INSTANCE_SIZE*16}.')
            sys.exit(1)
        out_size = output_dim[ll][0] * output_dim[ll][1] * out_expand[ll] \
            * 4 * output_width[ll] // 8
        if (not streaming[ll] or ll == layers - 1) \
           and out_size + out_offset[ll] > tc.dev.INSTANCE_SIZE*16:
            eprint(f'Layer {ll}: 4-channel, {output_width[ll]}-bit output size {out_size} '
                   f'with output offset 0x{out_offset[ll]:04x} and expansion {out_expand[ll]}x '
                   f'exceeds data memory instance size of {tc.dev.INSTANCE_SIZE*16}.')
            sys.exit(1)

        if operator[ll] != op.CONV1D:
            input_dim_str[ll] = f'{input_dim[ll][0]}x{input_dim[ll][1]}'
            output_dim_str[ll] = f'{output_dim[ll][0]}x{output_dim[ll][1]}'
            kernel_size_str[ll] = f'{kernel_size[ll][0]}x{kernel_size[ll][1]}'
            pool_str[ll] = f'{pool[ll][0]}x{pool[ll][1]}' \
                if pool[ll][0] > 1 or pool[ll][1] > 1 else '0x0'
            padding_str[ll] = f'{padding[ll][0]}/{padding[ll][1]}'
            pool_stride_str[ll] = f'{pool_stride[ll][0]}/{pool_stride[ll][1]}'
            stride_str[ll] = f'{stride[ll][0]}/{stride[ll][1]}'
        else:
            input_dim_str[ll] = f'{input_dim[ll][0]}'
            output_dim_str[ll] = f'{output_dim[ll][0]}'
            kernel_size_str[ll] = f'{kernel_size[ll][0]}'
            pool_str[ll] = f'{pool[ll][0]}' \
                if pool[ll][0] > 1 or pool[ll][1] > 1 else '0'
            padding_str[ll] = f'{padding[ll][0]}'
            pool_stride_str[ll] = f'{pool_stride[ll][0]}'
            stride_str[ll] = f'{stride[ll][0]}'

        if operator[ll] == op.NONE:
            tram_max[ll] = 1
        else:
            tram_max[ll] = max(0, pooled_dim[ll][1] + 2*padding[ll][1] - kernel_size[ll][1]) + 1
            if operator[ll] == op.CONVTRANSPOSE2D:
                tram_max[ll] *= stride[ll][1]

        if input_chan[ll] % conv_groups[ll] != 0 or output_chan[ll] % conv_groups[ll] != 0:
            eprint(f'Layer {ll}: convolution groups {conv_groups[ll]} does not divide'
                   f' the input channels {input_chan[ll]} or output channels {output_chan[ll]}.')
            sys.exit(1)

    # Create comment of the form "k1_b0-1x32x32b_2x2s2p14-..."
    test_name = prefix
    if not embedded_code:
        for ll in range(layers):
            test_name += f'-{input_chan[ll]}x{input_dim_str[ll]}' \
                         f'{"b" if big_data[ll] else "l"}' \
                         f'{"f" if flatten[ll] else ""}_' \
                         + ("avg" if pool_average[ll]
                            and (pool[ll][0] > 1 or pool[ll][1] > 1) else "") \
                         + ("max" if not pool_average[ll]
                            and (pool[ll][0] > 1 or pool[ll][1] > 1) else "") \
                         + f'{pool_str[ll]}s{pool_stride[ll][0]}' \
                         f'p{padding[ll][0]}' \
                         f'm{output_chan[ll]}'
            if activation[ll] == op.ACT_RELU:
                test_name += "_relu"
            elif activation[ll] == op.ACT_ABS:
                test_name += "_abs"
        if repeat_layers > 1:
            test_name += f'_repeat{repeat_layers}'
    MAX_PATH = 255
    if len(test_name) + len(base_directory) > MAX_PATH - 10:
        h = hashlib.md5(test_name.encode()).hexdigest()  # Immutable hash from test name
        cutoff = MAX_PATH - len(test_name) - len(base_directory) - len(h) - 10
        test_name = test_name[:cutoff] + '-' + h
    print(f'{test_name}...')

    os.makedirs(os.path.join(base_directory, test_name), exist_ok=True)

    # Redirect stdout?
    if log:
        sys.stdout = open(os.path.join(base_directory, test_name, log_filename), 'w')
        print(f'{test_name}')

    if block_mode:
        filename = input_filename + '.mem'
    else:
        filename = c_filename + ('_riscv' if riscv else '') + '.c'
    if not block_mode and (embedded_code or compact_data):
        sampledata_header = \
            open(os.path.join(base_directory, test_name, sample_filename), mode='w')
    else:
        sampledata_header = None
    if not block_mode and (embedded_code or mexpress or compact_weights):
        weight_header = \
            open(os.path.join(base_directory, test_name, weight_filename), mode='w')
    else:
        weight_header = None

    # Calculate the groups needed, and groups and processors used overall
    processors_used = 0
    group_map = []
    for ll in range(layers):
        bits = processor_map[ll]
        processors_used |= bits

        if input_chan[ll] > tc.dev.MAX_CHANNELS:
            eprint(f'Layer {ll} is configured for {input_chan[ll]} inputs, which exceeds '
                   f'the system maximum of {tc.dev.MAX_CHANNELS}.')
            sys.exit(1)
        if output_chan[ll] > tc.dev.MAX_CHANNELS:
            eprint(f'Layer {ll} is configured for {output_chan[ll]} outputs, which exceeds '
                   f'the system maximum of {tc.dev.MAX_CHANNELS}.')
            sys.exit(1)
        if (ll != 0 or not fast_fifo_quad) and popcount(processor_map[ll]) != in_expand_thresh[ll]:
            eprint(f'Layer {ll} has {input_chan[ll]} inputs with input expansion '
                   f'{in_expand[ll]}, threshold {in_expand_thresh[ll]}, but '
                   f'enabled processor map 0x{processor_map[ll]:016x} '
                   f'has {popcount(processor_map[ll])} bits instead of the '
                   f'expected number of {in_expand_thresh[ll]}.')
            sys.exit(1)
        if ll == 0 and fast_fifo_quad and popcount(processor_map_0) != in_expand_thresh[ll]:
            eprint(f'Layer {ll} has {input_chan[ll]} inputs with input expansion '
                   f'{in_expand[ll]}, threshold {in_expand_thresh[ll]}, but '
                   f'enabled processor map 0x{processor_map[ll]:016x} '
                   f'has {popcount(processor_map[ll])} bits instead of the '
                   f'expected number of {in_expand_thresh[ll]}.')
            sys.exit(1)
        if popcount(output_processor_map[ll]) != out_expand_thresh[ll]:
            eprint(f'Layer {ll} has {output_chan[ll]} outputs with output expansion '
                   f'{out_expand[ll]}, threshold {out_expand_thresh[ll]}, but '
                   f'processor output map 0x{output_processor_map[ll]:016x} '
                   f'has {popcount(output_processor_map[ll])} bits instead of the '
                   f'expected number of {out_expand_thresh[ll]}.')
            sys.exit(1)
        this_map = []
        for group in range(tc.dev.P_NUMGROUPS):
            if (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
                this_map.append(group)
        group_map.append(this_map)

    groups_used = []
    for group in range(tc.dev.P_NUMGROUPS):
        if ((processors_used |
             output_processor_map[-1]) >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
            groups_used.append(group)

    if 0 not in groups_used:
        eprint('Group 0 is not used, this currently does not work.')
        sys.exit(1)

    # Create ARM code wrapper if needed
    if riscv and not block_mode:
        with open(os.path.join(base_directory, test_name, c_filename + '.c'), mode='w') as f:
            apb = apbaccess.apbwriter(
                f,
                apb_base,
                device=device,
                master=False,
                riscv=False,
                riscv_flash=riscv_flash,
                riscv_cache=riscv_cache,
                riscv_exclusive=riscv_exclusive,
                sleep=sleep,
            )
            apb.copyright_header()

            apb.output(f'// ARM wrapper code\n// {test_name}\n')
            apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

            apb.header(
                embedded_arm=embedded_code,
            )
            apb.main(
                clock_trim=clock_trim,
                embedded_arm=embedded_code,
                groups=list(set().union(groups_used)),
                boost=boost,
                forever=forever,
                mexpress=mexpress,
                fifo=fifo,
            )

    if input_csv is not None:
        csv = os.path.join(base_directory, test_name, input_csv)
    else:
        csv = None

    with open(os.path.join(base_directory, test_name, filename), mode='w') as memfile:
        apb = apbaccess.apbwriter(
            memfile,
            apb_base,
            block_level=block_mode,
            verify_writes=verify_writes,
            no_error_stop=no_error_stop,
            weight_header=weight_header,
            sampledata_header=sampledata_header,
            embedded_code=embedded_code,
            compact_weights=compact_weights or mexpress,
            compact_data=compact_data,
            write_zero_registers=write_zero_regs,
            weight_filename=weight_filename,
            sample_filename=sample_filename,
            device=device,
            verify_kernels=verify_kernels,
            master=groups_used[0] if oneshot > 0 or stopstart else False,
            riscv=True if riscv else None,
            riscv_flash=riscv_flash,
            riscv_cache=riscv_cache,
            riscv_debug=riscv_debug,
            riscv_debugwait=riscv_debugwait,
            fast_fifo=fast_fifo,
            input_csv=input_csv,
            input_csv_format=input_csv_format,
            input_chan=input_chan[0],
            sleep=sleep,
        )

        apb.copyright_header()

        apb.output(f'// {test_name}\n')
        apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n')

        # Human readable description of test
        apb.output(f'\n// Configuring {repeat_layers * layers} '
                   f'layer{"s" if repeat_layers * layers > 1 else ""}:\n')

        for r in range(repeat_layers):
            for ll in range(layers):
                apb.output(f'// Layer {r * layers + ll}: {input_chan[ll]}x{input_dim_str[ll]} ('
                           f'{"streaming " if streaming[ll] else ""}'
                           f'{"flattened " if flatten[ll] else ""}'
                           f'{"CHW/big data)" if big_data[ll] else "HWC/little data)"}, ')
                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    apb.output(f'{pool_str[ll]} {"avg" if pool_average[ll] else "max"} '
                               f'pool with stride {pool_stride_str[ll]}')
                else:
                    apb.output(f'no pooling')
                if operator[ll] in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D]:
                    conv_str = f', {op.string(operator[ll])} with kernel size ' \
                               f'{kernel_size_str[ll]}, '
                else:
                    conv_str = ', no convolution, '
                apb.output(conv_str +
                           f'stride {stride_str[ll]}, '
                           f'pad {padding_str[ll]}, '
                           f'{output_chan[ll]}x{output_dim_str[ll]} output\n')

        apb.output('\n')
        apb.header()

        if embedded_code or compact_data or mexpress:
            apb.output('void memcpy32(uint32_t *dst, const uint32_t *src, int n)\n{\n')
            apb.output('  while (n-- > 0) {\n'
                       '    *dst++ = *src++;\n'
                       '  }\n}\n\n')

        if (embedded_code and not fifo) or compact_data or input_csv:
            # Pre-define data memory loader. Inline later when generating RTL sim.
            if input_fifo:
                apb.output('#define USE_FIFO\n')
            load.load(
                True,
                apb,
                big_data[0],
                processor_map_0,
                in_offset[0],
                [input_chan[0], input_dim[0][0], input_dim[0][1]],
                in_expand[0],
                operands[0],
                in_expand_thresh[0],
                data,
                padding[0],
                split=split,
                fifo=fifo,
                slowdown=slow_load,
                synthesize=synthesize_input,
                riscv_flash=riscv_flash,
                csv_file=csv,
                camera_format=input_csv_format,
                camera_retrace=input_csv_retrace,
                fixed_input=fixed_input,
                debug=debug,
            )
        if not block_mode and (embedded_code or mexpress or compact_weights):
            # Pre-define the kernels and bias values
            kern_offs, kern_len = kernels.load(
                verbose,
                True,
                device,
                apb,
                layers,
                operator,
                kernel,
                kernel_size,
                quantization,
                processor_map,
                output_processor_map,
                input_chan,
                output_chan,
                out_expand,
                out_expand_thresh,
                in_expand,
                in_expand_thresh,
                flatten,
                mexpress,
                verify_kernels,
                riscv_flash and not riscv_cache,
                fast_fifo_quad,
                debug,
                block_mode,
            )
            bias_offs, bias_group, group_bias_max = kbias.load(
                verbose,
                True,
                apb,
                layers,
                bias,
                quantization,
                group_map,
                output_chan,
                streaming,
                debug,
            )

        apb.load_header()

        # Initialize CNN registers

        if verbose:
            print('\nGlobal registers:')
            print('-----------------')

        # Reset
        if device != 84:
            apb.write_fifo_ctl(tc.dev.AON_CTL, tc.dev.AON_READY_SEL,
                               verbose, comment=f' // AON control', force_write=True)

        # Disable completely unused groups
        for group in range(tc.dev.P_NUMGROUPS):
            if group not in groups_used:
                apb.write_ctl(group, tc.dev.REG_CTL, 0,
                              verbose, comment=f' // Disable group {group}')

        # Configure global control registers for used groups
        for _, group in enumerate(groups_used):
            if init_tram:
                # Zero out Tornado RAM
                if not embedded_code:
                    for p in range(tc.dev.P_NUMPRO):
                        for offs in range(tc.dev.TRAM_SIZE):
                            apb.write_tram(group, p, offs, 0, comment='Zero ')
                    apb.output('\n')
                else:
                    for p in range(tc.dev.P_NUMPRO):
                        addr = apb_base + tc.dev.C_GROUP_OFFS*group + tc.dev.C_TRAM_BASE \
                            + p * tc.dev.TRAM_OFFS * 4
                        apb.output(f'  memset((uint32_t *) 0x{addr:08x}, 0, '
                                   f'{tc.dev.TRAM_SIZE}); // Zero TRAM {group}\n')
                        apb.output('\n')

            # Stop state machine - will be overwritten later; enable FIFO
            val = tc.dev.READY_SEL << 1
            if fifo:
                val |= 1 << 15
            if device != 84:
                val |= 1 << 3  # Enable clocks
            if mexpress:
                val |= 1 << 20
            apb.write_ctl(group, tc.dev.REG_CTL, val,
                          verbose, comment=' // Stop SM')
            # SRAM Control - does not need to be changed
            apb.write_ctl(group, tc.dev.REG_SRAM, 0x40e,
                          verbose, comment=' // SRAM control')
            # Number of layers
            apb.write_ctl(group, tc.dev.REG_LCNT_MAX, repeat_layers * layers - 1,
                          verbose, comment=' // Layer count')
            apb.output('\n')

        if device != 84 and zero_sram:
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 0,
                              verbose, comment=' // Data SRAM BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.wait_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 27 | 1 << 18, 1 << 27 | 1 << 18,
                             comment=' // Wait for BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 14, 0,
                               comment=' // Return on BIST error')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                              verbose, comment=' // Reset BIST', force_write=True)
            apb.output('\n')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 2,
                              verbose, comment=' // Mask SRAM BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.wait_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 27 | 1 << 19, 1 << 27 | 1 << 19,
                             comment=' // Wait for BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 15, 0,
                               comment=' // Return on BIST error')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                              verbose, comment=' // Reset BIST', force_write=True)
            apb.output('\n')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 4,
                              verbose, comment=' // Tornado SRAM BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.wait_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 27 | 1 << 20, 1 << 27 | 1 << 20,
                             comment=' // Wait for BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 16, 0,
                               comment=' // Return on BIST error')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                              verbose, comment=' // Reset BIST', force_write=True)
            apb.output('\n')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 6,
                              verbose, comment=' // Bias Rfile BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.wait_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 27 | 1 << 21, 1 << 27 | 1 << 21,
                             comment=' // Wait for BIST')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 17, 0,
                               comment=' // Return on BIST error')
            for group in range(tc.dev.P_NUMGROUPS):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                              verbose, comment=' // Reset BIST', force_write=True)
            apb.output('\n')

        if block_mode or not (embedded_code or mexpress or compact_weights):
            kern_offs, kern_len = kernels.load(
                verbose,
                embedded_code,
                device, apb,
                layers,
                operator,
                kernel,
                kernel_size,
                quantization,
                processor_map,
                output_processor_map,
                input_chan,
                output_chan,
                out_expand,
                out_expand_thresh,
                in_expand,
                in_expand_thresh,
                flatten,
                mexpress,
                verify_kernels,
                riscv_flash and not riscv_cache,
                fast_fifo_quad,
                debug,
                block_mode,
            )
            bias_offs, bias_group, group_bias_max = kbias.load(
                verbose,
                embedded_code,
                apb,
                layers,
                bias,
                quantization,
                group_map,
                output_chan,
                streaming,
                debug,
            )
        else:
            apb.output('  load_kernels();\n')
            if verify_kernels:
                apb.output('  if (!verify_kernels()) return 0;\n')
            if max(group_bias_max) > 0:
                apb.output('  load_bias();\n')

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors     = 0x{processors_used:016x}')
            print(f'Used groups         = {groups_used}')

            print('\nPer-group configuration:')
            print('-----------------------')
            print(f'Used bias memory    = {group_bias_max}')

            print('\nPer-layer configuration:')
            print('------------------------')
            if repeat_layers > 1:
                print(f'Layer repeat count  = {repeat_layers}')
            print(f'Input dimensions    = {input_dim}')
            print(f'Input channels      = {input_chan}')
            print(f'Convolution groups  = {conv_groups}')
            print(f'Flatten             = {flatten}')
            print('Processor map       = [',
                  ', '.join('0x{:016x}'.format(k) for k in processor_map), ']', sep='',)
            if device != 84:
                print(f'Input expansion     = {in_expand}')
                print(f'Expansion threshold = {in_expand_thresh}')
                print(f'Element-wise op     = [',
                      ', '.join(op.string(k, elt=True) for k in eltwise), ']', sep='',)
                print(f'Operand expansion   = {operands}')

            print('Input offsets       = [',
                  ', '.join('0x{:04x}'.format(k) for k in in_offset), ']', sep='',)

            print(f'Output dimensions   = {output_dim}')
            print(f'Output channels     = {output_chan}')
            print('Output processors   = [',
                  ', '.join('0x{:016x}'.format(k) for k in output_processor_map), ']', sep='',)
            if device != 84:
                print(f'Output expansion    = {out_expand}')
                print(f'Expansion threshold = {out_expand_thresh}')
                print(f'Output data bits    = {output_width}')
            print('Output offsets      = [',
                  ', '.join('0x{:04x}'.format(k) for k in out_offset), ']', sep='',)

            print(f'Group map           = {group_map}')

            print(f'Kernel offsets      = {kern_offs}')
            print(f'Kernel lengths      = {kern_len}')
            if device != 84:
                print(f'Kernel dimensions   = {kernel_size}')
                print(f'Kernel size         = {quantization}')
                print(f'Output shift        = {output_shift}')
            print(f'Operator            = [',
                  ', '.join(op.string(k) for k in operator), ']', sep='',)
            print(f'Stride              = {stride}')

            print(f'Padding             = {padding}')
            print(f'Group with bias     = {bias_group}')
            print(f'Bias offsets        = {bias_offs}')
            print(f'Pooling             = {pool}')
            print(f'Pooling stride      = {pool_stride}')
            print(f'Pooled dimensions   = {pooled_dim}')
            print(f'Streaming           = {streaming}')
            print('')

        if verbose:
            print('Layer register configuration:')
            print('-----------------------------')

        # Configure per-layer control registers
        for r in range(repeat_layers):
            for ll in range(layers):

                local_source = False
                for _, group in enumerate(groups_used):
                    # Local output must be used:
                    # - When parallel processing is enabled (not currently supported), or
                    # - When there are gaps in the output, and
                    #   - the gaps are non-uniform, or
                    #   - the layer is in passthrough mode
                    # Uniform gaps (when not in passthrough mode) can be achieved using the
                    # time slot offset.

                    if local_source:
                        break

                    gap_max, gap_min = 0, tc.dev.MAX_PROC
                    gmap = \
                        output_processor_map[ll] & 2**tc.dev.P_NUMPRO - 1 << group*tc.dev.P_NUMPRO
                    if popcount(gmap) > 1:
                        p = ffs(gmap)
                        while p < fls(gmap):
                            gap = ffs(gmap & ~(2**(p+1) - 1)) - p - 1
                            gap_min, gap_max = min(gap, gap_min), max(gap, gap_max)
                            p += gap + 1
                        local_source = \
                            gap_min != gap_max or gap_max > 0 and operator[ll] == op.NONE

                    # FIXME: Check that we don't overlap by-16 groups when in local_source mode
                    # FIXME: Non-uniform gaps are not supported

                for _, group in enumerate(groups_used):
                    apb.output(f'\n  // Layer {r * layers + ll} group {group}\n')

                    if device == 84 and operator[ll] == op.CONV1D:
                        # For 1D convolutions on AI84, the column count is always 3, and the
                        # row count is divided by 3. Padding is divided by 3.
                        val = (padding[ll][0] // 3 << 8) \
                               | (input_dim[ll][0] + 2*padding[ll][0]) // 3 - 1
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_RCNT, val,
                                       verbose, comment=' // Rows')
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_CCNT, 2,
                                       verbose, comment=' // Columns')
                    else:
                        # Configure row count
                        # [9:0]   maxcount: lower 8 bits = total of width + pad - 1
                        # [17:16] pad: 2 bits pad
                        if flatten[ll]:
                            val = 0
                        else:
                            if operator[ll] == op.CONVTRANSPOSE2D:
                                val = stride[ll][1]*input_dim[ll][0] - 1
                            else:
                                val = input_dim[ll][0] - 1
                        assert padding[ll][0] < 2**2
                        assert val + 2*padding[ll][0] < 2**10
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_RCNT,
                                       padding[ll][0] << 16 | val + 2*padding[ll][0],
                                       verbose, comment=' // Rows')

                        # Configure column count (evaluates to 0 for 1D convolutions)
                        # [9:0]   width including padding - 1
                        # [17:16] pad count (0 = no pad, 1 = half pad, 2 = full pad)
                        if flatten[ll]:
                            val = 0
                        else:
                            if operator[ll] == op.CONVTRANSPOSE2D:
                                val = stride[ll][1]*input_dim[ll][1] - 1
                            else:
                                val = input_dim[ll][1] - 1
                        assert padding[ll][1] < 2**2
                        assert val + 2*padding[ll][1] < 2**10
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_CCNT,
                                       padding[ll][1] << 16 | val + 2 * padding[ll][1],
                                       verbose, comment=' // Columns')

                    # Configure pooling row count
                    val = pool[ll][0]-1
                    if device == 84 and pool[ll][0] == 1:
                        val = 1
                    else:
                        val = pool[ll][0]-1
                        assert val < 2**4
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_PRCNT, val,
                                   verbose, comment=' // Pooling rows')

                    # Configure pooling column count
                    if device == 84 and pool[ll][1] == 1:
                        val = 1
                    else:
                        val = pool[ll][1]-1
                        assert val < 2**4
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_PCCNT, val,
                                   verbose, comment=' // Pooling columns')

                    # Configure pooling stride count
                    if pool[ll][0] > 1 or pool[ll][1] > 1:
                        val = pool_stride[ll][0]-1
                    elif operator[ll] == op.CONVTRANSPOSE2D:
                        val = 0
                    else:
                        val = stride[ll][0]-1
                    if device == 84 and operator[ll] == op.CONV1D:
                        val //= 3
                    assert val < 2**4
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STRIDE, val,
                                   verbose, comment=' // Stride')

                    val = out_offset[ll] // 4
                    if not local_source:
                        # Configure SRAM write pointer -- write ptr is global
                        # Get offset to first available instance of the first used processor of the
                        # next layer.
                        if operator[ll] != op.NONE:
                            instance = ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1)
                        else:
                            instance = ffs(output_processor_map[ll]
                                           & 2**tc.dev.P_NUMPRO - 1 << group*tc.dev.P_NUMPRO) \
                                & ~(tc.dev.P_SHARED-1)

                        val |= (instance % tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE \
                            | (instance // tc.dev.P_SHARED) << tc.dev.INSTANCE_SHIFT
                    else:
                        instance = ffs(output_processor_map[ll] >> group * tc.dev.P_SHARED) \
                               & ~(tc.dev.P_SHARED-1)
                        val |= (instance + group * tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE
                    assert val < 2**17
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_BASE, val,
                                   verbose, comment=' // SRAM write ptr')

                    if device == 84:
                        # Configure write pointer mask offset count
                        # [15:0]  Timeslot offset
                        #         [11:0]  12 bits for memory - word address every time
                        #                 we reach limit
                        #         [13:12] instance in group
                        #         [15:14] by-16 group
                        # [31:16] Mask offset (0x10000000, required when writing more than 4 masks)
                        if input_chan[ll] * kern_len[ll] > 4:
                            val = 1 << tc.dev.INSTANCE_SHIFT + 16
                        else:
                            val = 0
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_OFFS, val,
                                       verbose, comment=' // Mask offset count')
                    else:
                        # [15:0] Write Pointer Timeslot Offset Register
                        # Used for 1x1 convolution, and pooling without convolution
                        if operator[ll] == op.CONV2D and kernel_size[ll] == [1, 1]:
                            val = 1
                        elif operator[ll] == op.NONE:
                            if popcount(processor_map[ll]) > 4 \
                               or operands[ll] > 1 and in_expand[ll] > 1:
                                val = tc.dev.INSTANCE_SIZE * 4
                            else:
                                val = tc.dev.INSTANCE_SIZE
                        else:
                            val = 0
                        assert val < 2**17
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_TOFFS, val,
                                       verbose, comment=' // Write ptr time slot offs')

                        # [15:0] Write Pointer Mask Offset Register
                        val = 1 << tc.dev.INSTANCE_SHIFT
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_MOFFS, val,
                                       verbose, comment=' // Write ptr mask offs')

                        # [15:0] Write Pointer Multi-Pass Channel Offset Register
                        val = output_width[ll] // 8
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_CHOFFS, val,
                                       verbose, comment=' // Write ptr multi-pass channel offs')

                    # Configure sram read ptr count -- read ptr is local
                    # Source address must match write pointer of previous layer (minus global
                    # offset)
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_RPTR_BASE,
                                   in_offset[ll] // 4,
                                   verbose, comment=' // SRAM read ptr')

                    # Configure per-layer control
                    # [3:0] s_slave: enable the by-4 group within the by-16 mask RAM to slave
                    #                to first input volume; also enable timeslot
                    # [4]   m_slave: slaves to 16x masters
                    # [5]   master: sums all 16 processor outputs (vs 4 sums)
                    # [6]   parallel: equals CHW/big data (per layer control)
                    # [7]   pool_enable
                    # [8]   maxpool_enable
                    # [9]   activation_enable
                    # [10]  cpad_only (column pad only, no row pad) for parallel processing
                    # [11]  sramlsrc: global/local output SRAM data memory input select
                    # [15:12] cnnsiena: enable externally sourced summed values from other
                    #         processors
                    # [16]  bigdwrt (AI85 only) Enables 32-bit output
                    val = (0x200 if activation[ll] == op.ACT_RELU else 0) | \
                          (0x100 if not pool_average[ll] else 0) | \
                          (0x80 if pool[ll][0] > 1 or pool[ll][1] > 1 else 0) | \
                          (0x40 if big_data[ll] else 0) | \
                          (0x20)
                    if not local_source:
                        val |= 0x800

                    if device != 84 and output_width[ll] != 8:
                        val |= 1 << 16

                    if (ll != 0 or not fast_fifo_quad) \
                       and operator[ll] != op.NONE and group == groups_used[0]:
                        # Set external source for other active processing groups (can be zero if no
                        # other groups are processing). Do not set the bit corresponding to this
                        # group (e.g., if group == 0, do not set bit 12)
                        sources = 0
                        for t in range(groups_used[0]+1, tc.dev.P_NUMGROUPS):
                            # See if any processors other than this one are operating
                            # and set the cnnsiena bit if true
                            if (processor_map[ll] >> (t * tc.dev.P_NUMPRO)) % 2**tc.dev.P_NUMPRO:
                                sources |= 1 << t
                        val |= sources << 12
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL, val,
                                   verbose, comment=' // Layer control')

                    if device != 84:
                        flatten_prod = 0
                        # [3:0]  inpchexp[3:0]
                        # [7:4]  wptr_inc[3:0]
                        # [16:8] xpch_max[8:0] Selects the maximum channel processor number used
                        #                      in channel expansion mode (bottom 3 are for bits)
                        if flatten[ll]:
                            # Store all bits, top programmed in post processing register
                            flatten_prod = \
                                in_expand[ll] * pooled_dim[ll][0] * pooled_dim[ll][1] - 1
                            in_exp = flatten_prod % 2**4
                        else:
                            in_exp = in_expand[ll] - 1

                        assert in_exp < 2**4  # Cannot have more than 4 bits

                        val = (fls(output_processor_map[ll])
                               - (ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1))) \
                            * quantization[ll] << 8 \
                            | in_exp
                        if operator[ll] != op.NONE:
                            assert out_expand[ll] <= 2**4  # Cannot have more than 4 bits (+1)
                            val |= (out_expand[ll] - 1) << 4

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL2, val,
                                       verbose, comment=' // Layer control 2')

                    # Configure mask count
                    # Restriction: Every one of the mask memories will have to start from same
                    # offset
                    # AI84:
                    # [6:0]   Max count (output channels)
                    # [7]     RFU
                    # [14:8]  Starting address for group of 16
                    # [15]    RFU
                    # [23:16] Bias pointer starting address
                    # [24]    Bias enable
                    # [31:25] RFU
                    # AI85:
                    # [15:0]  Max count (output channels)
                    # [31:16] Starting address for group of 16
                    if device == 84:
                        val = kern_offs[ll] << tc.dev.MCNT_SAD_OFFS \
                            | (kern_len[ll] << tc.dev.MCNT_MAX_OFFS) - 1
                        if group == bias_group[ll]:
                            # Enable bias only for one group
                            val |= 0x1000000 | bias_offs[ll] << 16
                    else:
                        oned_sad = 0
                        if operator[ll] in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D]:
                            in_exp = in_expand[ll]
                            if flatten[ll]:
                                in_exp *= pooled_dim[ll][0] * pooled_dim[ll][1]
                            kl = (((fls(output_processor_map[ll])
                                    - (ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1))) + 1)
                                  * quantization[ll]) * out_expand[ll] * in_exp \
                                - quantization[ll]
                            if ll == 0 and fast_fifo_quad:
                                kl = (kl + 3) // 4
                            koffs, oned_sad = divmod(9 * kern_offs[ll],
                                                     kernel_size[ll][0] * kernel_size[ll][1])
                            koffs *= 8

                            assert koffs < 2**16
                            assert kl + koffs < 2**16
                            # kern_offs is always bytes
                            val = \
                                koffs << tc.dev.MCNT_SAD_OFFS | kl + koffs << tc.dev.MCNT_MAX_OFFS
                        else:
                            assert operator[ll] == op.NONE
                            val = (out_expand[ll] - 1) * 8
                            assert val < 2**16
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT, val,
                                   verbose, comment=' // Mask offset and count')

                    if device != 84:
                        #   [3:0] tscnt_max[3:0]      Maximum timeslot count register
                        #   [7:4] oned_sad[3:0]       Start mask address (offset within 9 byte
                        #                             mask)
                        #  [11:8] oned_width[3:0]     1D mask width (0-9). Width > 0 enables 1D.
                        #    [12] oned_iena           Input data is 1-dimensional
                        #    [13] ewise_ena           Enable element-wise operation
                        # [17:14] ewise_fun           Elementwise function select
                        #         .3   - Enables 2D convolution of the ewise result.
                        #                Standard 2D processing applies.
                        #         .2   - Enables pre-pooling of the input data before element-wise
                        #                operation.
                        #         .1/0 - 2'b00 = add
                        #                2'b01 = subtract
                        #                2'b10 = bitwise XOR
                        #                2'b11 = bitwise OR.
                        # [21:18] ewise_cnt           Element wise operand count

                        val = 0
                        if operator[ll] == op.NONE:
                            val |= (popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                             % 2**tc.dev.P_NUMPRO) * output_width[ll]//8 - 1) // 4
                            assert 0 <= val < 2**4
                        if operator[ll] == op.CONV1D:
                            val |= kernel_size[ll][0] << 8 | 1 << 12
                            assert kernel_size[ll][0] < 2**4
                        elif (operator[ll] == op.CONV2D and kernel_size[ll] == [1, 1]
                              or operator[ll] == op.NONE and operands[ll] == 1):
                            val |= 1 << 8
                        if operands[ll] > 1:
                            val |= \
                                1 << 13 | op.eltwise_fn(eltwise[ll]) << 14 | operands[ll] - 1 << 18
                            if (pool[ll][0] > 1 or pool[ll][1] > 1) and pool_first[ll]:
                                val |= 1 << 16
                            if operator[ll] in [op.CONV2D, op.CONVTRANSPOSE2D]:
                                val |= 1 << 17
                        assert 0 <= oned_sad < 2**4
                        val |= oned_sad << 4

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_ONED, val,
                                       verbose, comment=' // 1D')

                    # Configure tram pointer max
                    if operator[ll] == op.CONV1D or \
                       operator[ll] == op.CONV2D and kernel_size[ll] == [1, 1]:
                        if flatten_prod >= 2**4:
                            assert flatten_prod < 2**16
                            val = flatten_prod << 16 \
                                | (flatten_prod + pooled_dim[ll][0] * pooled_dim[ll][1])
                        else:
                            val = 0
                    else:
                        val = tram_max[ll] - 1
                        assert val < 2**16
                        if ll > 0 and streaming[ll]:
                            prev_max = np.multiply(tram_max[:ll], in_expand[:ll]).sum()
                            assert prev_max < 2**12
                            val += prev_max
                            assert val < 2**16
                            val |= prev_max << 16
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_TPTR, val,
                                   verbose, comment=' // TRAM ptr max')

                    if device != 84:
                        # Compensate for the smaller weights by adjusting the output shift
                        if quantization[ll] == 1:
                            val = 1 << 22
                        elif quantization[ll] == 2:
                            val = 2 << 22
                        elif quantization[ll] == 4:
                            val = 3 << 22
                        else:
                            assert quantization[ll] == 8
                            val = 0  # Do not shift
                        # Scale Control - bit 4 determines shift direction (1>>,0<<),
                        # bits[3:0] determine magnitude
                        if output_shift[ll] < 0:
                            val |= (-output_shift[ll] | 2**4) << 13
                        else:
                            val |= output_shift[ll] << 13

                        # [24] ts_ena
                        # [25] onexone_ena

                        if group == bias_group[ll]:
                            # Enable bias only for one group
                            assert bias_offs[ll] < 2**12
                            val |= 1 << 12 | bias_offs[ll]

                        if operator[ll] == op.NONE:
                            if operands[ll] == 1:
                                val |= 3 << 24
                            else:
                                val |= 1 << 24

                        if activation[ll] == op.ACT_ABS:
                            val |= 1 << 26

                        if flatten_prod >= 2**4:
                            val |= 1 << 27 | (flatten_prod >> 4) << 18  # flatten_ena, xpmp_cnt

                        if operator[ll] == op.CONVTRANSPOSE2D:
                            val |= 1 << 28

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_POST, val,
                                       verbose, comment=' // Post processing register')

                    # Configure mask and processor enables
                    # [15:0]  processor enable
                    # [31:16] mask enable
                    # When the input data is sourced from 16 independent byte streams, all 16
                    # processors and compute elements need to be enabled.  If there were only 4
                    # input channels, 0x000f000f would be correct.
                    #
                    # Enable at most 16 processors and masks
                    val = (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                    if operator[ll] in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D]:
                        val = val << 16 | val
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_ENA, val,
                                   verbose, comment=' // Mask and processor enables')

                    if ll == 0 and fifo:
                        # Start: 1
                        if override_start is not None:
                            stream_start = override_start
                        elif streaming[ll]:
                            stream_start = (pool[ll][0] - 1) * input_dim[ll][1] + pool[ll][1]
                        else:
                            val = input_dim[0][0] * input_dim[0][1]
                            if big_data[0]:
                                val = (val + 3) // 4
                            stream_start = val
                        assert stream_start < 2**14

                        if streaming[ll]:
                            # Delta 1: This layer's pooling stride
                            if override_delta1 is not None:
                                delta1 = override_delta1
                            else:
                                delta1 = (pool_stride[ll][1] - 1) * operands[ll]
                            assert delta1 < 2**5
                            if override_delta2 is not None:
                                delta2 = override_delta2
                            else:
                                delta2 = (pool[ll][0] - 1) * input_dim[ll][1] * operands[ll]
                            assert delta2 < 2**12
                        else:
                            delta1 = 0
                            delta2 = 0

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STREAM1, stream_start,
                                       verbose, comment=' // Stream processing start')
                        val = delta2 << 16 | delta1 << 4
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STREAM2, val,
                                       verbose, comment=' // Stream processing delta')
                    elif ll > 0 and streaming[ll]:
                        # [13:0]:  strm_isval[13:0]  Per stream start count - based on previous
                        #                            layer tptr_inc count
                        # Start count – defines the current layer rcnt (TRAM shift count) that
                        # triggers processing of the next layer

                        # [16:12]: strm_dsval1[4:0]  Per stream in-row delta count - based on
                        #                            previous layer tptr_inc count
                        # [31:20]: strm_dsval2[11:0] Per stream multi-row delta count - based on
                        #                            previous layer tptr_inc count
                        #
                        # Delta1 count – defines the current layer count once the start count is
                        # triggered that enables incremental layer processing.  This count is
                        # used when layer processing is contained within a single row.
                        # Delta2 count – defines the current layer count once the start count is
                        # triggered that enables incremental layer processing.  This count is
                        # used when layer processing spans multiple rows.

                        # Start: Prior layer's padded pooled row width * prior layer's kernel
                        # height + prior layer's kernel width + prior layer's pad
                        stream_start = (pooled_dim[ll-1][1] + 2 * padding[ll-1][1]) \
                            * (kernel_size[ll-1][0] - 1 + pool[ll][0] - 1) \
                            + kernel_size[ll-1][1] - 1 + pool[ll][1] + increase_start
                        assert stream_start < 2**14

                        # Delta 1: This layer's pooling stride
                        delta1 = pool_stride[ll][1] * operands[ll] + increase_delta1
                        assert delta1 < 2**5
                        # Delta 2: (This layer's pooling - 1) * full prior layer's padded rows +
                        # prior layer's pad
                        delta2 = (pool_stride[ll][0] - 1) \
                            * (pooled_dim[ll-1][1] + 2 * padding[ll-1][1]) \
                            + pool[ll][1] * operands[ll] + increase_delta2
                        assert delta2 < 2**12

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STREAM1, stream_start,
                                       verbose, comment=' // Stream processing start')
                        # [3:0]:   strm_invol[3:0]   Per stream invol offset - based on stream
                        #                            count
                        val = sum(in_expand[:ll])
                        assert val < 2**4
                        val |= delta2 << 16 | delta1 << 4
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STREAM2, val,
                                       verbose, comment=' // Stream processing delta')

                    if fifo and streaming[ll]:
                        if ll == 0 and override_rollover is not None:
                            val = override_rollover
                        else:
                            if big_data[ll]:
                                # FIXME stream_start + max(stride[ll][1], pool_stride[ll][1])
                                val = 12
                            else:
                                val = stream_start + (pool[ll][0] - 1) * input_dim[ll][1] \
                                    + max(stride[ll][1], pool_stride[ll][1], pool[ll][1])
                            # Rollover must be multiple of multi-pass:
                            rem = val % in_expand[ll]
                            if rem > 0:
                                val = val + in_expand[ll] - rem
                        assert val < 2**17

                        # Check rollover vs available data memory
                        if in_offset[ll] < out_offset[ll]:
                            if in_offset[ll] + val * 4 >= out_offset[ll]:
                                eprint('Overlapping input and output: '
                                       f'in_offset 0x{in_offset[ll]:08x} < '
                                       f'out_offset 0x{out_offset[ll]:08x}, '
                                       f'rollover 0x{val:08x}.',
                                       error=not no_error_stop)
                                if not no_error_stop:
                                    sys.exit(1)
                        else:
                            if out_offset[ll] + val * 4 >= in_offset[ll]:
                                eprint('Overlapping input and output: '
                                       f'in_offset 0x{in_offset[ll]:08x} >= '
                                       f'out_offset 0x{out_offset[ll]:08x}, '
                                       f'rollover 0x{val:08x}.',
                                       error=not no_error_stop)
                                if not no_error_stop:
                                    sys.exit(1)
                        if in_offset[ll] + val * 4 >= tc.dev.INSTANCE_SIZE * tc.dev.P_SHARED * 4:
                            eprint('Input plus rollover exceeds instance size: '
                                   f'in_offset 0x{in_offset[ll]:08x}, '
                                   f'out_offset 0x{out_offset[ll]:08x}, '
                                   f'rollover 0x{val:08x}, '
                                   f'instance size 0x{tc.dev.INSTANCE_SIZE*4:08x}.',
                                   error=not no_error_stop)
                            if not no_error_stop:
                                sys.exit(1)

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_FMAX, val,
                                       verbose, comment=' // Rollover')

                    if ll == 0 and fifo:
                        val = input_dim[0][0] * input_dim[0][1]
                        if big_data[0]:
                            val = (val + 3) // 4
                        assert val < 2**20
                        apb.write_ctl(group, tc.dev.REG_IFRM, val, verbose,
                                      comment=' // Input frame size')

        if zero_unused:
            for r in range(repeat_layers):
                for ll in range(layers, tc.dev.MAX_LAYERS):
                    for _, group in enumerate(groups_used):
                        for reg in range(tc.dev.MAX_LREG+1):
                            if reg == tc.dev.LREG_RFU:  # Register 2 not implemented
                                continue
                            apb.write_lreg(group, r * layers + ll, reg, 0,
                                           verbose, force_write=True,
                                           comment=f' // Zero unused layer {ll} registers')

        if not fifo:
            # Load data memory
            if embedded_code or compact_data or input_csv:
                # Do the actual code generation later
                apb.output('\n  load_input(); // Load data input\n\n')
            else:
                load.load(
                    embedded_code,
                    apb,
                    big_data[0],
                    processor_map_0,
                    in_offset[0],
                    [input_chan[0], input_dim[0][0], input_dim[0][1]],
                    in_expand[0],
                    operands[0],
                    in_expand_thresh[0],
                    data,
                    padding[0],
                    split=split,
                    fifo=fifo,
                    slowdown=slow_load,
                    riscv_flash=riscv_flash,
                    csv_file=csv,
                    camera_format=input_csv_format,
                    camera_retrace=input_csv_retrace,
                    debug=debug,
                )

        if verbose:
            print('\nGlobal registers:')
            print('-----------------')

        # Configure the FIFOs when we're using them
        if fifo:
            apb.output('\n')

            # FIFO control
            # [1:0] rdy_sel[1:0]        Sets the number of wait states added to the APB access.
            # [4:2] fthres[2:0]         FIFO almost full threshold. If the difference between the
            #                           write and read pointer exceeds this number of bytes, the
            #                           almost full flag is set.
            # [9:7] ethres[2:0]         FIFO almost empty threshold. If the difference between the
            #                           write and read pointer falls below this number of bytes,
            #                           the almost empty flag is set.
            # [11] fifo_cpl             Setting this bit forces the FIFO to operate in lock-step.
            #                           Data available status is dependent on all FIFO having
            #                           identical write pointer values.
            # [15:12] fifo_ena          Per FIFO enable. A logic 1 enables the FIFO. Unused FIFOs
            #                           must be disabled.
            # [19:16] full_iena         FIFO full interrupt enable. Logic '1' enables the interrupt
            #                           request based on the fifo full flag.
            # [23:20] empty_iena        FIFO empty interrupt enable. Logic '1' enables the
            #                            interrupt request based on the fifo empty flag.
            # [27:24] almost_full_iena  FIFO almost full interrupt enable. Logic '1' enables the
            #                           interrupt request based on the fifo almost full threshold
            #                           flag.
            # [31:28] almost_empty_iena FIFO almost empty interrupt enable. Logic '1' enables the
            #                           interrupt request based on the fifo almost empty threshold
            #                           flag.
            if not fast_fifo:
                val = 0x02 << 2 | 0x02 << 7 | 1 << 11 | tc.dev.FIFO_READY_SEL
                for i in range(input_chan[0]):
                    if processor_map_0 & 1 << (i % tc.dev.P_NUMGROUPS) * tc.dev.P_NUMPRO != 0:
                        val |= 1 << i % tc.dev.P_NUMGROUPS + 12
                apb.write_fifo_ctl(tc.dev.FIFO_CTL, val,
                                   verbose, comment=f' // FIFO control')
            else:
                apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_IE, 0,
                                        verbose, comment=f' // Fast FIFO interrupt enable')
                val = 10 << 4  # Async, threshold 10
                apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_CR, val,
                                        verbose, comment=f' // Fast FIFO control')

        # [0]     enable
        # [2:1]   rdy_sel  (wait states - set to max)
        # [3]     clock_ena
        # [4]     calcmax
        # [5]     poolena
        # [6]     bigdata
        # [7]     actena
        # [8]     one-shot (stop after single layer)
        # [10:9]  ext_sync[1:0] (slave to other group)
        # [11]    ext_sync[2] (external slave)
        # [12]    irq
        # [13]    pool_rnd
        # [14]    strm_ena - cnn_ctl register bit 14. Master stream processor enable. Layers are
        #         processed up to the first layer with a zero start count value. After the last
        #         stream layer (non-zero start and delta >> values) processing is complete,
        #         standard processing follows for the remaining layers.
        # [15]    fifo_ena
        # [16]    mlat_ena
        # [18:17] mlat_sel
        # [19]    lil_buf - enables ifrm and frm_max
        # [20]    mexpress = Enable loading of the mask memories using packed data. A change in
        #         state of the two lsb of the address trigger a reload of the address counter.
        # [21]    simple1b - Enable simple logic for 1 bit weights.
        # [22]    ffifoena - Fast FIFO enable.  Enables the datapath between the ME17x
        #         synch/asynch FIFO and the CNN
        # [23]    fifogrp - Enables sending all "little data" channels to the first 4 processors.
        #         When this bit is not set, each byte of FIFO data is directed to the first little
        #         data channel of each x16 processor.
        # [26:24] fclk_dly[2:0] - Selects the clock delay of the fast FIFO clock relative to the
        #         primary CNN clock.
        val = 1 << 14 if any(streaming) else 0
        if avg_pool_rounding:
            val |= 1 << 13
        if fifo:
            val |= 1 << 11
        if fifo and any(streaming):
            val |= 1 << 19
        if device != 84:
            val |= 1 << 3  # Enable clocks
        if mexpress:
            val |= 1 << 20
        if simple1b:
            val |= 1 << 21
        if fast_fifo_quad:
            val |= 1 << 31  # Qupac bit

        # Enable all needed groups except the first one
        for _, group in enumerate(groups_used):
            # Turn on the FIFO for this group if it's being loaded
            if fifo and processor_map_0 & 0x0f << group * 16 != 0:
                fval = 1 << 15
                if fast_fifo:
                    fval |= 1 << 22
                if fifo_group:
                    fval |= 1 << 23
            elif fifo:
                fval = 1 << 15
            else:
                fval = 0
            if group != groups_used[0]:
                fval |= 0x01
            apb.write_ctl(group, tc.dev.REG_CTL, val | 0x800 | tc.dev.READY_SEL << 1
                          | fval | groups_used[0] << 9,
                          verbose, comment=f' // Enable group {group}')

        if powerdown:
            unused_groups = [group for group in list(range(tc.dev.P_NUMGROUPS))
                             if group not in groups_used]
            val2 = 0
            for _, group in enumerate(unused_groups):
                val2 |= 1 << 12 + group
            apb.write_fifo_ctl(tc.dev.AON_CTL, val2 | tc.dev.AON_READY_SEL,
                               verbose, comment=f' // AON control')

        if embedded_code:
            apb.output('\n  CNN_START; // Allow capture of processing time\n')

        # Master control - go
        if fifo and processor_map_0 & 0x0f << groups_used[0] * 16 != 0:
            val |= 1 << 15
            if fast_fifo:
                val |= 1 << 22
            if fifo_group:
                val |= 1 << 23
        apb.write_ctl(groups_used[0], tc.dev.REG_CTL, val | tc.dev.READY_SEL << 1 | 0x01,
                      verbose, comment=f' // Master enable group {groups_used[0]}')

        if fifo:
            # Load data memory
            if compact_data or input_csv:
                # Do the actual code generation later
                apb.output('\n  load_input(); // Load data input\n\n')
            else:
                load.load(
                    False,
                    apb,
                    big_data[0],
                    processor_map_0,
                    in_offset[0],
                    [input_chan[0], input_dim[0][0], input_dim[0][1]],
                    in_expand[0],
                    operands[0],
                    in_expand_thresh[0],
                    data,
                    padding[0],
                    split=split,
                    fifo=fifo,
                    slowdown=slow_load,
                    synthesize=synthesize_input,
                    csv_file=csv,
                    camera_format=input_csv_format,
                    camera_retrace=input_csv_retrace,
                    debug=debug,
                )

        apb.load_footer()
        # End of input

    in_map = apb.get_mem()

    if verbose:
        print('')

    def run_eltwise(
            data,
            ll,
    ):
        """
        In-flight element-wise operations
        """
        if operator[ll] == op.NONE:
            # Let element-wise do 32-bit, else 8-bit only
            o_width = output_width[ll]
        else:
            o_width = 8
        d_shape = data.shape

        data, out_size = eltwise_layer(
            eltwise[ll],
            ll,
            verbose,
            data[0].shape,
            output_shift[ll],
            data,
            output_width=o_width,
            device=device,
            debug=debug_computation,
            operands=operands[ll],
        )
        assert out_size[0] == d_shape[1] \
            and out_size[1] == d_shape[2] and out_size[2] == d_shape[3]

        return data

    data_buf = [data]
    # Compute layer-by-layer output and chain results into input
    for ll in range(layers):
        if debug_computation:
            compute.debug_open(ll, base_directory, test_name, log_filename)

        # Concatenate input data if needed
        if in_sequences[ll] is not None:
            if isinstance(in_sequences[ll], list):
                try:
                    data = np.concatenate([data_buf[i + 1] for i in in_sequences[ll]], axis=0)
                except ValueError as err:
                    eprint('Error in input data concatenation layer:', err)
                    sys.exit(1)
            else:
                data = data_buf[in_sequences[ll] + 1]
        else:
            data = data_buf[-1]

        # Split data into multiple inputs if needed
        if operands[ll] > 1:
            if ll == 0 and legacy_test:
                data = np.array(np.split(data, operands[ll], axis=0))
            else:
                d = np.empty((operands[ll],
                              data.shape[0], data.shape[1], data.shape[2] // operands[ll]),
                             dtype=np.int64)
                for i in range(operands[ll]):
                    d[i, :, :, :] = data[:, :, i::operands[ll]]
                data = d
        else:
            data = np.expand_dims(data, 0)

        show_data(
            ll,
            verbose,
            data.shape,
            data,
            debug=debug_computation,
            expand=in_expand[ll],
            expand_thresh=in_expand_thresh[ll],
            operation=operator[ll],
            operands=operands[ll],
        )

        in_chan = input_chan[ll]

        # Run in-flight element-wise operations first?
        if operands[ll] > 1 and not pool_first[ll]:
            data = np.expand_dims(run_eltwise(data, ll), 0)

        # Allow 1D <-> 2D and 2D W/L conversions
        if operator[ll] == op.CONV1D:
            assert input_dim[ll][1] == 1
            data = data.reshape(data.shape[0], data.shape[1], input_dim[ll][0])
        else:
            data = data.reshape(data.shape[0], data.shape[1], input_dim[ll][0], input_dim[ll][1])

        # In-flight pooling
        data, out_size = pooling_layer(
            ll,
            verbose,
            data[0].shape,
            pool[ll],
            pool_stride[ll],
            pool_average[ll],
            data,
            debug=debug_computation,
            expand=in_expand[ll],
            expand_thresh=in_expand_thresh[ll],
            operation=operator[ll],
            operands=data.shape[0],
            rounding=avg_pool_rounding,
            debug_data=None if not log_pooling else os.path.join(base_directory, test_name),
        )

        if operator[ll] == op.CONV1D:
            assert out_size[0] == in_chan \
                and out_size[1] == pooled_dim[ll][0] \
                and pooled_dim[ll][1] == 1
        else:
            assert out_size[0] == in_chan \
                and out_size[1] == pooled_dim[ll][0] \
                and out_size[2] == pooled_dim[ll][1]

        if operands[ll] > 1 and pool_first[ll]:
            data = run_eltwise(data, ll)
        else:
            data = np.squeeze(data, axis=0)

        # Convolution or passthrough
        if operator[ll] == op.CONV2D:
            if flatten[ll]:
                in_chan *= input_dim[ll][0] * input_dim[ll][1]
                data = data.reshape(in_chan, 1, 1)
                if verbose:
                    print(f"FLATTEN TO {in_chan}x1x1...\n")

            out_buf, out_size = conv2d_layer(
                ll,
                verbose,
                data.shape,
                kernel_size[ll],
                output_shift[ll],
                output_chan[ll],
                padding[ll],
                dilation[ll],
                stride[ll],
                activation[ll],
                kernel[ll].reshape(
                    output_chan[ll],
                    in_chan,
                    kernel_size[ll][0],
                    kernel_size[ll][1]
                ),
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                device=device,
                debug=debug_computation,
            )
        elif operator[ll] == op.CONVTRANSPOSE2D:
            out_buf, out_size = convtranspose2d_layer(
                ll,
                verbose,
                data.shape,
                kernel_size[ll],
                output_shift[ll],
                output_chan[ll],
                padding[ll],
                dilation[ll],
                stride[ll],
                [1, 1],  # output_padding
                activation[ll],
                kernel[ll].reshape(
                    output_chan[ll],
                    in_chan,
                    kernel_size[ll][0],
                    kernel_size[ll][1],
                ),
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                device=device,
                debug=debug_computation,
            )
        elif operator[ll] == op.CONV1D:
            out_buf, out_size = conv1d_layer(
                ll,
                verbose,
                data.shape,
                kernel_size[ll][0],
                output_shift[ll],
                output_chan[ll],
                padding[ll][0],
                dilation[ll][0],
                stride[ll][0],
                activation[ll],
                kernel[ll].reshape(
                    output_chan[ll],
                    input_chan[ll],
                    kernel_size[ll][0],
                ),
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                device=device,
                debug=debug_computation,
            )
        elif operator[ll] == op.NONE:  # '0'D (pooling only or passthrough)
            out_buf, out_size = passthrough_layer(
                ll,
                verbose,
                data.shape,
                data,
                device=device,
                debug=debug_computation,
            )
        else:
            eprint(f'Unknown operator `{op.string(operator[ll])}`.')
            sys.exit(1)

        assert out_size[0] == output_chan[ll] \
            and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]

        # Write .mem file for output or create the C cnn_check() function to verify the output
        out_map = [None] * tc.dev.C_GROUP_OFFS * tc.dev.P_NUMGROUPS
        if block_mode:
            if ll == layers-1:
                filename = output_filename + '.mem'  # Final output
            else:
                filename = f'{output_filename}-{ll}.mem'  # Intermediate output
            filemode = 'w'
        else:
            if ll == layers-1:
                filename = c_filename + ('_riscv' if riscv else '') + '.c'  # Final output
            else:
                filename = None  # Intermediate output - used for layer overwrite check
            filemode = 'a'

        try:
            if filename:
                memfile = open(os.path.join(base_directory, test_name, filename), mode=filemode)
            else:
                memfile = None
            apb.set_memfile(memfile)

            apb.output(f'// {test_name}\n// Expected output of layer {ll}\n')
            apb.verify_header()
            if ll == layers-1 and mlator and not mlator_noverify:
                apb.verify_unload(
                    ll,
                    in_map,
                    None,
                    out_buf,
                    output_processor_map[ll],
                    out_size,
                    out_offset[ll],
                    out_expand[ll],
                    out_expand_thresh[ll],
                    output_width[ll],
                    pool[ll],
                    pool_stride[ll],
                    overwrite_ok or streaming[ll],
                    no_error_stop,
                    mlator=False,
                )
            if log_intermediate:
                filename2 = f'{output_filename}-{ll}.mem'  # Intermediate output
                memfile2 = open(os.path.join(base_directory, test_name, filename2), mode='w')
                apb2 = apbaccess.apbwriter(
                    memfile2,
                    0,
                    block_level=True,
                    verify_writes=False,
                    no_error_stop=False,
                    weight_header=None,
                    sampledata_header=None,
                    embedded_code=False,
                    compact_weights=False,
                    compact_data=False,
                    write_zero_registers=True,
                    weight_filename=None,
                    sample_filename=None,
                    device=device,
                    verify_kernels=verify_kernels,
                    master=groups_used[0] if oneshot > 0 or stopstart else False,
                    riscv=None,
                    riscv_flash=False,
                    riscv_cache=False,
                    fast_fifo=False,
                    input_csv=False,
                    input_chan=input_chan[0],
                    sleep=False,
                    debug_mem=True,
                )
                apb2.verify_unload(
                    ll,
                    in_map,
                    out_map,
                    out_buf,
                    output_processor_map[ll],
                    out_size,
                    out_offset[ll],
                    out_expand[ll],
                    out_expand_thresh[ll],
                    output_width[ll],
                    pool[ll],
                    pool_stride[ll],
                    overwrite_ok or streaming[ll],
                    no_error_stop,
                    mlator=mlator if ll == layers-1 else False,
                )
            apb.verify_unload(
                ll,
                in_map,
                out_map,
                out_buf,
                output_processor_map[ll],
                out_size,
                out_offset[ll],
                out_expand[ll],
                out_expand_thresh[ll],
                output_width[ll],
                pool[ll],
                pool_stride[ll],
                overwrite_ok or streaming[ll],
                no_error_stop,
                mlator=mlator if ll == layers-1 else False,
                max_count=max_count,
            )
            apb.verify_footer()
        finally:
            if memfile:
                memfile.close()

        data_buf.append(out_buf.reshape(out_size))
        if streaming[ll]:
            # When streaming, the output should not overwrite the input of prior layers since
            # these layers are still needed.
            in_map = [a if a is not None else b for a, b, in zip(in_map, out_map)]
        else:
            in_map = out_map

        if debug_computation:
            compute.debug_close()

    data = data_buf[-1]

    if not block_mode:
        with open(os.path.join(base_directory, test_name, filename), mode=filemode) as memfile:
            apb.set_memfile(memfile)

            if fc_weights or softmax or unload:
                apb.unload(
                    output_processor_map[-1],
                    out_size,
                    out_offset[layers-1],
                    out_expand[-1],
                    out_expand_thresh[-1],
                    output_width[-1],
                    pool[-1],
                    pool_stride[-1],
                    mlator=mlator,
                )

            if fc_weights:
                data = data.flatten()

                out_buf, out_size = linear_layer(
                    verbose=verbose,
                    activation=None,
                    data=data,
                    weight=fc_weights[0],
                    bias=fc_bias[0],
                    debug=debug,
                )

                apb.fc_layer(
                    fc_weights[0],
                    fc_bias[0],
                    output_width=output_width[-1],
                )
                apb.fc_verify(out_buf)
            elif softmax:
                apb.fc_layer(
                    None,
                    None,
                    softmax_only=True,
                    output_width=output_width[-1],
                    num_classes=output_chan[-1],
                )

            apb.main(
                oneshot=layers - 1 if oneshot else 0,
                softmax=softmax,
                unload=unload,
                stopstart=stopstart,
                num_classes=output_chan[-1],
                output_width=output_width[-1],
                clock_trim=clock_trim,
                groups=list(set().union(groups_used)),
                boost=boost,
                forever=forever,
                mexpress=mexpress,
                fifo=fifo,
            )

    # Close header files
    if sampledata_header is not None:
        sampledata_header.close()
    if weight_header is not None:
        weight_header.close()

    # Create run_test.sv
    if not embedded_code and not block_mode:
        if not timeout:
            # If no timeout specified, calculate one based on reads/writes
            timeout = 10 * (apb.get_time() + rtlsim.GLOBAL_TIME_OFFSET)
            if zero_sram:
                timeout += 16
        rtlsim.create_runtest_sv(
            block_mode,
            base_directory,
            test_name,
            runtest_filename,
            input_filename,
            c_filename,
            timeout,
            riscv=riscv,
            input_csv=input_csv,
            input_period=input_csv_period,
            input_sync=input_sync,
        )
        assets.copy('assets', 'all', base_directory, test_name)
        if riscv_cache:
            assets.copy('assets', 'riscv-cache', base_directory, test_name)
        elif riscv_flash:
            assets.copy('assets', 'riscv-flash', base_directory, test_name)
        elif riscv:
            assets.copy('assets', 'riscv', base_directory, test_name)
    elif block_mode:
        assets.copy('assets', 'blocklevel', base_directory, test_name)
    elif embedded_code:
        if riscv:
            assets.copy('assets', 'embedded-riscv-ai' + str(device), base_directory, test_name)
        else:
            assets.copy('assets', 'embedded-ai' + str(device), base_directory, test_name)

    return test_name


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=np.inf, linewidth=190)

    args = commandline.get_parser()

    if args.device != 84 and args.fc_layer:
        print("WARNING: --fc-layer should only be used on AI84")

    # Configure device
    tc.dev = tc.get_device(args.device)

    if args.device == 87:
        args.device = 85  # For now, there are no differences between AI85 and AI87

    if args.apb_base:
        apb_base = args.apb_base
    else:
        apb_base = tc.dev.APB_BASE
    if args.max_proc:
        tc.dev.MAX_PROC = args.max_proc
        tc.dev.P_NUMPRO = args.max_proc
        tc.dev.P_NUMGROUPS = 1
    if args.ready_sel:
        tc.dev.READY_SEL = args.ready_sel
    if args.ready_sel_fifo:
        tc.dev.FIFO_READY_SEL = args.ready_sel_fifo
    if args.ready_sel_aon:
        tc.dev.AON_READY_SEL = args.ready_sel_aon

    # Load configuration file
    cfg, params = yamlcfg.parse(args.config_file, args.device)

    # If not using test data, load weights and biases
    # This also configures the network's output channels
    if cfg['arch'] != 'test':
        if not args.checkpoint_file:
            eprint("--checkpoint-file is a required argument.")
            sys.exit(1)
        fext = args.checkpoint_file.rsplit(sep='.', maxsplit=1)[1].lower()
        if fext == 'onnx':
            # ONNX file selected
            layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels = \
                onnxcp.load(
                    args.checkpoint_file,
                    cfg['arch'],
                    args.fc_layer,
                    params['quantization'],
                    params['bias_quantization'],
                    params['kernel_size'],
                    params['operator'],
                    args.display_checkpoint,
                    args.no_bias,
                )
        else:
            # PyTorch checkpoint file selected
            layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels = \
                checkpoint.load(
                    args.checkpoint_file,
                    cfg['arch'],
                    args.fc_layer,
                    params['quantization'],
                    params['bias_quantization'],
                    params['kernel_size'],
                    params['operator'],
                    args.display_checkpoint,
                    args.no_bias,
                )
    else:  # Get some hard-coded sample weights
        layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels = \
            sampleweight.load(
                cfg['dataset'],
                params['quantization'],
                params['bias_quantization'],
                len(cfg['layers']),
                cfg['weights'] if 'weights' in cfg else None,
                cfg['bias'] if 'bias' in cfg else None,
                args.no_bias,
            )

    cfg_layers = len(cfg['layers'])
    if cfg_layers > layers:
        # Add empty weights/biases and channel counts for layers not in checkpoint file.
        # The checkpoint file does not contain weights for non-convolution operations.
        # Insert empty input channels/output channels/weights/biases and increase `layers`
        # accordingly.
        for ll in range(cfg_layers):
            operator = params['operator'][ll]

            if operator == op.NONE or op.eltwise(operator):
                weights.insert(ll, None)
                bias.insert(ll, None)
                input_channels.insert(ll, 0)
                output_channels.insert(ll, 0)
                layers += 1

    if layers != cfg_layers:
        eprint(f"Number of layers in the YAML configuration file ({cfg_layers}) "
               f"does not match the checkpoint file ({layers}).")
        sys.exit(1)

    if any(p < 0 or p > 4*tc.dev.MEM_SIZE for p in params['output_offset']):
        eprint('Unsupported value for `out_offset` in YAML configuration.')
        sys.exit(1)

    if args.device == 84:
        if any(q != 8 for q in params['quantization']):
            eprint('All quantization configuration values must be 8 for AI84.')
            sys.exit(1)

        if any(p0 != 1 and p0 & 1 != 0 or p0 < 0 or p0 > 4 or p1 != 1 and p1 & 1 != 0
               or p1 < 0 or p1 > 4 for [p0, p1] in params['pool']):
            eprint('Unsupported value for `max_pool`/`avg_pool` for AI84 in YAML configuration.')
            sys.exit(1)

        if any(p0 == 3 or p0 < 0 or p0 > 4
               or p1 == 3 or p1 < 0 or p1 > 4 for [p0, p1] in params['pool_stride']):
            eprint('Unsupported value for `pool_stride` in YAML configuration.')
            sys.exit(1)

    if any(q != 8 for q in params['bias_quantization']):
        eprint('All bias quantization configuration values must be 8.')
        sys.exit(1)

    if args.stop_after is not None:
        layers = args.stop_after + 1

    in_sequences = params['in_sequences'][:layers]

    # Override channels
    for ll in range(layers):
        if in_sequences[ll] is not None:
            if isinstance(in_sequences[ll], list):
                input_channels[ll] = sum(output_channels[i] for i in in_sequences[ll])
            else:
                input_channels[ll] = output_channels[in_sequences[ll]]
        if input_channels[ll] <= 0:
            input_channels[ll] = output_channels[ll-1]
        if params['input_chan'][ll] is not None:
            input_channels[ll] = params['input_chan'][ll]
        if output_channels[ll] <= 0:
            output_channels[ll] = input_channels[ll]
        if params['output_chan'][ll] is not None:
            output_channels[ll] = params['output_chan'][ll]

    processor_map = params['processor_map']
    output_processor_map = params['output_processor_map'][:layers]

    if 'output_map' in cfg:
        # Use optional configuration value if it's specified
        output_processor_map[-1] = cfg['output_map']
    elif len(processor_map) == layers and output_processor_map[-1] is None:
        # Default to packed, 0-aligned output map
        expand = (output_channels[layers-1] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
        expand_chunk = (output_channels[layers-1] + expand-1) // expand
        if output_channels[layers-1] > tc.dev.MAX_PROC:
            expand_chunk = min((expand_chunk + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                               tc.dev.MAX_PROC)
        output_processor_map[-1] = 2**expand_chunk-1

    # Remove extraneous layer configuration values (when --stop-after is used)
    processor_map = processor_map[:layers]

    input_channels = input_channels[:layers]
    output_channels = output_channels[:layers]
    output_offset = params['output_offset'][:layers]
    conf_input_dim = params['input_dim'][:layers]
    input_offset = params['input_offset'][:layers]
    kernel_size = params['kernel_size'][:layers]
    quantization = params['quantization'][:layers]
    output_shift = params['output_shift'][:layers]
    pool = params['pool'][:layers]
    pool_stride = params['pool_stride'][:layers]
    padding = params['padding'][:layers]
    stride = params['stride'][:layers]
    dilation = params['dilation'][:layers]
    big_data = params['big_data'][:layers]
    output_width = params['output_width'][:layers]
    operator = params['operator'][:layers]
    if args.ignore_streaming:
        streaming = [False] * layers
    else:
        streaming = params['streaming'][:layers]
    flatten = params['flatten'][:layers]
    operands = params['operands'][:layers]
    eltwise = params['eltwise'][:layers]
    pool_first = params['pool_first'][:layers]
    activation = params['activation'][:layers]
    conv_groups = params['conv_groups'][:layers]

    # Command line override
    if args.input_offset is not None:
        input_offset[0] = args.input_offset

    # Derived configuration options
    pool_average = [bool(x) for x in params['average']]

    print(f"Configuring data set: {cfg['dataset']}.")
    if args.sample_input is None:
        sampledata_file = os.path.join('tests', f'sample_{cfg["dataset"].lower()}.npy')
    else:
        sampledata_file = args.sample_input
    data = sampledata.get(sampledata_file)
    if np.max(data) > 127 or np.min(data) < -128:
        raise ValueError(f'Input data {sampledata_file} contains values that exceed 8-bit!')
    # Work with 1D input data
    if len(data.shape) < 3:
        data = np.expand_dims(data, axis=2)

    input_size = list(data.shape)

    if args.input_csv_format == 555:
        assert input_size[0] == 3
        data = data & ~0x7
    elif args.input_csv_format == 565:
        assert input_size[0] == 3
        data[0] = data[0] & ~0x7
        data[1] = data[1] & ~0x3
        data[2] = data[2] & ~0x7

    # Trace output sizes of the network
    auto_input_dim = [None] * layers
    input_dim = [None] * layers
    pooled_dim = [None] * layers
    output_dim = [None] * layers

    auto_input_dim[0] = [input_size[1], input_size[2]]
    if conf_input_dim[0] is None:
        input_dim[0] = auto_input_dim[0]
    else:
        input_dim[0] = conf_input_dim[0]
    for ll in range(layers):
        if input_channels[ll] <= 0:
            eprint(f'Must specify `in_channels` for layer {ll}.')
            sys.exit(1)
        if operator[ll] != op.NONE:
            assert weights[ll].min() >= -1 << quantization[ll] - 1
            assert weights[ll].max() <= (1 << quantization[ll] - 1) - 1

        if input_dim[ll] is None:
            if in_sequences[ll] is not None:
                if isinstance(in_sequences[ll], list):
                    dim = output_dim[in_sequences[ll][0]]
                    for _, e in enumerate(in_sequences[ll], start=1):
                        if output_dim[e] != dim:
                            eprint('Cannot concatenate outputs of different dimensions in layer '
                                   f'{ll}: {dim} vs {output_dim[e]}.')
                            sys.exit(1)
                    auto_input_dim[ll] = dim
                else:
                    auto_input_dim[ll] = output_dim[in_sequences[ll]]
            else:
                auto_input_dim[ll] = output_dim[ll-1]
            if conf_input_dim[ll] is None:
                input_dim[ll] = auto_input_dim[ll]
            else:
                input_dim[ll] = conf_input_dim[ll]
        if pool[ll][0] > 1 or pool[ll][1] > 1:
            if operator[ll] != op.CONV1D:
                if pool_stride[ll][0] != pool_stride[ll][1]:
                    eprint(f'{op.string(operator[ll])} in layer {ll} does not support non-square'
                           f'pooling stride (currently set to '
                           f'{pool_stride[ll][0]}x{pool_stride[ll][1]}).')
                    sys.exit(1)
                pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0])
                               // pool_stride[ll][0],
                               (input_dim[ll][1] + pool_stride[ll][1] - pool[ll][1])
                               // pool_stride[ll][1]]
            else:
                pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0])
                               // pool_stride[ll][0],
                               1]
        else:
            pooled_size = input_dim[ll]

        pooled_dim[ll] = pooled_size
        if any(dim == 0 for dim in pooled_dim[ll]):
            eprint(f'Pooling in layer {ll} results in a zero data dimension '
                   f'(input {input_dim[ll]}, pooled {pooled_dim[ll]}).')
            sys.exit(1)

        if operator[ll] != op.CONV1D:
            if stride[ll][0] != stride[ll][1]:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support non-square '
                       f'stride (currently set to {stride[ll][0]}x{stride[ll][1]}).')
                sys.exit(1)
            if operator[ll] != op.CONVTRANSPOSE2D and stride[ll][0] != 1:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support stride other '
                       f'than 1 (currently set to {stride[ll][0]}x{stride[ll][1]}).')
                sys.exit(1)
            if operator[ll] in [op.NONE, op.CONV2D]:
                output_dim[ll] = [(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1)
                                   - 1 + 2 * padding[ll][0]) // stride[ll][0] + 1,
                                  (pooled_size[1] - dilation[ll][1] * (kernel_size[ll][1] - 1)
                                   - 1 + 2 * padding[ll][1]) // stride[ll][1] + 1]
            elif operator[ll] == op.CONVTRANSPOSE2D:
                # output padding is always 1
                output_padding = 1
                output_dim[ll] = [(pooled_size[0] - 1) * stride[ll][0] - 2 * padding[ll][0]
                                  + dilation[ll][0] * (kernel_size[ll][0] - 1)
                                  + output_padding + 1,
                                  (pooled_size[1] - 1) * stride[ll][1] - 2 * padding[ll][1]
                                  + dilation[ll][1] * (kernel_size[ll][1] - 1)
                                  + output_padding + 1]
            else:  # Element-wise
                output_dim[ll] = [pooled_size[0], pooled_size[1]]
            if flatten[ll]:
                output_dim[ll] = [1, 1]
                input_channels[ll] //= input_dim[ll][0] * input_dim[ll][1]
            if padding[ll][0] >= 3:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                       f'(currently set to {padding[ll][0]}).')
                sys.exit(1)
        else:
            # We don't have to consider padding for the width calculation,
            # since padding has to be a multiple of 3 and we check for that.
            if args.device == 84:
                if pooled_size[0] % 3 != 0:
                    eprint(f'{op.string(operator[ll])} in layer {ll} requires a multiple of 3 for'
                           f'the pooled input length (currently {pooled_size[0]}).')
                    sys.exit(1)
                if padding[ll][0] % 3 != 0:
                    eprint(f'{op.string(operator[ll])} in layer {ll} requires a multiple of 3 for '
                           f'`pad` (currently set to {padding[ll][0]}).')
                    sys.exit(1)
            else:
                if padding[ll][0] >= 3:
                    eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                           f'(currently set to {padding[ll][0]}).')
                    sys.exit(1)
            output_dim[ll] = [(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1) - 1 +
                               2 * padding[ll][0]) // stride[ll][0] + 1,
                              1]

        # Prohibit pad greater than or equal to kernel size
        if padding[ll][0] >= kernel_size[ll][0] or padding[ll][1] >= kernel_size[ll][1]:
            eprint(f'Pad size for layer {ll} exceeds kernel size.')
            sys.exit(1)

        # Check for max dimensions
        if any(dim > tc.dev.MAX_ROW_COL for dim in input_dim[ll]):
            eprint(f'Input dimension {input_dim[ll]} exceeds system maximum of '
                   f'{tc.dev.MAX_ROW_COL} in layer {ll}.')
            sys.exit(1)
        if any(dim > tc.dev.MAX_ROW_COL for dim in output_dim[ll]):
            eprint(f'Output dimension {output_dim[ll]} exceeds system maximum of '
                   f'{tc.dev.MAX_ROW_COL} in layer {ll}.')
            sys.exit(1)

    if args.riscv and not args.riscv_cache and args.embedded_code:
        eprint("Embedded code on RISC-V requires --riscv-cache.")
        sys.exit(1)

    if not args.cmsis_software_nn:
        tn = create_net(
            args.prefix,
            args.verbose,
            args.debug,
            args.debug_computation,
            args.no_error_stop,
            args.overwrite_ok,
            args.log,
            apb_base,
            layers,
            operator,
            input_dim,
            pooled_dim,
            output_dim,
            processor_map,
            output_processor_map,
            kernel_size,
            quantization,
            output_shift,
            input_channels,
            output_channels,
            conv_groups,
            output_width,
            padding,
            dilation,
            stride,
            pool,
            pool_stride,
            pool_average,
            activation,
            data,
            weights,
            bias,
            big_data,
            fc_weights,
            fc_bias,
            args.input_split,
            input_offset,
            output_offset,
            streaming,
            flatten,
            operands,
            eltwise,
            pool_first,
            in_sequences,
            args.input_filename,
            args.output_filename,
            args.c_filename,
            args.test_dir,
            args.runtest_filename,
            args.log_filename,
            args.zero_unused,
            args.timeout,
            not args.top_level,
            args.verify_writes,
            args.verify_kernels,
            args.embedded_code,
            args.compact_weights,
            args.compact_data,
            args.write_zero_registers,
            args.weight_filename,
            args.sample_filename,
            args.device,
            args.init_tram,
            args.avg_pool_rounding,
            args.fifo,
            args.fast_fifo,
            args.fast_fifo_quad,
            args.zero_sram,
            args.mlator,
            args.one_shot,
            args.stop_start,
            args.mexpress,
            args.riscv,
            args.riscv_exclusive,
            args.riscv_flash,
            args.riscv_cache,
            args.riscv_debug,
            args.riscv_debugwait,
            args.override_start,
            args.increase_start,
            args.override_rollover,
            args.override_delta1,
            args.increase_delta1,
            args.override_delta2,
            args.increase_delta2,
            args.slow_load,
            args.synthesize_input,
            args.mlator_noverify,
            args.input_csv,
            args.input_csv_period,
            args.input_csv_format,
            args.input_csv_retrace,
            args.input_fifo,
            args.input_sync,
            args.deepsleep,
            args.powerdown,
            args.simple1b,
            args.legacy_test,
            args.log_intermediate,
            args.log_pooling,
            args.allow_streaming,
            args.softmax,
            args.unload,
            args.clock_trim,
            args.repeat_layers,
            args.fixed_input,
            args.max_count,
            args.boost,
            args.forever,
        )
        if not args.embedded_code and args.autogen.lower() != 'none':
            rtlsim.append_regression(
                args.top_level,
                tn,
                args.queue_name,
                args.autogen,
            )
    else:
        eprint('--cmsis-software-nn is not supported.', error=False)

        cmsisnn.create_net(
            args.prefix,
            args.verbose,
            args.debug,
            args.log,
            layers,
            operator,
            auto_input_dim,
            input_dim,
            pooled_dim,
            output_dim,
            kernel_size,
            quantization,
            output_shift,
            input_channels,
            output_channels,
            output_width,
            padding,
            dilation,
            stride,
            pool,
            pool_stride,
            pool_average,
            activation,
            data,
            weights,
            bias,
            fc_weights,
            fc_bias,
            flatten,
            args.c_filename,
            args.test_dir,
            args.log_filename,
            args.weight_filename,
            args.sample_filename,
            args.device,
        )

    print("SUMMARY OF OPS")
    stats.print_summary(factor=args.repeat_layers, debug=args.debug)


def signal_handler(
        _signal,
        _frame,
):
    """
    Ctrl+C handler
    """
    sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
