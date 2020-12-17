#!/usr/bin/env python3
###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
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
import devices
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
from eprint import eprint, wprint
from simulate import (conv1d_layer, conv2d_layer, convtranspose2d_layer, eltwise_layer,
                      linear_layer, passthrough_layer, pooling_layer, print_data, show_data)
from utils import ffs, fls, popcount


def create_net(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
        prefix,
        verbose,
        verbose_all,
        debug,
        debug_computation,
        debug_latency,  # pylint: disable=unused-argument
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
        input_skip,
        input_channel_skip,
        input_filename,
        output_filename,
        c_filename,
        api_filename,
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
        device=85,
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
        debugwait=1,
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
        legacy_kernels=False,
        log_intermediate=False,
        log_pooling=False,
        allow_streaming=False,
        softmax=False,
        clock_trim=None,
        repeat_layers=1,
        fixed_input=False,
        max_count=None,
        boost=None,
        forever=False,
        write_gap=None,
        start_layer=0,
        pipeline=False,
        pll=False,
        reshape_inputs=False,
        link_layer=False,
        measure_energy=False,
        timer=None,
        board_name='',
        rd_ahead=False,
        calcx4=False,
        rtl_preload=False,
        result_output=False,
        weight_start=0,
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

    if start_layer > tc.dev.MAX_START_LAYER:
        eprint(f"--start-layer is set to {start_layer}, but the device only supports "
               f"a maximum of {tc.dev.MAX_START_LAYER}.")

    if link_layer and not hasattr(tc.dev, 'LREG_NXTLYR'):
        eprint("--link-layer is not supported on this device.")

    if rd_ahead and not hasattr(tc.dev, 'RD_AHEAD_OFFS'):
        eprint("--read-ahead is not supported on this device.")

    if calcx4 and not tc.dev.SUPPORT_CALCX4:
        eprint("--calcx4 is not supported on this device.")

    if pipeline and not tc.dev.SUPPORT_PIPELINE:
        eprint("--pipeline is not supported on this device.")

    if pll and not tc.dev.SUPPORT_PLL:
        eprint("--pll is not supported on this device.")

    if pipeline is None:
        pipeline = tc.dev.SUPPORT_PIPELINE

    if pll is None:
        pll = pipeline

    if riscv_debug:
        riscv = True
    if riscv_cache:
        riscv = True
        riscv_flash = True
    if riscv_flash or riscv_exclusive:
        riscv = True

    if result_output and mlator:
        result_output = False

    if result_output:
        max_count = None

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
        if big_data[0] and processor_map[0] & ~0x0001000100010001 != 0 \
           or not big_data[0] and processor_map[0] & ~0x000f000f000f000f != 0:
            eprint("The FIFO is restricted to processors 0, 16, 32, 48 (CHW) or "
                   "0-3, 16-19, 32-35, 48-51 (HWC).")
        if fast_fifo:
            if big_data[0] and input_chan[0] > 1:
                eprint("Fast FIFO supports only a single CHW input channel; "
                       f"this test is using {input_chan[0]} channels.")
            elif not big_data[0] and input_chan[0] > 4:
                eprint("Fast FIFO supports up to four HWC input channels; "
                       f"this test is using {input_chan[0]} channels.")
            if processor_map[0] != 1 and processor_map[0] & 0x0e == 0:
                fifo_group = False
            if output_width[0] != 8:
                eprint('Single-layer fast FIFO setup requires output width of 8.')
            if operator[0] == op.NONE:
                eprint('Fast FIFO requies a convolution operation in the first layer.')
    elif streaming[0] and not allow_streaming:
        eprint('Streaming in the first layer requires use of a FIFO.')

    if mlator and (output_dim[-1][0] * output_dim[-1][1] < 4 or output_width[-1] > 8):
        wprint('--mlator should only be used with 4 or more 8-bit outputs per channel; ignoring.')
        mlator = False

    if fast_fifo and not riscv:
        eprint('--fast-fifo requires --riscv')

    if sleep and not riscv:
        eprint('--deepsleep requires --riscv')

    if oneshot and timer is not None:
        eprint('--timer is not supported when using --one-shot')

    processor_map_0 = processor_map[0]
    if fast_fifo_quad:
        processor_map[0] = processor_map_0 << 48 | processor_map_0 << 32 \
            | processor_map_0 << 16 | processor_map_0

    # Check that input channels are in separate memory instances if CHW (big) data format is used,
    # and calculate input and output expansion
    for ll in range(layers):
        if quantization[ll] is None:
            quantization[ll] = 8  # Set default
        if output_shift[ll] is None:
            output_shift[ll] = 0  # Set default

        if output_shift[ll] < -15 or output_shift[ll] > 15:
            implicit_shift = 8 - quantization[ll]
            eprint(f"Layer {ll} with {quantization[ll]}-bit weight quantization supports an "
                   f"output_shift range of [{-15 - implicit_shift}, +{15 - implicit_shift}]. "
                   f"The specified value of output_shift is {output_shift[ll] - implicit_shift} "
                   "which exceeds the system limits.")

        if big_data[ll]:
            p = processor_map[ll] >> (ffs(processor_map[ll]) & ~(tc.dev.P_SHARED-1))
            while p:
                if popcount(p & (tc.dev.P_SHARED-1)) > 1:
                    eprint(f"Layer {ll} uses CHW input format, but multiple channels "
                           "share the same memory instance. Modify the processor map for "
                           f"layer {ll}.")
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
        if not streaming[ll] and in_size + in_offset[ll] > tc.dev.INSTANCE_WIDTH*16:
            eprint(f'Layer {ll}: {1 if big_data[ll] else 4}-channel {input_dim[ll][0]}x'
                   f'{input_dim[ll][1]} input (size {in_size}) '
                   f'with input offset 0x{in_offset[ll]:04x} and expansion {in_expand[ll]}x '
                   f'exceeds data memory instance size of {tc.dev.INSTANCE_WIDTH*16}.')
        out_size = output_dim[ll][0] * output_dim[ll][1] * out_expand[ll] \
            * 4 * output_width[ll] // 8
        if (not streaming[ll] or ll == layers - 1) \
           and out_size + out_offset[ll] > tc.dev.INSTANCE_WIDTH*16:
            eprint(f'Layer {ll}: 4-channel, {output_width[ll]}-bit {output_dim[ll][0]}x'
                   f'{output_dim[ll][1]} output (size {out_size}) '
                   f'with output offset 0x{out_offset[ll]:04x} and expansion {out_expand[ll]}x '
                   f'exceeds data memory instance size of {tc.dev.INSTANCE_WIDTH*16}.')

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
            eprint(f'Layer {ll}: convolution groups ({conv_groups[ll]}) does not divide'
                   f' the input channels ({input_chan[ll]}) or'
                   f' output channels ({output_chan[ll]}).')

        if flatten[ll] and operator[ll] == op.NONE:
            eprint(f'Layer {ll}: `flatten` is not compatible with passthrough layers.')

        if flatten[ll] and (pool[ll][0] > 1 or pool[ll][1] > 1):
            eprint(f'Layer {ll}: `flatten` is not compatible with pooling.')

        if conv_groups[ll] > 1:
            if not tc.dev.SUPPORT_DEPTHWISE:
                eprint(f'Layer {ll}: convolution groups ({conv_groups[ll]}) > 1 are not supported'
                       f' on this device.')
            if conv_groups[ll] != input_chan[ll] or conv_groups[ll] != output_chan[ll]:
                eprint(f'Layer {ll}: convolution groups ({conv_groups[ll]}) must be equal to the'
                       f' number of input channels ({input_chan[ll]}), and output '
                       f' channels ({output_chan[ll]}) must be equal to input channels.')

        if input_skip[ll] != 0 and not hasattr(tc.dev, 'MP_STRIDE_OFFS'):
            eprint(f'Layer {ll}: `in_skip` must be 0 for this device.')

    # Create comment of the form "k1_b0-1x32x32b_2x2s2p14-..."
    test_name = prefix
    if not embedded_code:
        for ll in range(start_layer, layers):
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
        print(f'{" ".join(str(x) for x in sys.argv)}')
        print(f'{devices.partnum(device)}\n')
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
        if output_chan[ll] > tc.dev.MAX_CHANNELS:
            eprint(f'Layer {ll} is configured for {output_chan[ll]} outputs, which exceeds '
                   f'the system maximum of {tc.dev.MAX_CHANNELS}.')
        if (ll != 0 or not fast_fifo_quad) \
           and popcount(processor_map[ll]) != in_expand_thresh[ll]:
            eprint(f'Layer {ll} has {input_chan[ll]} inputs with input expansion '
                   f'{in_expand[ll]}, {operands[ll]} operands, threshold {in_expand_thresh[ll]}, '
                   f'but enabled processor map 0x{processor_map[ll]:016x} '
                   f'has {popcount(processor_map[ll])} bits instead of the '
                   f'expected number of {in_expand_thresh[ll]}.')
        if ll == 0 and fast_fifo_quad and popcount(processor_map_0) != in_expand_thresh[ll]:
            eprint(f'Layer {ll} has {input_chan[ll]} inputs with input expansion '
                   f'{in_expand[ll]}, threshold {in_expand_thresh[ll]}, but '
                   f'enabled processor map 0x{processor_map[ll]:016x} '
                   f'has {popcount(processor_map[ll])} bits instead of the '
                   f'expected number of {in_expand_thresh[ll]}.')
        if popcount(output_processor_map[ll]) != out_expand_thresh[ll]:
            eprint(f'Layer {ll} has {output_chan[ll]} outputs with output expansion '
                   f'{out_expand[ll]}, threshold {out_expand_thresh[ll]}, but '
                   f'processor output map 0x{output_processor_map[ll]:016x} '
                   f'has {popcount(output_processor_map[ll])} bits instead of the '
                   f'expected number of {out_expand_thresh[ll]}.')
        this_map = []
        for group in range(tc.dev.P_NUMGROUPS):
            if (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
                this_map.append(group)
        group_map.append(this_map)

        # Ensure input and output map are the same for passthrough layers
        if operator[ll] == op.NONE:
            for group in range(tc.dev.P_NUMGROUPS):
                in_pro = 2**popcount(
                    (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                ) - 1
                out_pro = (output_processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                if out_pro != 0:
                    out_pro >>= ffs(out_pro)
                if out_pro != in_pro:
                    eprint(f'Layer {ll} is a pass-through layer. The output processors must be a '
                           'packed version of the input processors for each x16. Configured are: '
                           f'input {processor_map[ll]:08x}, output '
                           f'{output_processor_map[ll]:08x}.')

    groups_used = []
    for group in range(tc.dev.P_NUMGROUPS):
        if ((processors_used |
             output_processor_map[-1]) >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
            groups_used.append(group)

    if 0 not in groups_used:
        eprint('Group 0 is not used, this currently does not work.')

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
                mexpress=mexpress,
                mem_output=rtl_preload,
                mem_output_final=result_output,
                debugwait=debugwait,
                embedded_arm=embedded_code,
                fail_indicator=forever,
                groups=list(set().union(groups_used)),
                clock_trim=clock_trim,
                pll=pll,
            )
            apb.copyright_header()

            apb.output(f'// ARM wrapper code\n// {test_name}\n')
            apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n\n')

            apb.header()
            apb.main()

    if input_csv is not None:
        csv = os.path.join(base_directory, test_name, input_csv)
    else:
        csv = None

    if embedded_code and api_filename.lower() != 'none':
        apifile = open(os.path.join(base_directory, test_name, api_filename), mode='w')
    else:
        apifile = None

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
            master=groups_used[0] if oneshot > 0 or stopstart or (apifile is not None) else False,
            riscv=True if riscv else None,
            riscv_flash=riscv_flash,
            riscv_cache=riscv_cache,
            riscv_debug=riscv_debug,
            debugwait=debugwait,
            fast_fifo=fast_fifo,
            input_csv=input_csv,
            input_csv_format=input_csv_format,
            input_chan=input_chan[0],
            sleep=sleep,
            mexpress=mexpress,
            mem_output=rtl_preload,
            mem_output_final=result_output,
            apifile=apifile,
            measure_energy=measure_energy,
            timer=timer,
            pll=pll,
            boost=boost,
            forever=forever,
            fifo=fifo,
            fail_indicator=forever,
            groups=list(set().union(groups_used)),
            clock_trim=clock_trim,
            oneshot=layers - 1 if oneshot else 0,
            softmax=softmax,
            stopstart=stopstart,
            num_classes=output_chan[-1],
            output_width=output_width[-1],
            bias=any(b is not None for b in bias),
        )

        apb.copyright_header()

        apb.output(f'// {test_name}\n')
        apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n\n')
        if apifile is not None:
            apb.output(f'// {test_name}\n', True)
            apb.output(f'// Created using {" ".join(str(x) for x in sys.argv)}\n\n', True)
            apb.output('// DO NOT EDIT - regenerate this file instead!\n\n', True)

        # Human readable description of test
        apb.output(f'// Configuring {repeat_layers * layers} '
                   f'layer{"s" if repeat_layers * layers > 1 else ""}:\n', embedded_code)

        for r in range(repeat_layers):
            for ll in range(start_layer, layers):
                apb.output(f'// Layer {r * layers + ll}: '
                           f'{str(operands[ll])+"x" if operands[ll] > 1 else ""}'
                           f'{input_chan[ll]}x{input_dim_str[ll]} ('
                           f'{"streaming " if streaming[ll] else ""}'
                           f'{"flattened " if flatten[ll] else ""}'
                           f'{"CHW data)" if big_data[ll] else "HWC data)"}, ',
                           embedded_code)
                if pool[ll][0] > 1 or pool[ll][1] > 1:
                    apb.output(f'{pool_str[ll]} {"avg" if pool_average[ll] else "max"} '
                               f'pool with stride {pool_stride_str[ll]}', embedded_code)
                else:
                    apb.output('no pooling', embedded_code)
                if operator[ll] != op.NONE:
                    conv_str = f', {op.string(operator[ll])} with kernel size ' \
                               f'{kernel_size_str[ll]}, ' \
                               f'stride {stride_str[ll]}, ' \
                               f'pad {padding_str[ll]}, '
                else:
                    conv_str = ', no convolution, '
                apb.output(conv_str +
                           f'{output_chan[ll]}x{output_dim_str[ll]} output\n', embedded_code)

        apb.output('\n', embedded_code)

        apb.header()

        if embedded_code or compact_data or mexpress:
            apb.function_header(prefix='', function='memcpy32', return_type='void',
                                arguments='uint32_t *dst, const uint32_t *src, int n')
            apb.output('  while (n-- > 0) {\n'
                       '    *dst++ = *src++;\n'
                       '  }\n', embedded_code)
            apb.function_footer(return_value='void')  # memcpy32()

        if input_fifo:
            apb.output('#define USE_FIFO\n')

        if embedded_code or compact_data or input_csv:
            # Pre-define data memory loader. Inline later when generating RTL sim.
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
            kern_offs, kern_len, kern_count = kernels.load(
                verbose,
                True,
                device,
                apb,
                0,
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
                legacy_kernels=legacy_kernels,
                calcx4=calcx4,
                api=embedded_code,
                start_offs=weight_start,
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

        apb.function_header(function='init')

        # Initialize CNN registers

        if verbose:
            # startup, lat = stats.calc_latency(
            #     streaming,
            #     layers,
            #     eltwise,
            #     pool,
            #     pooled_dim,
            #     in_expand,
            #     output_chan,
            #     output_dim,
            #     input_dim,
            #     padding,
            #     kernel_size,
            # )
            # print('\nEstimated latency:')
            # print('------------------')
            # if lat is None:
            #     print('N/A')
            # else:
            #     total = startup
            #     print(f'Startup{startup:14,}')
            #     for k in range(start_layer, layers):
            #         total += lat[k][0]
            #         print(f'Layer {k:<3}{lat[k][0]:12,}', end='')
            #         if debug_latency:
            #             print('', lat[k][1])
            #         else:
            #             print('')
            #     print('           ==========')
            #     print(f'Total{total:16,} cycles')

            print('\nGlobal registers:')
            print('-----------------')

        if tc.dev.REQUIRE_REG_CLEAR:
            for _, group in enumerate(groups_used):
                apb.write_ctl(group, tc.dev.REG_CTL, 1 << 3 | tc.dev.READY_SEL << 1,
                              verbose, comment=' // Enable clocks', no_verify=True)
            for _, group in enumerate(groups_used):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 7,
                              verbose, comment=' // Clear registers', no_verify=True)
            for _, group in enumerate(groups_used):
                apb.wait_ctl(group, tc.dev.REG_SRAM_TEST, 1 << 25, 1 << 25,
                             comment=' // Wait for clear')
            for _, group in enumerate(groups_used):
                apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                              verbose, comment=' // Reset BIST', force_write=True, no_verify=True)
            apb.output('\n', embedded_code)

        # Reset
        apb.write_fifo_ctl(tc.dev.AON_CTL, tc.dev.AON_READY_SEL,
                           verbose, comment=' // AON control', force_write=True)

        # Configure global control registers for used groups
        for _, group in enumerate(groups_used):
            if init_tram:
                # Zero out Tornado RAM
                if not embedded_code:
                    for p in range(tc.dev.P_NUMPRO):
                        for offs in range(tc.dev.TRAM_SIZE):
                            apb.write_tram(group, p, offs, 0, comment='Zero ')
                    apb.output('\n', embedded_code)
                else:
                    for p in range(tc.dev.P_NUMPRO):
                        addr = apb_base + tc.dev.C_GROUP_OFFS*group + tc.dev.C_TRAM_BASE \
                            + p * tc.dev.TRAM_OFFS * 4
                        apb.output(f'  memset((uint32_t *) 0x{addr:08x}, 0, '
                                   f'{tc.dev.TRAM_SIZE}); // Zero TRAM {group}\n', embedded_code)
                        apb.output('\n', embedded_code)

            # Stop state machine - will be overwritten later; enable FIFO
            val = tc.dev.READY_SEL << 1
            if fifo:
                val |= 1 << 15
            val |= 1 << 3  # Enable clocks
            if mexpress:
                val |= 1 << 20
            apb.write_ctl(group, tc.dev.REG_CTL, val,
                          verbose, comment=' // Stop SM')
            # SRAM Control - does not need to be changed
            apb.write_ctl(group, tc.dev.REG_SRAM, 0x40e,
                          verbose, comment=' // SRAM control')
            # Number of layers and start layer
            val = (repeat_layers * layers - 1) | (start_layer << 8)
            apb.write_ctl(group, tc.dev.REG_LCNT_MAX, val,
                          verbose, comment=' // Layer count')

        if zero_sram:
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
            apb.output('\n', embedded_code)
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
            apb.output('\n', embedded_code)
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
            apb.output('\n', embedded_code)
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
            apb.output('\n', embedded_code)

        apb.function_footer()

        if block_mode or not (embedded_code or mexpress or compact_weights):
            kern_offs, kern_len, kern_count = kernels.load(
                verbose,
                embedded_code,
                device,
                apb,
                0,
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
                legacy_kernels=legacy_kernels,
                calcx4=calcx4,
                start_offs=weight_start,
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

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors     = 0x{processors_used:016x}')
            print(f'Used groups         = {groups_used}')
            if start_layer > 0:
                print(f'Starting layer      = {start_layer}')

            print('\nPer-group configuration:')
            print('-----------------------')
            print(f'Used bias memory    = {group_bias_max}')

            print('\nPer-layer configuration:')
            print('------------------------')
            if repeat_layers > 1:
                print(f'Layer repeat count  = {repeat_layers}')
            print(f'Group map           = {group_map}')

            print('Input offset        = [',
                  ', '.join('0x{:04x}'.format(k) for k in in_offset), ']', sep='',)
            print(f'Streaming           = {streaming}')
            print(f'Input channels      = {input_chan}')
            print(f'Input dimensions    = {input_dim}')
            print(f'Flatten             = {flatten}')
            if any(s > 0 for s in input_skip):
                print(f'Input skip          = {input_skip}')
            if any(s > 0 for s in input_channel_skip):
                print(f'Input channel skip  = {input_channel_skip}')
            print(f'Input expansion     = {in_expand}')
            print(f'Expansion threshold = {in_expand_thresh}')

            print(f'Pooling             = {pool}')
            print(f'Pooling stride      = {pool_stride}')
            print(f'Pooled dimensions   = {pooled_dim}')

            print('Processor map       = [',
                  ', '.join('0x{:016x}'.format(k) for k in processor_map), ']', sep='',)

            print('Element-wise op     = [',
                  ', '.join(op.string(k, elt=True) for k in eltwise), ']', sep='',)
            print(f'Operand expansion   = {operands}')

            print(f'Output channels     = {output_chan}')
            print(f'Output dimensions   = {output_dim}')
            print(f'Output expansion    = {out_expand}')
            print(f'Expansion threshold = {out_expand_thresh}')
            print(f'Output shift        = {output_shift}')
            print('Output processors   = [',
                  ', '.join('0x{:016x}'.format(k) for k in output_processor_map), ']', sep='',)
            print(f'Output data bits    = {output_width}')

            print(f'Group with bias     = {bias_group}')
            print(f'Bias offset         = {bias_offs}')

            print('Output offset       = [',
                  ', '.join('0x{:04x}'.format(k) for k in out_offset), ']', sep='',)

            print('Operator            = [',
                  ', '.join(op.string(k) for k in operator), ']', sep='',)
            print(f'Kernel offset       = {kern_offs}')
            print(f'Kernel length       = {kern_len}')
            print(f'Kernel count        = {kern_count}')
            print(f'Kernel dimensions   = {kernel_size}')
            print(f'Kernel size (bits)  = {quantization}')
            print(f'Convolution groups  = {conv_groups}')
            print(f'Padding             = {padding}')
            print(f'Stride              = {stride}')
            print('')

        if verbose:
            print('Layer register configuration:')
            print('-----------------------------')

        apb.function_header(function='configure')

        # Configure per-layer control registers
        for r in range(repeat_layers):
            for ll in range(start_layer, layers):

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

                # For passthrough, determine time slot count (maximum across all used groups)
                tscnt_max = 0
                for _, group in enumerate(groups_used):
                    if operator[ll] == op.NONE:
                        if popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                    % 2**tc.dev.P_NUMPRO) != 0:
                            tscnt_max = max(
                                tscnt_max,
                                (popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                          % 2**tc.dev.P_NUMPRO) * output_width[ll] // 8 - 1) // 4
                            )
                    elif conv_groups[ll] > 1:
                        tscnt_max = max(
                            tscnt_max,
                            popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                     % 2**tc.dev.P_NUMPRO) - 1
                        )

                for _, group in enumerate(groups_used):
                    apb.output(f'  // Layer {r * layers + ll} group {group}\n', embedded_code)

                    if hasattr(tc.dev, 'LREG_NXTLYR'):
                        if link_layer:
                            if ll < layers-1:
                                val = 1 << 7 | (ll + 1)
                            else:
                                val = 1 << 8  # Stop
                        else:
                            val = 0
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_NXTLYR, val,
                                       verbose, comment=' // Next Layer')

                    # Configure row count
                    if flatten[ll]:
                        in_row = pool[ll][0]
                        in_col = pool[ll][1]
                    else:
                        if operator[ll] == op.CONVTRANSPOSE2D:
                            in_row = stride[ll][0] * input_dim[ll][0]
                            in_col = stride[ll][1] * input_dim[ll][1]
                        else:
                            in_row = input_dim[ll][0]
                            in_col = input_dim[ll][1]
                    if hasattr(tc.dev, 'CNT_DIFF_OFFS'):
                        diff = (in_row - ((in_row - pool[ll][0])
                                          // pool_stride[ll][0]) * pool_stride[ll][0])
                        val = in_row - diff  # Stop row, 0-based
                        assert val < 2**tc.dev.MAX_CNT_BITS

                        # Stop column
                        if operator[ll] == op.CONV1D:
                            diff = 1
                        else:
                            diff = (in_col - ((in_col - pool[ll][1])
                                              // pool_stride[ll][1]) * pool_stride[ll][1])
                        # Bytes to next starting element
                        diff = (diff + (pool_stride[ll][0] - 1) * in_col) \
                            * (input_skip[ll] + 1) * operands[ll] * in_expand[ll]

                        val |= diff << tc.dev.CNT_DIFF_OFFS
                        if padding[ll][0] > 0:
                            assert padding[ll][0] - 1 < 2**2
                            val |= 1 << tc.dev.PAD_ENA_OFFS
                            val |= padding[ll][0] - 1 << tc.dev.PAD_CNT_OFFS
                    else:
                        val = in_row - 1
                        assert padding[ll][0] < 2**2
                        assert val + 2*padding[ll][0] < 2**tc.dev.MAX_CNT_BITS
                        val |= padding[ll][0] << tc.dev.PAD_CNT_OFFS
                        val += 2*padding[ll][0]
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_RCNT, val,
                                   verbose, comment=' // Rows')

                    # Configure column count (evaluates to 0 for 1D convolutions)
                    if hasattr(tc.dev, 'CNT_DIFF_OFFS'):
                        # Calculate last pooling fetch before advancing to next row
                        diff = (in_col - ((in_col - pool[ll][1])
                                          // pool_stride[ll][1]) * pool_stride[ll][1])
                        val = in_col - diff
                        assert val < 2**tc.dev.MAX_CNT_BITS
                        val |= diff << tc.dev.CNT_DIFF_OFFS
                        if padding[ll][1] > 0:
                            assert padding[ll][1] - 1 < 2**2
                            val |= 1 << tc.dev.PAD_ENA_OFFS
                            val |= padding[ll][1] - 1 << tc.dev.PAD_CNT_OFFS
                    else:
                        val = in_col - 1
                        assert padding[ll][1] < 2**2
                        assert val + 2 * padding[ll][1] < 2**tc.dev.MAX_CNT_BITS
                        val |= padding[ll][1] << tc.dev.PAD_CNT_OFFS
                        val += 2 * padding[ll][1]
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_CCNT, val,
                                   verbose, comment=' // Columns')

                    # Configure pooling row count
                    val = pool[ll][0]-1
                    assert val < 2**4
                    if hasattr(tc.dev, 'CNT_INC_OFFS'):
                        val |= 0 << tc.dev.CNT_INC_OFFS  # FIXME
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_PRCNT, val,
                                   verbose, comment=' // Pooling rows')

                    # Configure pooling column count
                    val = pool[ll][1]-1
                    assert val < 2**4
                    if hasattr(tc.dev, 'CNT_INC_OFFS'):
                        val |= 0 << tc.dev.CNT_INC_OFFS  # FIXME
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_PCCNT, val,
                                   verbose, comment=' // Pooling columns')

                    # Configure pooling stride count
                    if operator[ll] == op.CONVTRANSPOSE2D:
                        val = 0
                    elif pool_stride[ll][0] > 1:
                        val = pool_stride[ll][0]-1
                    else:
                        val = stride[ll][0]-1
                    assert val < 2**4
                    if hasattr(tc.dev, 'MP_STRIDE_OFFS'):  # Multipass stride
                        val |= pool_stride[ll][0] * operands[ll] * in_expand[ll] \
                            * (input_skip[ll] + 1) << tc.dev.MP_STRIDE_OFFS
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
                            if output_processor_map[ll] & \
                               2**tc.dev.P_NUMPRO - 1 << group*tc.dev.P_NUMPRO > 0:
                                instance = ffs(output_processor_map[ll]
                                               & 2**tc.dev.P_NUMPRO - 1 << group*tc.dev.P_NUMPRO) \
                                    & ~(tc.dev.P_SHARED-1)
                            else:
                                instance = 0

                        val |= (instance % tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE \
                            | (instance // tc.dev.P_SHARED) << tc.dev.WRITE_PTR_SHIFT
                    else:
                        instance = ffs(output_processor_map[ll] >> group * tc.dev.P_SHARED) \
                               & ~(tc.dev.P_SHARED-1)
                        val |= (instance + group * tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE
                    assert val < 2**tc.dev.MAX_PTR_BITS
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_BASE, val,
                                   verbose, comment=' // SRAM write ptr')

                    # Write Pointer Timeslot Offset Register
                    # Used for 1x1 convolution, and pooling without convolution
                    if operator[ll] == op.CONV2D and kernel_size[ll] == [1, 1]:
                        val = 1 if conv_groups[ll] == 1 else 0
                    elif operator[ll] == op.NONE:
                        if popcount(processor_map[ll]) > 4 \
                           or operands[ll] > 1 and in_expand[ll] > 1:
                            val = tc.dev.INSTANCE_SIZE * 4
                        else:
                            val = tc.dev.INSTANCE_SIZE
                    else:
                        val = 0
                    assert val < 2**tc.dev.MAX_PTR_BITS
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_TOFFS, val,
                                   verbose, comment=' // Write ptr time slot offs')

                    # [15:0] Write Pointer Mask Offset Register
                    val = 1 << tc.dev.WRITE_PTR_SHIFT
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
                    val = (0x200 if activation[ll] == op.ACT_RELU else 0) | \
                          (0x100 if not pool_average[ll] else 0) | \
                          (0x80 if pool[ll][0] > 1 or pool[ll][1] > 1 else 0) | \
                          (0x40 if big_data[ll] else 0) | \
                          (0x20)
                    if not local_source:
                        val |= 0x800

                    if output_width[ll] != 8:
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

                    if rd_ahead and hasattr(tc.dev, 'RD_AHEAD_OFFS'):
                        val |= 1 << tc.dev.RD_AHEAD_OFFS

                    if hasattr(tc.dev, 'CPRIME_MAX_OFFS') and operator[ll] != op.NONE:
                        val |= kernel_size[ll][0] - 1 << tc.dev.RPRIME_MAX_OFFS
                        val |= kernel_size[ll][1] - 1 << tc.dev.CPRIME_MAX_OFFS

                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL, val,
                                   verbose, comment=' // Layer control')

                    flatten_prod = 0
                    if flatten[ll]:
                        # Store all bits, top programmed in post processing register
                        flatten_prod = \
                            in_expand[ll] * pooled_dim[ll][0] * pooled_dim[ll][1] - 1
                        in_exp = flatten_prod & 0x0f  # Lower 4 bits only
                    else:
                        in_exp = in_expand[ll] - 1

                    assert in_exp < 2**4  # Cannot have more than 4 bits

                    val = (fls(output_processor_map[ll])
                           - (ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1))) \
                        * quantization[ll] << 8 \
                        | in_exp
                    if operator[ll] != op.NONE:
                        wptr_skip = out_expand[ll] * (write_gap[ll] + 1)
                        assert wptr_skip <= 2**4  # Cannot have more than 4 bits (+1)
                        val |= (wptr_skip - 1) << 4
                    else:
                        assert write_gap[ll] + 1 <= 2**4  # Cannot have more than 4 bits (+1)
                        val |= write_gap[ll] << 4

                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL2, val,
                                   verbose, comment=' // Layer control 2')

                    # Configure mask start and end addresses
                    # Every mask memory starts from the same offset for all processors
                    oned_sad = 0
                    if operator[ll] != op.NONE:
                        kl = (kern_count[ll] - 1) * quantization[ll]
                        ochan = kern_count[ll] - 1

                        if ll == 0 and fast_fifo_quad or calcx4:
                            if calcx4:
                                kl += quantization[ll]
                            kl = (kl + 3) // 4  # FIXME: Handle fast_fifo_quad and calcx4
                            if calcx4:
                                kl -= quantization[ll]
                        koffs, oned_sad = divmod(9 * kern_offs[ll],
                                                 kernel_size[ll][0] * kernel_size[ll][1])
                        koffs *= 8

                    if hasattr(tc.dev, 'LREG_MCNT1'):
                        if operator[ll] != op.NONE:
                            assert koffs < 2**19
                            assert kl + koffs < 2**19
                            apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT1, kl + koffs,
                                           verbose, comment=' // Mask count')
                            apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT2, koffs,
                                           verbose, comment=' // Mask offset')
                        else:
                            val = (out_expand[ll] - 1) * 8
                            assert val < 2**19
                            apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT2, val,
                                           verbose, comment=' // Mask offset')
                    else:
                        if operator[ll] != op.NONE:
                            assert koffs < 2**16
                            assert kl + koffs < 2**16
                            # kern_offs is always bytes
                            val = \
                                koffs << tc.dev.MCNT_SAD_OFFS | kl + koffs << tc.dev.MCNT_MAX_OFFS
                        else:
                            val = (out_expand[ll] - 1) * 8
                            assert val < 2**16
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT, val,
                                       verbose, comment=' // Mask offset and count')

                    if hasattr(tc.dev, 'LREG_OCHAN'):
                        val = 0
                        if operator[ll] != op.NONE:
                            if calcx4:
                                ochan //= 4
                            if ochan > 0:
                                val = ochan - 1
                        elif tscnt_max > 0:
                            val = tscnt_max - 1
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_OCHAN, val,
                                       verbose, comment=' // Output channel count')

                    val = tscnt_max
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
                        if (pool[ll][0] > 1 or pool[ll][1] > 1) \
                           and pool_first[ll]:
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
                            assert prev_max < 2**tc.dev.MAX_TPTR_BITS
                            val += prev_max
                            assert val < 2**16
                            val |= prev_max << 16
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_TPTR, val,
                                   verbose, comment=' // TRAM ptr max')

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
                    assert operator[ll] != op.NONE or output_shift[ll] == 0
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

                    if conv_groups[ll] > 1:
                        val |= 1 << 30 | 1 << 24  # depthwise_ena, ts_ena

                    if calcx4:
                        val |= 1 << 29

                    if rd_ahead and in_expand[ll] > 1:
                        val |= 1 << 31  # tcalc

                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_POST, val,
                                   verbose, comment=' // Post processing register')

                    # Configure mask and processor enables
                    # Enable at most 16 processors and masks
                    val = (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                    if operator[ll] != op.NONE:
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
                        # Start: Prior layer's padded pooled row width * prior layer's kernel
                        # height + prior layer's kernel width + prior layer's pad
                        stream_start = (pooled_dim[ll-1][1] + 2 * padding[ll-1][1]) \
                            * (kernel_size[ll-1][0] - 1 + pool[ll][0] - 1) \
                            + kernel_size[ll-1][1] - 1 + pool[ll][1] + increase_start
                        assert stream_start < 2**tc.dev.MAX_ISVAL_BITS

                        # Delta 1: This layer's pooling stride
                        delta1 = pool_stride[ll][1] * operands[ll] + increase_delta1
                        assert delta1 < 2**5
                        # Delta 2: (This layer's pooling - 1) * full prior layer's padded rows +
                        # prior layer's pad
                        delta2 = (pool_stride[ll][0] - 1) \
                            * (pooled_dim[ll-1][1] + 2 * padding[ll-1][1]) \
                            + pool[ll][1] * operands[ll] + increase_delta2
                        assert delta2 < 2**tc.dev.MAX_DSVAL2_BITS

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_STREAM1, stream_start,
                                       verbose, comment=' // Stream processing start')
                        # strm_invol[3:0]   Per stream invol offset - based on stream count
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
                        assert val < 2**tc.dev.MAX_FBUF_BITS

                        # Check rollover vs available data memory
                        if in_offset[ll] < out_offset[ll]:
                            if in_offset[ll] + val * 4 >= out_offset[ll]:
                                eprint('Overlapping input and output: '
                                       f'in_offset 0x{in_offset[ll]:08x} < '
                                       f'out_offset 0x{out_offset[ll]:08x}, '
                                       f'rollover 0x{val:08x}.',
                                       error=not no_error_stop)
                        else:
                            if out_offset[ll] + val * 4 >= in_offset[ll]:
                                eprint('Overlapping input and output: '
                                       f'in_offset 0x{in_offset[ll]:08x} >= '
                                       f'out_offset 0x{out_offset[ll]:08x}, '
                                       f'rollover 0x{val:08x}.',
                                       error=not no_error_stop)
                        if in_offset[ll] + val * 4 >= tc.dev.INSTANCE_WIDTH * tc.dev.P_SHARED * 4:
                            eprint('Input plus rollover exceeds instance size: '
                                   f'in_offset 0x{in_offset[ll]:08x}, '
                                   f'out_offset 0x{out_offset[ll]:08x}, '
                                   f'rollover 0x{val:08x}, '
                                   f'instance size 0x{tc.dev.INSTANCE_WIDTH*4:08x}.',
                                   error=not no_error_stop)

                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_FMAX, val,
                                       verbose, comment=' // Rollover')

                    if ll == 0 and fifo:
                        val = input_dim[0][0] * input_dim[0][1]
                        if big_data[0]:
                            val = (val + 3) // 4
                        assert val < 2**tc.dev.MAX_IFRM_BITS
                        apb.write_ctl(group, tc.dev.REG_IFRM, val, verbose,
                                      comment=' // Input frame size')

                    apb.output('\n', embedded_code)  # End of group

        if zero_unused:
            for r in range(repeat_layers):
                for ll in range(start_layer, layers, tc.dev.MAX_LAYERS):
                    for _, group in enumerate(groups_used):
                        for reg in range(tc.dev.MAX_LREG+1):
                            if reg == tc.dev.LREG_RFU:  # Register 2 not implemented
                                continue
                            apb.write_lreg(group, r * layers + ll, reg, 0,
                                           verbose, force_write=True,
                                           comment=f' // Zero unused layer {ll} registers')
                if hasattr(tc.dev, 'MIN_STREAM_LREG'):
                    for ll in range(start_layer, layers, tc.dev.MAX_STREAM_LAYERS):
                        for _, group in enumerate(groups_used):
                            for reg in range(tc.dev.MIN_STREAM_LREG, tc.dev.MAX_STREAM_LREG+1,
                                             tc.dev.MAX_STREAM_LAYERS):
                                apb.write_lreg(group, r * layers + ll, reg, 0,
                                               verbose, force_write=True,
                                               comment=f' // Zero unused layer {ll} registers')

        if not fifo:
            # Load data memory
            if embedded_code or compact_data or input_csv:
                # Do the actual code generation later
                if not embedded_code:
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
            apb.output('\n', embedded_code)

            # FIFO control
            if not fast_fifo:
                val = 0x02 << 2 | 0x02 << 7 | 1 << 11 | tc.dev.FIFO_READY_SEL
                for i in range(input_chan[0]):
                    if processor_map_0 & 1 << (i % tc.dev.P_NUMGROUPS) * tc.dev.P_NUMPRO != 0:
                        val |= 1 << i % tc.dev.P_NUMGROUPS + 12
                apb.write_fifo_ctl(tc.dev.FIFO_CTL, val,
                                   verbose, comment=' // FIFO control')
            else:
                apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_IE, 0,
                                        verbose, comment=' // Fast FIFO interrupt enable')
                val = 10 << 4  # Async, threshold 10
                apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_CR, val,
                                        verbose, comment=' // Fast FIFO control')

        val = 1 << 14 if any(streaming) else 0
        if avg_pool_rounding:
            val |= 1 << 13
        if fifo:
            val |= 1 << 11
        if fifo and any(streaming):
            val |= 1 << 19
        val |= 1 << 3  # Enable clocks
        if mexpress:
            val |= 1 << 20
        if simple1b:
            val |= 1 << 21
        if fast_fifo_quad:
            val |= 1 << 31  # Qupac bit
        if oneshot:
            val |= 1 << 8
        if hasattr(tc.dev, 'CTL_PIPELINE_OFFS'):
            if not pipeline:
                val |= 1 << tc.dev.CTL_PIPELINE_OFFS
            if streaming[0] and big_data[0]:
                val |= 1 << 6

        if embedded_code:
            apb.function_footer()
            apb.function_header(function='start')

        apb.output('  cnn_time = 0;\n\n', embedded_code)

        # Enable all needed groups except the first one
        rdy_sel = tc.dev.READY_SEL if not pipeline else tc.dev.PIPELINE_READY_SEL
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
            apb.write_ctl(group, tc.dev.REG_CTL, val | 0x800 | rdy_sel << 1
                          | fval | groups_used[0] << 9,
                          verbose, comment=f' // Enable group {group}')

        if powerdown:
            unused_groups = [group for group in list(range(tc.dev.P_NUMGROUPS))
                             if group not in groups_used]
            val2 = 0
            for _, group in enumerate(unused_groups):
                val2 |= 1 << 12 + group
            apb.write_fifo_ctl(tc.dev.AON_CTL, val2 | tc.dev.AON_READY_SEL,
                               verbose, comment=' // AON control')

        if pll and not measure_energy:
            apb.select_clock('ITO', 'DIV1', 'Switch CNN clock to PLL (ITO)')
        if embedded_code:
            apb.output('\n#ifdef CNN_INFERENCE_TIMER\n'
                       '  MXC_TMR_SW_Start(CNN_INFERENCE_TIMER);\n'
                       '#endif\n\n', embedded_code)
            if not measure_energy:
                apb.output('  CNN_START; // Allow capture of processing time\n', embedded_code)

        # Master control - go
        if fifo and processor_map_0 & 0x0f << groups_used[0] * 16 != 0:
            val |= 1 << 15
            if fast_fifo:
                val |= 1 << 22
            if fifo_group:
                val |= 1 << 23
        apb.write_ctl(groups_used[0], tc.dev.REG_CTL, val | rdy_sel << 1 | 0x01,
                      verbose, comment=f' // Master enable group {groups_used[0]}')

        if fifo:
            # Load data memory
            if embedded_code or compact_data or input_csv:
                # Do the actual code generation later
                if not embedded_code:
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

        apb.function_footer()
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
            verbose_all or ll == layers-1,
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
    for ll in range(start_layer, layers):
        if debug_computation:
            compute.debug_open(ll, base_directory, test_name, log_filename)

        # Concatenate input data if needed
        if in_sequences[ll] is not None:
            if isinstance(in_sequences[ll], list):
                try:
                    data = np.concatenate([data_buf[i + 1] for i in in_sequences[ll]], axis=0)
                except ValueError as err:
                    eprint('Error in input data concatenation layer:', err)
            else:
                data = data_buf[in_sequences[ll] + 1]
        else:
            data = data_buf[-1]

        # Split data into multiple inputs if needed
        if operands[ll] > 1:
            if ll == 0 and legacy_test:
                data = np.array(np.split(data, operands[ll], axis=0))
            elif legacy_test:
                d = np.empty((operands[ll],
                              data.shape[0], data.shape[1], data.shape[2] // operands[ll]),
                             dtype=np.int64)
                for i in range(operands[ll]):
                    d[i, :, :, :] = data[:, :, i::operands[ll]]
                data = d
            else:
                data = np.array(np.split(data, operands[ll], axis=0))
        else:
            data = np.expand_dims(data, 0)

        in_chan = input_chan[ll]

        # Drop input channels?
        if reshape_inputs:
            if input_channel_skip[ll] > 0:
                data = np.delete(data, np.s_[:input_channel_skip[ll]], axis=1)
            data = np.delete(data, np.s_[in_chan:], axis=1)

        show_data(
            ll,
            verbose,
            verbose_all or ll == layers-1,
            data.shape,
            data,
            debug=debug_computation,
            expand=in_expand[ll],
            expand_thresh=in_expand_thresh[ll],
            operation=operator[ll],
            operands=operands[ll],
        )

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
            verbose_all or ll == layers-1,
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
            if out_size[0] != in_chan \
               or out_size[1] != pooled_dim[ll][0] or pooled_dim[ll][1] != 1:
                eprint(f'Input dimensions do not match in layer {ll}. '
                       f'Expected: {in_chan}x{pooled_dim[ll][0]}, '
                       f'got {out_size[0]}x{out_size[1]}.')
        else:
            if out_size[0] != in_chan \
               or out_size[1] != pooled_dim[ll][0] or out_size[2] != pooled_dim[ll][1]:
                eprint(f'Input dimensions do not match in layer {ll}. '
                       f'Expected: {in_chan}x{pooled_dim[ll][0]}x{pooled_dim[ll][1]}, '
                       f'got {out_size[0]}x{out_size[1]}x{out_size[2]}.')

        if operands[ll] > 1 and pool_first[ll]:
            data = run_eltwise(data, ll)
        else:
            data = np.squeeze(data, axis=0)

        # Convolution or passthrough
        if operator[ll] == op.CONV2D:
            if flatten[ll]:
                in_chan *= pooled_dim[ll][0] * pooled_dim[ll][1]
                data = data.reshape(in_chan, 1, 1)
                if verbose:
                    print_data(
                        verbose,
                        f'FLATTEN TO {in_chan}x1x1',
                        data,
                        data.shape,
                        in_expand[ll],
                        in_chan,
                    )

            out_buf, out_size = conv2d_layer(
                ll,
                verbose,
                verbose_all or ll == layers-1,
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
                    in_chan // conv_groups[ll],
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
                verbose_all or ll == layers-1,
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
                    in_chan // conv_groups[ll],
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
                verbose_all or ll == layers-1,
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
                    input_chan[ll] // conv_groups[ll],
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
                verbose_all or ll == layers-1,
                data.shape,
                data,
                device=device,
                debug=debug_computation,
            )
        else:
            eprint(f'Unknown operator `{op.string(operator[ll])}`.')

        assert out_size[0] == output_chan[ll] \
            and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]

        # Write .mem file for output or create the C check_output() function to verify the output
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

            apb.output(f'// Expected output of layer {ll} for {test_name} '
                       'given the sample input\n')
            apb.function_header(dest='wrapper', prefix='', function='check_output')
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
                    write_gap=write_gap[ll],
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
                out_map2 = [None] * tc.dev.C_GROUP_OFFS * tc.dev.P_NUMGROUPS
                apb2.verify_unload(
                    ll,
                    in_map,
                    out_map2,
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
                    write_gap=write_gap[ll],
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
                write_gap=write_gap[ll],
                layers=layers,
            )
            apb.function_footer(dest='wrapper')  # check_output()
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

            if fc_weights or softmax or embedded_code:
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
                    write_gap=write_gap[ll],
                )

            if fc_weights:
                data = data.flatten()

                out_buf, out_size = linear_layer(
                    verbose=verbose,
                    verbose_data=verbose_all or ll == layers-1,
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

            summary_stats = '/*\n' + \
                            stats.summary(factor=repeat_layers, debug=debug, spaces=2,
                                          weights=kernel, w_size=quantization, bias=bias) + \
                            '*/\n'
            apb.main()
            apb.output(summary_stats + '\n')

    # Close header files
    if sampledata_header is not None:
        sampledata_header.close()
    if weight_header is not None:
        weight_header.close()
    if apifile is not None:
        apifile.close()
    if rtl_preload:
        apb.write_mem(base_directory, test_name)

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
            rtl_preload=rtl_preload,
            result_output=result_output,
        )
        assets.copy('assets', 'rtlsim-ai' + str(device), base_directory, test_name)
        if riscv_cache:
            assets.copy('assets', 'rtlsim-riscv-cache-ai' + str(device), base_directory, test_name)
        elif riscv_flash:
            assets.copy('assets', 'rtlsim-riscv-flash-ai' + str(device), base_directory, test_name)
        elif riscv:
            assets.copy('assets', 'rtlsim-riscv-ai' + str(device), base_directory, test_name)
        if result_output:
            assets.copy('assets', 'rtlsim-verify-output', base_directory, test_name)
    elif block_mode:
        assets.copy('assets', 'blocklevel-ai' + str(device), base_directory, test_name)
    elif embedded_code:
        insert = summary_stats + \
                 '\n/* Number of outputs for this network */\n' \
                 '#define CNN_NUM_OUTPUTS ' \
                 f'{fc_weights[0].shape[0] if fc_weights else output_chan[-1]}'
        if timer is not None:
            insert += '\n\n/* Use this timer to time the inference */\n' \
                      f'#define CNN_INFERENCE_TIMER MXC_TMR{timer}'

        if riscv:
            assets.from_template('assets', 'embedded-riscv-ai' + str(device), base_directory,
                                 test_name, board_name, '', riscv=riscv)
        else:
            assets.from_template('assets', 'embedded-ai' + str(device), base_directory,
                                 test_name, board_name, '', riscv=riscv)
        assets.from_template('assets', 'eclipse', base_directory,
                             test_name, board_name, '', riscv=riscv)
        assets.from_template('assets', 'device-all', base_directory,
                             test_name, board_name, insert, riscv=riscv)
        assets.from_template('assets', 'device-ai' + str(device), base_directory,
                             test_name, board_name, '', riscv=riscv)

    return test_name


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=np.inf, linewidth=190)

    args = commandline.get_parser()

    if args.device != 84 and args.fc_layer:
        wprint('--fc-layer should only be used on AI84.')

    # Configure device
    tc.dev = tc.get_device(args.device)

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
    cfg, cfg_layers, params = yamlcfg.parse(args.config_file, args.stop_after, args.device)

    # If not using test data, load weights and biases
    # This also configures the network's output channels
    if cfg['arch'] != 'test':
        if not args.checkpoint_file:
            eprint("--checkpoint-file is a required argument.")
        fext = args.checkpoint_file.rsplit(sep='.', maxsplit=1)[1].lower()
        if fext == 'onnx':
            # ONNX file selected
            layers, weights, bias, output_shift, fc_weights, \
                fc_bias, input_channels, output_channels = \
                onnxcp.load(
                    args.checkpoint_file,
                    cfg['arch'],
                    args.fc_layer,
                    params['quantization'],
                    params['bias_quantization'],
                    params['output_shift'],
                    params['kernel_size'],
                    params['operator'],
                    args.display_checkpoint,
                    args.no_bias,
                )
        else:
            # PyTorch checkpoint file selected
            layers, weights, bias, output_shift, fc_weights, \
                fc_bias, input_channels, output_channels = \
                checkpoint.load(
                    args.checkpoint_file,
                    cfg['arch'],
                    args.fc_layer,
                    params['quantization'],
                    params['bias_quantization'],
                    params['output_shift'],
                    params['kernel_size'],
                    params['operator'],
                    args.display_checkpoint,
                    args.no_bias,
                    params['conv_groups'],
                )
    else:  # Get some hard-coded sample weights
        layers, weights, bias, output_shift, fc_weights, \
            fc_bias, input_channels, output_channels = \
            sampleweight.load(
                cfg['dataset'],
                params['quantization'],
                params['bias_quantization'],
                params['output_shift'],
                cfg_layers,
                cfg['weights'] if 'weights' in cfg else None,
                cfg['bias'] if 'bias' in cfg else None,
                args.no_bias,
                params['conv_groups'],
                params['operator'],
            )

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

    if any(p < 0 or p > 4*tc.dev.MEM_SIZE for p in params['output_offset']):
        eprint('Unsupported value for `out_offset` in YAML configuration.')

    if any(q != 8 for q in params['bias_quantization']):
        eprint('All bias quantization configuration values must be 8.')

    in_sequences = params['in_sequences'][:layers]

    # Override channels
    for ll in range(layers):
        if in_sequences[ll] is not None:
            if isinstance(in_sequences[ll], list):
                if params['eltwise'][ll] == op.NONE:
                    # Concatenate
                    input_channels[ll] = sum(output_channels[i] for i in in_sequences[ll])
                else:
                    # Element-wise operation
                    input_channels[ll] = output_channels[in_sequences[ll][0]]
                    for i in range(1, len(in_sequences[ll])):
                        assert output_channels[in_sequences[ll][0]] \
                            == output_channels[in_sequences[ll][i]]
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

    if args.device != devices.CMSISNN:
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
    input_skip = params['input_skip'][:layers]
    input_channel_skip = params['input_chan_skip'][:layers]
    output_channels = output_channels[:layers]
    output_offset = params['output_offset'][:layers]
    conf_input_dim = params['input_dim'][:layers]
    input_offset = params['input_offset'][:layers]
    kernel_size = params['kernel_size'][:layers]
    quantization = params['quantization'][:layers]
    output_shift = output_shift[:layers]
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
    if args.streaming_layers is not None:
        # Additional (or only) streaming layers from command line
        for _, e in enumerate(args.streaming_layers):
            streaming[e] = True
    flatten = params['flatten'][:layers]
    operands = params['operands'][:layers]
    eltwise = params['eltwise'][:layers]
    pool_first = params['pool_first'][:layers]
    activation = params['activation'][:layers]
    conv_groups = params['conv_groups'][:layers]
    write_gap = params['write_gap'][:layers]

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
                    auto_input_dim[ll] = dim
                else:
                    auto_input_dim[ll] = output_dim[in_sequences[ll]]
            else:
                auto_input_dim[ll] = output_dim[ll-1]
            if conf_input_dim[ll] is None:
                input_dim[ll] = auto_input_dim[ll]
            else:
                input_dim[ll] = conf_input_dim[ll]
        if operator[ll] != op.CONV1D:
            if pool_stride[ll][0] != pool_stride[ll][1]:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support non-square '
                       f'pooling stride (currently set to '
                       f'{pool_stride[ll][0]}x{pool_stride[ll][1]}).')
            pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0])
                           // pool_stride[ll][0],
                           (input_dim[ll][1] + pool_stride[ll][1] - pool[ll][1])
                           // pool_stride[ll][1]]
        else:
            pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0])
                           // pool_stride[ll][0],
                           1]

        pooled_dim[ll] = pooled_size
        if any(dim == 0 for dim in pooled_dim[ll]):
            eprint(f'Pooling in layer {ll} results in a zero data dimension '
                   f'(input {input_dim[ll]}, pooled {pooled_dim[ll]}).')

        if operator[ll] != op.CONV1D:
            if stride[ll][0] != stride[ll][1]:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support non-square '
                       f'stride (currently set to {stride[ll][0]}x{stride[ll][1]}).')
            if operator[ll] != op.CONVTRANSPOSE2D and stride[ll][0] != 1:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support stride other '
                       f'than 1 (currently set to {stride[ll][0]}x{stride[ll][1]}).')
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
                if pooled_dim[ll][0] * pooled_dim[ll][1] > 256:
                    eprint(f'`flatten` in layer {ll} exceeds supported input dimensions '
                           f'({pooled_dim[ll][0]} * {pooled_dim[ll][1]} > 256)).')
                output_dim[ll] = [1, 1]
                input_channels[ll] //= pooled_dim[ll][0] * pooled_dim[ll][1]
                assert input_channels[ll] > 0
            if padding[ll][0] >= 3 and args.device != devices.CMSISNN:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                       f'(currently set to {padding[ll][0]}).')
        else:
            # We don't have to consider padding for the width calculation,
            # since padding has to be a multiple of 3 and we check for that.
            if padding[ll][0] >= 3 and args.device != devices.CMSISNN:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                       f'(currently set to {padding[ll][0]}).')
            output_dim[ll] = [(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1) - 1 +
                               2 * padding[ll][0]) // stride[ll][0] + 1,
                              1]

        # Prohibit pad greater than or equal to kernel size
        if padding[ll][0] >= kernel_size[ll][0] or padding[ll][1] >= kernel_size[ll][1]:
            eprint(f'Pad size ({padding[ll]}) for layer {ll} is greater than or equal to'
                   f' kernel size ({kernel_size[ll]}).')

        # Check for max dimensions
        if any(dim > tc.dev.MAX_ROW_COL for dim in input_dim[ll]):
            eprint(f'Input dimension {input_dim[ll]} exceeds system maximum of '
                   f'{tc.dev.MAX_ROW_COL} in layer {ll}.')
        if any(dim > tc.dev.MAX_ROW_COL for dim in output_dim[ll]):
            eprint(f'Output dimension {output_dim[ll]} exceeds system maximum of '
                   f'{tc.dev.MAX_ROW_COL} in layer {ll}.')

        assert input_channels[ll] > 0

    if args.riscv and not args.riscv_cache and args.embedded_code:
        eprint("Embedded code on RISC-V requires --riscv-cache.")

    if args.device != devices.CMSISNN:
        tn = create_net(
            args.prefix,
            args.verbose,
            args.verbose_all,
            args.debug,
            args.debug_computation,
            args.debug_latency,
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
            input_skip,
            input_channel_skip,
            args.input_filename,
            args.output_filename,
            args.c_filename,
            args.api_filename,
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
            args.compact_data and not args.rtl_preload,
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
            args.debugwait,
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
            args.legacy_kernels,
            args.log_intermediate,
            args.log_pooling,
            args.allow_streaming,
            args.softmax,
            args.clock_trim,
            args.repeat_layers,
            args.fixed_input,
            args.max_count,
            args.boost,
            args.forever,
            write_gap,
            args.start_layer,
            args.pipeline,
            args.pll,
            args.reshape_inputs,
            args.link_layer,
            args.energy,
            args.timer,
            args.board_name,
            args.rd_ahead,
            args.calcx4,
            args.rtl_preload,
            args.result_output,
            args.weight_start,
        )
        if not args.embedded_code and args.autogen.lower() != 'none':
            rtlsim.append_regression(
                args.top_level,
                tn,
                args.queue_name,
                args.autogen,
            )
    else:
        wprint('CMSIS-NN code generation is unsupported.')

        cmsisnn.create_net(
            args.prefix,
            args.verbose,
            args.verbose_all,
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
            fc_weights,
            fc_bias,
            flatten,
            operands,
            eltwise,
            pool_first,
            in_sequences,
            args.c_filename,
            args.test_dir,
            args.log_filename,
            args.weight_filename,
            args.sample_filename,
            args.avg_pool_rounding,
            args.device,
            args.legacy_test,
        )

    print(stats.summary(factor=args.repeat_layers, debug=args.debug,
                        weights=weights, w_size=quantization, bias=bias))


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
