###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Backend for MAX7800X embedded code generation and RTL simulations
"""
import hashlib
import os
import sys

import numpy as np

from . import apbaccess, assets, compute, kbias, kernels, load, op, rtlsim, stats
from . import tornadocnn as tc
from .eprint import eprint, wprint
from .simulate import (conv1d_layer, conv2d_layer, convtranspose2d_layer, eltwise_layer,
                       passthrough_layer, pooling_layer, print_data, show_data)
from .utils import ffs, fls, popcount


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
        split,
        in_offset,
        out_offset,
        streaming,
        flatten,
        operands,
        eltwise,
        pool_first,
        in_sequences,
        next_sequence,
        prev_sequence,
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
        init_tram=False,
        avg_pool_rounding=False,
        fifo=False,
        fast_fifo=False,
        fast_fifo_quad=False,
        zero_sram=False,
        mlator=False,
        oneshot=0,
        ext_rdy=False,
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
        first_layer_used=0,
        final_layer=-1,
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
        wfi=True,
        bypass=None,
):
    """
    Chain multiple CNN layers, create and save input and output
    """
    device = tc.dev.device

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

    if start_layer > 0 and not tc.dev.SUPPORT_LINK_LAYER:
        eprint("`--start-layer` is not supported on this device.")

    if start_layer > tc.dev.MAX_START_LAYER:
        eprint(f"`--start-layer` is set to {start_layer}, but the device only supports "
               f"a maximum of {tc.dev.MAX_START_LAYER}.")

    if link_layer and not tc.dev.SUPPORT_LINK_LAYER:
        eprint("`--link-layer` is not supported on this device.")

    if rd_ahead and not tc.dev.SUPPORT_READ_AHEAD:
        eprint("`--read-ahead` is not supported on this device.")

    if calcx4 and not tc.dev.SUPPORT_CALCX4:
        eprint("`--calcx4` is not supported on this device.")

    if pipeline and not tc.dev.SUPPORT_PIPELINE:
        eprint("`--pipeline` is not supported on this device.")

    if pll and not tc.dev.SUPPORT_PLL:
        eprint("`--pll` is not supported on this device.")

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

    if result_output and (mlator or oneshot or stopstart):
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
        if start_layer != 0:
            eprint('`--start_layer` must be 0 when using a FIFO.')

        if input_chan[start_layer] > 16 or big_data[start_layer] and input_chan[start_layer] > 4:
            eprint("Using the FIFO is restricted to a maximum of 4 input channels (CHW) or "
                   f"16 channels (HWC); this test is using {input_chan[start_layer]} channels.")
        if big_data[start_layer] and processor_map[start_layer] & ~0x0001000100010001 != 0 \
           or not big_data[start_layer] and processor_map[start_layer] & ~0x000f000f000f000f != 0:
            eprint("The FIFO is restricted to processors 0, 16, 32, 48 (CHW) or "
                   "0-3, 16-19, 32-35, 48-51 (HWC).")
        if fast_fifo:
            if big_data[start_layer] and input_chan[start_layer] > 1:
                eprint("Fast FIFO supports only a single CHW input channel; "
                       f"this test is using {input_chan[start_layer]} channels.")
            elif not big_data[start_layer] and input_chan[start_layer] > 4:
                eprint("Fast FIFO supports up to four HWC input channels; "
                       f"this test is using {input_chan[start_layer]} channels.")
            if processor_map[start_layer] != 1 and processor_map[start_layer] & 0x0e == 0:
                fifo_group = False
            if output_width[start_layer] != 8:
                eprint('Single-layer fast FIFO setup requires output width of 8.')
            if operator[start_layer] == op.NONE:
                eprint('Fast FIFO requires a convolution operation in the first layer.')
    elif streaming[start_layer] and not allow_streaming:
        eprint('Streaming in the first layer requires use of a FIFO.')
    if any(streaming) and start_layer != 0:
        eprint('`--start_layer` must be 0 when using streaming.')
    for ll in range(min(tc.dev.MAX_STREAM_LAYERS, layers)):
        if next_sequence[ll] != -1 and next_sequence[ll] != ll + 1 and streaming[ll]:
            eprint(f'`next_sequence` must be {ll+1} when using streaming in layer {ll}. '
                   f'Currently configured: {next_sequence[ll]}')

    if mlator and (output_dim[final_layer][start_layer] * output_dim[final_layer][1] < 4
                   or output_width[final_layer] > 8):
        wprint('--mlator should only be used with 4 or more 8-bit outputs per channel; ignoring.')
        mlator = False

    if fast_fifo and not riscv:
        eprint('--fast-fifo requires --riscv')

    if sleep and not riscv:
        eprint('--deepsleep requires --riscv')

    if oneshot and timer is not None:
        eprint('--timer is not supported when using --one-shot')

    if not tc.dev.SUPPORT_KERNEL_BYPASS \
       and any(bypass[ll] for ll in range(first_layer_used, layers)):
        eprint('Kernel bypass is not supported on this device.')

    processor_map_0 = processor_map[start_layer]
    if fast_fifo_quad:
        processor_map[start_layer] = processor_map_0 << 48 | processor_map_0 << 32 \
            | processor_map_0 << 16 | processor_map_0

    binary_quantization = any(quantization[ll] == -1 for ll in range(first_layer_used, layers))

    # Check that input channels are in separate memory instances if CHW (big) data format is used,
    # and calculate input and output expansion
    for ll in range(first_layer_used, layers):
        if quantization[ll] is None:
            quantization[ll] = 8 if not bypass[ll] else 0  # Set default
        elif quantization[ll] == 1 and binary_quantization:
            eprint(f"Cannot combine binary quantization in layer {ll} with 1-bit quantization.")
        if output_shift[ll] is None:
            output_shift[ll] = 0 if not bypass[ll] else 7  # Set default

        if output_shift[ll] < -15 or output_shift[ll] > 15:
            implicit_shift = 8 - abs(quantization[ll]) if not bypass[ll] else 0
            eprint(f"Layer {ll} with {abs(quantization[ll])}-bit weight quantization supports an "
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
        if (not streaming[ll] or ll == final_layer) \
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
            if flatten[ll]:
                eprint(f'Layer {ll}: convolution groups ({conv_groups[ll]}) > 1 are not supported'
                       f' when flattening.')
            # if output_width[ll] != 8:
            #     eprint(f'Layer {ll}: convolution groups ({conv_groups[ll]}) > 1 are not'
            #            f' supported when using `wide` output.')

        if input_skip[ll] != 0 and not tc.dev.SUPPORT_MULTIPASS_STRIDE:
            eprint(f'Layer {ll}: `in_skip` must be 0 for this device.')

    # Create comment of the form "k1_b0-1x32x32b_2x2s2p14-..."
    test_name = prefix
    if not embedded_code:
        for ll in range(first_layer_used, layers):
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

    try:
        target_dir = os.path.join(base_directory, test_name)
        os.makedirs(target_dir, exist_ok=False)
    except OSError:
        wprint(target_dir, 'exists')

    # Redirect stdout?
    if log:
        sys.stdout = open(os.path.join(base_directory, test_name, log_filename), 'w')
        print(f'{" ".join(str(x) for x in sys.argv)}')
        print(f'{tc.dev.partnum}\n')
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
    group_map = [None] * layers
    broadcast_mode = [None] * layers
    emulate_eltwise = [False] * layers
    for ll in range(first_layer_used, layers):
        bits = processor_map[ll]
        processors_used |= bits

        if input_chan[ll] > tc.dev.MAX_CHANNELS:
            eprint(f'Layer {ll} is configured for {input_chan[ll]} inputs, which exceeds '
                   f'the system maximum of {tc.dev.MAX_CHANNELS}.')
        if output_chan[ll] > tc.dev.MAX_CHANNELS:
            eprint(f'Layer {ll} is configured for {output_chan[ll]} outputs, which exceeds '
                   f'the system maximum of {tc.dev.MAX_CHANNELS}.')
        if (ll != start_layer or not fast_fifo_quad) \
           and popcount(processor_map[ll]) != in_expand_thresh[ll]:
            eprint(f'Layer {ll} has {input_chan[ll]} inputs with input expansion '
                   f'{in_expand[ll]}, {operands[ll]} operands, threshold {in_expand_thresh[ll]}, '
                   f'but enabled processor map 0x{processor_map[ll]:016x} '
                   f'has {popcount(processor_map[ll])} bits instead of the '
                   f'expected number of {in_expand_thresh[ll]}.')
        if ll == start_layer and fast_fifo_quad \
           and popcount(processor_map_0) != in_expand_thresh[ll]:
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
        group_map[ll] = this_map

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
                           f'input {processor_map[ll]:016x}, output '
                           f'{output_processor_map[ll]:016x}.')

        # Ensure byte positions are the same in the input and output map for depthwise convolutions
        if conv_groups[ll] > 1:
            if ffs(output_processor_map[ll]) % tc.dev.P_SHARED != 0:
                eprint(f'Layer {ll} is a depth-wise convolution. Output processors '
                       'must be aligned to a multiple of 4. Configured for this layer: '
                       f'{output_processor_map[ll]:016x}.')
            if ffs(processor_map[ll]) % tc.dev.P_SHARED != 0 \
               and (processor_map[ll] >> ffs(processor_map[ll])) // 2**tc.dev.P_NUMPRO > 0:
                eprint(f'Layer {ll} is a depth-wise convolution. When spanning groups, processors '
                       'must be aligned to a multiple of 4. Configured for this layer: '
                       f'{processor_map[ll]:016x}.')
            if processor_map[ll] != output_processor_map[ll]:
                wprint(f'Layer {ll}: depth-wise convolution moves data across processors. This '
                       f'has a performance impact. Input {processor_map[ll]:016x}, output '
                       f'{output_processor_map[ll]:016x}.')
                broadcast_mode[ll] = False
            else:
                broadcast_mode[ll] = True

        # Block certain element-wise operations when not using passthrough mode
        if tc.dev.EMULATE_ELTWISE_MP and operands[ll] > 1 and in_expand[ll] > 1 \
           and operands[ll] * in_expand[ll] != operands[ll] + in_expand[ll]:
            if operator[ll] != op.NONE or pool[ll][0] > 1 or pool[ll][1] > 1 \
               or pool_stride[ll][0] > 1 or pool_stride[ll][1] > 1:
                eprint(f'The element-wise operation in layer {ll} exceeds a multi-pass of 2 '
                       'and therefore does not support pooling or convolution.')
            emulate_eltwise[ll] = True

    groups_used = []
    for group in range(tc.dev.P_NUMGROUPS):
        if ((processors_used |
             output_processor_map[final_layer]) >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
            groups_used.append(group)

    if 0 not in groups_used:
        eprint('Group 0 is not used, this currently does not work.')

    # Create ARM code wrapper if needed
    if riscv and not block_mode:
        with open(os.path.join(base_directory, test_name, c_filename + '.c'), mode='w') as f:
            apb = apbaccess.apbwriter(
                f,
                apb_base,
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
            input_chan=input_chan[start_layer],
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
            oneshot=final_layer if oneshot else 0,
            softmax=softmax,
            stopstart=stopstart,
            num_classes=output_chan[final_layer],
            output_width=output_width[final_layer],
            bias=any(b is not None for b in bias),
            wfi=wfi,
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
            for ll in range(first_layer_used, layers):
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
                big_data[start_layer],
                processor_map_0,
                in_offset[start_layer],
                [input_chan[start_layer], input_dim[start_layer][0], input_dim[start_layer][1]],
                in_expand[start_layer],
                operands[start_layer],
                in_expand_thresh[start_layer],
                data,
                padding[start_layer],
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
            kern_offs, kern_len, kern_count, kern_ochan = kernels.load(
                verbose,
                True,
                apb,
                first_layer_used,
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
                conv_groups,
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
                bypass=bypass,
            )
            bias_offs, bias_group, group_bias_max = kbias.load(
                verbose,
                True,
                apb,
                first_layer_used,
                layers,
                bias,
                group_map,
                output_chan,
                streaming,
                conv_groups,
                broadcast_mode,
                processor_map,
                output_processor_map,
                out_expand,
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
            #     for k in range(first_layer_used, layers):
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
            val = (repeat_layers * final_layer) | (start_layer << 8)
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
            kern_offs, kern_len, kern_count, kern_ochan = kernels.load(
                verbose,
                embedded_code,
                apb,
                first_layer_used,
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
                conv_groups,
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
                bypass=bypass,
            )
            bias_offs, bias_group, group_bias_max = kbias.load(
                verbose,
                embedded_code,
                apb,
                first_layer_used,
                layers,
                bias,
                group_map,
                output_chan,
                streaming,
                conv_groups,
                broadcast_mode,
                processor_map,
                output_processor_map,
                out_expand,
                debug,
            )

        if verbose:
            print('\nGlobal configuration:')
            print('---------------------')
            print(f'Used processors     = 0x{processors_used:016x}')
            print(f'Used groups         = {groups_used}')
            if start_layer > 0:
                print(f'Starting layer      = {start_layer}')
            if any(s != i+1 and (s != -1 or i != final_layer)
                   for i, s in enumerate(next_sequence)):
                print('Next layer sequence = [',
                      ', '.join(str(k) if k != -1 else 'stop' for k in next_sequence), ']',
                      sep='',)

            print('\nPer-group configuration:')
            print('-----------------------')
            print(f'Used bias memory    = {group_bias_max}')

            print('\nPer-layer configuration:')
            print('------------------------')
            if repeat_layers > 1:
                print(f'Layer repeat count  = {repeat_layers}')
            print(f'Group map           = {group_map}')

            print('Input offset        = [',
                  ', '.join('0x{:04x}'.format(k) if k is not None
                            else 'N/A' for k in in_offset), ']', sep='',)
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
                  ', '.join('0x{:016x}'.format(k) if k is not None
                            else 'N/A' for k in output_processor_map), ']', sep='',)
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
            print('Kernel size (bits)  = [',
                  ', '.join(str(k) if k >= 0
                            else 'b' for k in quantization), ']', sep='',)
            if any(bypass):
                print(f'Kernel bypass       = {bypass}')
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
            for ll in range(first_layer_used, layers):

                local_source = False
                for _, group in enumerate(groups_used):
                    # Local output must be used:
                    # - for depthwise convolutions
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
                        if broadcast_mode[ll]:
                            pop = popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                           % 2**tc.dev.P_NUMPRO)
                            tscnt_max = max(
                                tscnt_max,
                                (min(pop - 1, 3) + 1) * (output_width[ll] // 8) - 1
                            )
                        else:
                            tscnt_max = max(
                                tscnt_max,
                                (popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                          % 2**tc.dev.P_NUMPRO) * output_width[ll] + 7) // 8 - 1
                            )

                for _, group in enumerate(groups_used):
                    apb.output(f'  // Layer {r * layers + ll} group {group}\n', embedded_code)

                    if hasattr(tc.dev, 'LREG_NXTLYR'):
                        val = 0
                        if link_layer:
                            if ll != final_layer:
                                val = 1 << 7 | (ll + 1)
                            else:
                                val = 1 << 8  # Stop
                        else:
                            if next_sequence[ll] == -1:
                                if ll != layers - 1:  # Don't set stop bit when we don't have to
                                    val = 1 << 8
                            elif next_sequence[ll] != ll + 1:
                                val = 1 << 7 | next_sequence[ll]
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
                        elif operator[ll] == op.NONE and emulate_eltwise[ll]:
                            in_row = input_dim[ll][0] * in_expand[ll]
                            in_col = input_dim[ll][1]
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
                        # Configure SRAM write pointer -- write ptr is global (unless depth-wise
                        # w/o broadcast is used).
                        # Get offset to first available instance of the first used processor of the
                        # next layer.
                        if operator[ll] != op.NONE \
                           and conv_groups[ll] > 1 and not broadcast_mode[ll]:
                            # First group used
                            first_group = ffs(processor_map[ll]) // tc.dev.P_NUMPRO
                            if group - first_group >= 0:
                                # Target for first write in the group
                                wptr = (group - first_group) * tc.dev.P_NUMPRO \
                                    + ffs(output_processor_map[ll])
                                if group != first_group:
                                    # Correct for unused processors in the first group
                                    wptr -= ffs(processor_map[ll]) % tc.dev.P_NUMPRO

                                val |= (wptr // tc.dev.P_SHARED) << tc.dev.WRITE_PTR_SHIFT
                            else:
                                val = 0
                        else:
                            if operator[ll] != op.NONE:
                                instance = ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1)
                            elif (output_processor_map[ll] &
                                  2**tc.dev.P_NUMPRO - 1 << group*tc.dev.P_NUMPRO > 0):
                                instance = ffs(output_processor_map[ll]
                                               & 2**tc.dev.P_NUMPRO - 1
                                               << group*tc.dev.P_NUMPRO) \
                                    & ~(tc.dev.P_SHARED-1)
                            else:
                                instance = 0

                            val |= (instance % tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE \
                                | (instance // tc.dev.P_SHARED) << tc.dev.WRITE_PTR_SHIFT
                    else:
                        # FIXME: No test currently sets local_souce, so this code is suspect
                        instance = ffs(output_processor_map[ll] >> group * tc.dev.P_SHARED) \
                               & ~(tc.dev.P_SHARED-1)
                        val |= (instance + group * tc.dev.P_SHARED) * tc.dev.INSTANCE_SIZE
                    assert val < 2**tc.dev.MAX_PTR_BITS
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_BASE, val,
                                   verbose, comment=' // SRAM write ptr')

                    # Write Pointer Timeslot Offset Register
                    # Used for 1x1 convolution, and pooling without convolution
                    val = 0
                    if operator[ll] == op.CONV2D:
                        if kernel_size[ll] == [1, 1] and conv_groups[ll] == 1:
                            val = 1
                        elif conv_groups[ll] > 1 and not broadcast_mode[ll]:
                            val = tc.dev.INSTANCE_SIZE * 4
                    elif operator[ll] == op.NONE:
                        if popcount(processor_map[ll]) > 4 \
                           or operands[ll] > 1 and in_expand[ll] > 1:
                            val = tc.dev.INSTANCE_SIZE * 4
                        else:
                            val = tc.dev.INSTANCE_SIZE
                    assert val < 2**tc.dev.MAX_PTR_BITS
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_TOFFS, val,
                                   verbose, comment=' // Write ptr time slot offs')

                    if operator[ll] != op.NONE:
                        # [15:0] Write Pointer Mask Offset Register
                        val = 1 << tc.dev.WRITE_PTR_SHIFT
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_WPTR_MOFFS, val,
                                       verbose, comment=' // Write ptr mask offs')

                    # [15:0] Write Pointer Multi-Pass Channel Offset Register
                    val = 0
                    if out_expand[ll] > 1:
                        val = (output_width[ll] // 8) * (write_gap[ll] + 1)
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
                        val |= 1 << 11
                    if conv_groups[ll] > 1 and broadcast_mode[ll]:
                        val |= 1 << 29

                    if output_width[ll] != 8:
                        val |= 1 << 16

                    if (ll != start_layer or not fast_fifo_quad) \
                       and operator[ll] != op.NONE and group == groups_used[0] \
                       and conv_groups[ll] == 1:
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

                    if bypass[ll]:
                        val |= 1 << 30

                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL, val,
                                   verbose, comment=' // Layer control')

                    flatten_prod = 0
                    if flatten[ll]:
                        # Store all bits, top programmed in post processing register
                        flatten_prod = \
                            in_expand[ll] * pooled_dim[ll][0] * pooled_dim[ll][1] - 1
                        in_exp = flatten_prod & 0x0f  # Lower 4 bits only
                    elif operator[ll] == op.NONE and emulate_eltwise[ll]:
                        in_exp = 0
                    else:
                        in_exp = in_expand[ll] - 1

                    assert in_exp < 2**4  # Cannot have more than 4 bits

                    quant = abs(quantization[ll]) if not bypass[ll] else 8
                    val = (fls(output_processor_map[ll])
                           - (ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1))) \
                        * quant << tc.dev.XPCH_MAX_OFFS | in_exp
                    if operator[ll] != op.NONE:
                        wptr_skip = out_expand[ll] * (write_gap[ll] + 1) - 1
                    else:
                        wptr_skip = write_gap[ll]
                    assert wptr_skip < 2**tc.dev.MAX_WPTRINC_BITS
                    val |= wptr_skip << 4

                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_LCTL2, val,
                                   verbose, comment=' // Layer control 2')

                    # Configure mask start and end addresses
                    # Every mask memory starts from the same offset for all processors
                    oned_sad = 0
                    if operator[ll] != op.NONE:
                        kc = kern_count[ll] if not bypass[ll] \
                            else output_chan[ll] // conv_groups[ll]  # FIXME: bypass corner cases
                        kl = (kc - 1) * quant

                        if ll == start_layer and calcx4:
                            # FIXME: Handle fast_fifo_quad and calcx4
                            if calcx4:
                                kl += quant
                            kl = (kl + 3) // 4
                            if calcx4:
                                kl -= quant
                        koffs, oned_sad = divmod(9 * kern_offs[ll],
                                                 kernel_size[ll][0] * kernel_size[ll][1])
                        koffs *= 8
                    else:
                        kl = koffs = 0

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
                        elif emulate_eltwise[ll]:
                            val = 0
                        else:
                            val = (out_expand[ll] - 1) * 8
                            assert val < 2**16
                        apb.write_lreg(group, r * layers + ll, tc.dev.LREG_MCNT, val,
                                       verbose, comment=' // Mask offset and count')

                    if hasattr(tc.dev, 'LREG_OCHAN'):
                        if bypass[ll]:
                            val = output_chan[ll] - 1
                        elif operator[ll] != op.NONE and conv_groups[ll] == 1:
                            val = kern_ochan[ll] - 1
                            if calcx4:
                                val //= 4
                        elif conv_groups[ll] > 1:
                            val = (tscnt_max + 1) * in_expand[ll] - 1
                        else:
                            val = tscnt_max
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
                    if abs(quantization[ll]) == 1:
                        val = 1 << 22
                    elif quantization[ll] == 2:
                        val = 2 << 22
                    elif quantization[ll] == 4:
                        val = 3 << 22
                    else:
                        assert quantization[ll] in [0, 8]
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

                    if conv_groups[ll] == 1 and group == bias_group[ll]:
                        # Enable bias only for one group
                        assert bias_offs[ll][group] < 2**12
                        val |= 1 << 12 | bias_offs[ll][group]
                    elif bias_offs[ll][group] is not None and conv_groups[ll] > 1:
                        # Enable bias for all groups
                        assert bias_offs[ll][group] < 2**12
                        offs = bias_offs[ll][group]
                        if broadcast_mode[ll]:
                            offs //= 4
                        val |= 1 << 12 | offs

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
                    if operator[ll] != op.NONE and not bypass[ll]:
                        val = val << 16 | val  # Mask enables
                    apb.write_lreg(group, r * layers + ll, tc.dev.LREG_ENA, val,
                                   verbose, comment=' // Mask and processor enables')

                    if ll == start_layer and fifo:
                        # Start: 1
                        if override_start is not None:
                            stream_start = override_start
                        elif streaming[ll]:
                            stream_start = (pool[ll][0] - 1) * input_dim[ll][1] + pool[ll][1]
                        else:
                            val = input_dim[start_layer][0] * input_dim[start_layer][1]
                            if big_data[start_layer]:
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
                        stream_start = (pooled_dim[prev_sequence[ll]][1]
                                        + 2 * padding[prev_sequence[ll]][1]) \
                            * (kernel_size[prev_sequence[ll]][0] - 1 + pool[ll][0] - 1) \
                            + kernel_size[prev_sequence[ll]][1] - 1 + pool[ll][1] + increase_start
                        assert stream_start < 2**tc.dev.MAX_ISVAL_BITS

                        # Delta 1: This layer's pooling stride
                        delta1 = pool_stride[ll][1] * operands[ll] + increase_delta1
                        assert delta1 < 2**5
                        # Delta 2: (This layer's pooling - 1) * full prior layer's padded rows +
                        # prior layer's pad
                        delta2 = (pool_stride[ll][0] - 1) \
                            * (pooled_dim[prev_sequence[ll]][1]
                               + 2 * padding[prev_sequence[ll]][1]) \
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
                        if ll == start_layer and override_rollover is not None:
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
                                eprint(f'Layer {ll}: Overlapping input and output: '
                                       f'in_offset 0x{in_offset[ll]:08x} + '
                                       f'rollover 0x{val:08x} * 4 >= '
                                       f'out_offset 0x{out_offset[ll]:08x}.',
                                       error=not no_error_stop)
                        else:
                            if out_offset[ll] + val * 4 >= in_offset[ll]:
                                eprint(f'Layer {ll}: Overlapping input and output: '
                                       f'out_offset 0x{out_offset[ll]:08x} + '
                                       f'rollover 0x{val:08x} * 4 >= '
                                       f'in_offset 0x{in_offset[ll]:08x}.',
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

                    if ll == start_layer and fifo:
                        val = input_dim[start_layer][0] * input_dim[start_layer][1]
                        if big_data[start_layer]:
                            val = (val + 3) // 4
                        assert val < 2**tc.dev.MAX_IFRM_BITS
                        apb.write_ctl(group, tc.dev.REG_IFRM, val, verbose,
                                      comment=' // Input frame size')

                    apb.output('\n', embedded_code)  # End of group

        if zero_unused:
            for r in range(repeat_layers):
                for ll in range(first_layer_used, layers, tc.dev.MAX_LAYERS):
                    for _, group in enumerate(groups_used):
                        for reg in range(tc.dev.MAX_LREG+1):
                            if reg == tc.dev.LREG_RFU:  # Register 2 not implemented
                                continue
                            apb.write_lreg(group, r * layers + ll, reg, 0,
                                           verbose, force_write=True,
                                           comment=f' // Zero unused layer {ll} registers')
                if hasattr(tc.dev, 'MIN_STREAM_LREG'):
                    for ll in range(first_layer_used, layers, tc.dev.MAX_STREAM_LAYERS):
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
                    big_data[start_layer],
                    processor_map_0,
                    in_offset[start_layer],
                    [input_chan[start_layer],
                     input_dim[start_layer][0],
                     input_dim[start_layer][1]],
                    in_expand[start_layer],
                    operands[start_layer],
                    in_expand_thresh[start_layer],
                    data,
                    padding[start_layer],
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
                for i in range(input_chan[start_layer]):
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
        if binary_quantization:
            val |= 1 << 30
        if fast_fifo_quad:
            val |= 1 << 31  # Qupac bit
        if oneshot:
            val |= 1 << 8
        if ext_rdy:
            val |= 1 << 4
        if hasattr(tc.dev, 'CTL_PIPELINE_OFFS'):
            if not pipeline:
                val |= 1 << tc.dev.CTL_PIPELINE_OFFS
            if streaming[start_layer] and big_data[start_layer]:
                val |= 1 << 6

        if embedded_code:
            apb.function_footer()
            apb.function_header(function='start')

        if embedded_code or tc.dev.MODERN_SIM:
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
                    big_data[start_layer],
                    processor_map_0,
                    in_offset[start_layer],
                    [input_chan[start_layer],
                     input_dim[start_layer][0],
                     input_dim[start_layer][1]],
                    in_expand[start_layer],
                    operands[start_layer],
                    in_expand_thresh[start_layer],
                    data,
                    padding[start_layer],
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
            verbose_all or ll == final_layer,
            data[0].shape,
            output_shift[ll],
            data,
            output_width=o_width,
            debug=debug_computation,
            operands=operands[ll],
        )
        assert out_size[0] == d_shape[1] \
            and out_size[1] == d_shape[2] and out_size[2] == d_shape[3]

        return data

    ll = start_layer
    data_buf = [data]
    # Compute layer-by-layer output and chain results into input
    while ll < layers:
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
            if ll == start_layer and legacy_test:
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
            verbose_all or ll == final_layer,
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
            data = data.reshape(data.shape[0], -1, input_dim[ll][0])
        else:
            data = data.reshape(data.shape[0], -1, input_dim[ll][0], input_dim[ll][1])

        # In-flight pooling
        data, out_size = pooling_layer(
            ll,
            verbose,
            verbose_all or ll == final_layer,
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

            if not bypass[ll]:
                k = kernel[ll].reshape(
                        output_chan[ll],
                        in_chan // conv_groups[ll],
                        kernel_size[ll][0],
                        kernel_size[ll][1]
                    )
            else:
                k = np.full(
                        (output_chan[ll], in_chan, kernel_size[ll][0], kernel_size[ll][0]),
                        1,
                        dtype=np.int64,
                    )

            out_buf, out_size = conv2d_layer(
                ll,
                verbose,
                verbose_all or ll == final_layer,
                data.shape,
                kernel_size[ll],
                output_shift[ll],
                output_chan[ll],
                padding[ll],
                dilation[ll],
                stride[ll],
                activation[ll],
                k,
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                debug=debug_computation,
                bypass=bypass[ll],
            )
        elif operator[ll] == op.CONVTRANSPOSE2D:
            if not bypass[ll]:
                k = kernel[ll].reshape(
                        output_chan[ll],
                        in_chan // conv_groups[ll],
                        kernel_size[ll][0],
                        kernel_size[ll][1],
                    )
            else:
                k = np.full(
                        (output_chan[ll], in_chan, kernel_size[ll][0], kernel_size[ll][0]),
                        1,
                        dtype=np.int64,
                    )

            out_buf, out_size = convtranspose2d_layer(
                ll,
                verbose,
                verbose_all or ll == final_layer,
                data.shape,
                kernel_size[ll],
                output_shift[ll],
                output_chan[ll],
                padding[ll],
                dilation[ll],
                stride[ll],
                [1, 1],  # output_padding
                activation[ll],
                k,
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                debug=debug_computation,
                bypass=bypass[ll],
            )
        elif operator[ll] == op.CONV1D:
            if not bypass[ll]:
                k = kernel[ll].reshape(
                        output_chan[ll],
                        input_chan[ll] // conv_groups[ll],
                        kernel_size[ll][0],
                    )
            else:
                k = np.full(
                        (output_chan[ll], input_chan[ll], kernel_size[ll][0],),
                        1,
                        dtype=np.int64,
                    )

            out_buf, out_size = conv1d_layer(
                ll,
                verbose,
                verbose_all or ll == final_layer,
                data.shape,
                kernel_size[ll][0],
                output_shift[ll],
                output_chan[ll],
                padding[ll][0],
                dilation[ll][0],
                stride[ll][0],
                activation[ll],
                k,
                bias[ll],
                data,
                output_width=output_width[ll],
                groups=conv_groups[ll],
                debug=debug_computation,
                bypass=bypass[ll],
            )
        elif operator[ll] == op.NONE:  # '0'D (pooling only or passthrough)
            out_buf, out_size = passthrough_layer(
                ll,
                verbose,
                verbose_all or ll == final_layer,
                data.shape,
                data,
                debug=debug_computation,
            )
        else:
            eprint(f'Unknown operator `{op.string(operator[ll])}`.')

        assert out_size[0] == output_chan[ll] \
            and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]

        # Write .mem file for output or create the C check_output() function to verify the output
        out_map = [None] * tc.dev.C_GROUP_OFFS * tc.dev.P_NUMGROUPS
        if block_mode:
            if ll == final_layer:
                filename = output_filename + '.mem'  # Final output
            else:
                filename = f'{output_filename}-{ll}.mem'  # Intermediate output
            filemode = 'w'
        else:
            if ll == final_layer:
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
            if ll == final_layer and mlator and not mlator_noverify:
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
                    verify_kernels=verify_kernels,
                    master=groups_used[0] if oneshot > 0 or stopstart else False,
                    riscv=None,
                    riscv_flash=False,
                    riscv_cache=False,
                    fast_fifo=False,
                    input_csv=False,
                    input_chan=input_chan[start_layer],
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
                    overwrite_ok or streaming[ll],
                    no_error_stop,
                    mlator=mlator if ll == final_layer else False,
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
                overwrite_ok or streaming[ll],
                no_error_stop,
                mlator=mlator if ll == final_layer else False,
                max_count=max_count,
                write_gap=write_gap[ll],
                final_layer=final_layer,
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

        if next_sequence[ll] == -1:
            break
        ll = next_sequence[ll]

    data = data_buf[-1]

    if not block_mode:
        with open(os.path.join(base_directory, test_name, filename), mode=filemode) as memfile:
            apb.set_memfile(memfile)

            if softmax or embedded_code:
                apb.unload(
                    output_processor_map[final_layer],
                    out_size,
                    out_offset[final_layer],
                    out_expand[final_layer],
                    out_expand_thresh[final_layer],
                    output_width[final_layer],
                    mlator=mlator,
                    write_gap=write_gap[final_layer],
                )

            if softmax:
                apb.softmax_layer(
                    output_width=output_width[final_layer],
                )

            summary_stats = '/*\n' + \
                            stats.summary(factor=repeat_layers, debug=debug, spaces=2,
                                          weights=kernel, w_size=quantization, bias=bias,
                                          group_bias_max=group_bias_max) + \
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
        output_count = output_chan[final_layer] \
            * output_dim[final_layer][0] * output_dim[final_layer][1]
        insert = summary_stats + \
            '\n/* Number of outputs for this network */\n' \
            f'#define CNN_NUM_OUTPUTS {output_count}'
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

    print(stats.summary(factor=repeat_layers, debug=debug,
                        weights=kernel, w_size=quantization, bias=bias,
                        group_bias_max=group_bias_max))

    return test_name
