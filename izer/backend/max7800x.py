###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Backend for MAX7800X embedded code generation and RTL simulations
"""
import copy
import hashlib
import os
import sys
from typing import List, Tuple

import numpy as np

from izer import (apbaccess, assets, compute, console, datamem, kbias, kdedup, kernels, latency,
                  load, op, rtlsim, state, stats)
from izer import tornadocnn as tc
from izer.eprint import eprint, nprint, wprint
from izer.names import layer_pfx, layer_str
from izer.simulate import (conv1d_layer, conv2d_layer, convtranspose2d_layer, eltwise_layer,
                           passthrough_layer, pooling_layer, print_data, show_data)
from izer.utils import ffs, fls, overlap, plural, popcount

from . import backend


class Backend(backend.Backend):
    """
    Backend for MAX7800X CNN network code generation
    """

    def create_net(self) -> str:  # pylint: disable=too-many-locals,too-many-branches
        """
        Chain multiple CNN layers, create and save input and output
        """
        # Cache variables locally for faster access
        activation = state.activation
        allow_streaming = state.allow_streaming
        apb_base = state.apb_base
        api_filename = state.api_filename
        avg_pool_rounding = state.avg_pool_rounding
        base_directory = state.base_directory
        bias = state.bias
        bias_group_map = state.bias_group_map
        big_data = state.big_data
        block_mode = state.block_mode
        board_name = state.board_name
        bypass = state.bypass
        c_filename = state.c_filename
        calcx4 = state.calcx4
        compact_data = state.compact_data
        conv_groups = state.conv_groups
        data = state.data
        debug_new_streaming = state.debug_new_streaming
        debug_snoop = state.debug_snoop
        dilation = state.dilation
        eltwise = state.eltwise
        embedded_code = state.embedded_code
        ext_rdy = state.ext_rdy
        avgpool_reset_layer = state.avgpool_reset_layer
        fast_fifo = state.fast_fifo
        fast_fifo_quad = state.fast_fifo_quad
        fifo = state.fifo
        final_layer = state.final_layer
        first_layer_used = state.first_layer_used
        flatten = state.flatten
        forever = state.forever
        ignore_bias_groups = state.ignore_bias_groups
        in_offset = state.in_offset
        in_sequences = state.in_sequences
        increase_delta1 = state.increase_delta1
        increase_delta2 = state.increase_delta2
        increase_start = state.increase_start
        init_tram = state.init_tram
        input_chan = state.input_channels
        input_channel_skip = state.input_channel_skip
        input_csv = state.input_csv
        input_dim = state.input_dim
        input_skip = state.input_skip
        kernel = state.weights
        kernel_size = state.kernel_size
        layers = state.layers
        legacy_test = state.legacy_test
        link_layer = state.link_layer
        log = state.log
        log_filename = state.log_filename
        log_intermediate = state.log_intermediate
        log_pooling = state.log_pooling
        measure_energy = state.measure_energy
        next_sequence = state.next_sequence
        no_error_stop = state.no_error_stop
        oneshot = state.oneshot
        operands = state.operands
        operator = state.operator
        out_offset = state.out_offset
        output_chan = state.output_channels
        output_dim = state.output_dim
        output_size = list(zip(output_chan, (output_dim[x][0] for x in range(len(output_dim))),
                                            (output_dim[x][1] for x in range(len(output_dim)))))
        output_filename = state.output_filename
        output_layer = state.output_layer
        output_padding = state.output_padding
        output_processor_map = state.output_processor_map
        output_shift = state.output_shift
        output_width = state.output_width
        override_delta1 = state.override_delta1
        override_delta2 = state.override_delta2
        override_rollover = state.override_rollover
        override_start = state.override_start
        overwrite = state.overwrite
        overwrite_ok = state.overwrite_ok
        padding = state.padding
        pool = state.pool
        pool_average = state.pool_average
        pool_dilation = state.pool_dilation
        pool_first = state.pool_first
        pool_stride = state.pool_stride
        pooled_dim = state.pooled_dim
        powerdown = state.powerdown
        prefix = state.prefix
        pretend_zero_sram = state.pretend_zero_sram
        prev_sequence = state.prev_sequence
        processor_map = state.processor_map
        quantization = state.quantization
        rd_ahead = state.read_ahead
        repeat_layers = state.repeat_layers
        reshape_inputs = state.reshape_inputs
        riscv = state.riscv
        riscv_cache = state.riscv_cache
        riscv_flash = state.riscv_flash
        simple1b = state.simple1b
        simulated_sequence = state.simulated_sequence
        snoop = state.snoop
        snoop_sequence = state.snoop_sequence
        start_layer = state.start_layer
        stopstart = state.stopstart
        streaming = state.streaming
        stride = state.stride
        tcalc = state.tcalc
        timeout = state.timeout
        timer = state.timer
        verbose = state.verbose
        verify_kernels = state.verify_kernels
        verify_writes = state.verify_writes
        weight_filename = state.weight_filename
        write_gap = state.write_gap
        write_zero_regs = state.write_zero_regs
        zero_sram = state.zero_sram
        zero_unused = state.zero_unused

        if not os.path.isdir('assets'):
            eprint('The assets folder is missing from the current directory.')

        assert tc.dev is not None
        device = tc.dev.device

        in_expand = [0] * layers
        in_expand_invol = [0] * layers
        out_expand = [0] * layers
        in_expand_thresh = [0] * layers
        out_expand_thresh = [0] * layers
        tram_max = [0] * layers
        timeslots = [1] * layers
        hw_padding = padding.copy()

        input_dim_str = [None] * layers
        output_dim_str = [None] * layers
        kernel_size_str = [None] * layers
        pool_str = [None] * layers
        padding_str = [None] * layers
        pool_stride_str = [None] * layers
        pool_dilation_str = [None] * layers
        dilation_str = [None] * layers
        stride_str = [None] * layers
        stream_buf = [None] * layers

        out_ignore = [0] * layers
        out_pad = [0] * layers

        hw_add_layers = [0] * layers
        hw_flatten = [False] * layers
        flatten_prod = [0] * layers
        sum_hw_layers = 0

        rollover = [None] * layers

        all_outputs_map = None

        terminating_layer = final_layer
        for i, s in enumerate(simulated_sequence):
            if s == -1:
                terminating_layer = i
                break

        if zero_sram:
            state.rtl_preload = False

        if start_layer > 0 and not tc.dev.SUPPORT_LINK_LAYER:
            eprint('`--start-layer` is not supported on this device.')

        if start_layer > tc.dev.MAX_START_LAYER:
            eprint(f'`--start-layer` is set to {start_layer}, but the device only supports '
                   f'a maximum of {tc.dev.MAX_START_LAYER}.')

        if link_layer and not tc.dev.SUPPORT_LINK_LAYER:
            eprint('`--link-layer` is not supported on this device.')

        if any(rd_ahead) and not tc.dev.SUPPORT_READ_AHEAD:
            eprint('`readahead` is not supported on this device.')

        if any(calcx4) and not tc.dev.SUPPORT_CALCX4:
            eprint('`calcx4` is not supported on this device.')

        if state.pipeline and not tc.dev.SUPPORT_PIPELINE:
            eprint('`--pipeline` is not supported on this device.')

        if state.pll and not tc.dev.SUPPORT_PLL:
            eprint('`--pll` is not supported on this device.')

        if state.fifo_go and not tc.dev.SUPPORT_FIFO_GO:
            eprint('`--fifo-go` is not supported on this device.')

        if snoop is not None and not tc.dev.SUPPORT_SNOOP:
            eprint('`snoop` is not supported on this device.')

        if oneshot and not tc.dev.SUPPORT_ONESHOT:
            eprint('`--one-shot` is not supported on this device.')

        if state.pipeline is None:  # Turn the pipeline on by default
            state.pipeline = tc.dev.SUPPORT_PIPELINE
        pipeline = state.pipeline  # Cache

        if state.pll is None:  # Turn the PLL on by default
            state.pll = tc.dev.SUPPORT_PLL

        if not state.balance_power and not state.pll:
            eprint('`--max-speed` requires `--pll` or `--pipeline`.')

        clock_speed = tc.dev.PLL_SPEED if state.pll else tc.dev.APB_SPEED
        if state.clock_divider is None:
            if pipeline:
                state.clock_divider = 1
            else:
                # Pick smallest working clock divider
                cdiv = (clock_speed + tc.dev.MAX_NO_PIPELINE_SPEED - 1) \
                    // tc.dev.MAX_NO_PIPELINE_SPEED
                # Round up to the next power of 2
                cdiv -= 1
                cdiv |= cdiv >> 1
                cdiv |= cdiv >> 2
                cdiv |= cdiv >> 4
                cdiv |= cdiv >> 8
                cdiv += 1
                state.clock_divider = cdiv

        if clock_speed // state.clock_divider > tc.dev.MAX_NO_PIPELINE_SPEED and not pipeline:
            wprint(f'For a CNN clock speed of {clock_speed} MHz, the pipeline must be enabled.')
        elif clock_speed // state.clock_divider <= tc.dev.MAX_NO_PIPELINE_SPEED and pipeline:
            nprint(f'For a CNN clock speed of {clock_speed} MHz, the pipeline can be disabled.')
        if state.clock_divider > tc.dev.MAX_CNNCLKDIV:
            nprint(f'The clock divider of {state.clock_divider} exceeds the device maximum '
                   f'({tc.dev.MAX_CNNCLKDIV}).')

        if zero_sram or pretend_zero_sram:
            # Clear every seventh kernel so we can test the BIST
            for i, _ in enumerate(kernel):
                kernel[i][::7] = np.full(shape=kernel[i][0].shape, fill_value=0, dtype=np.int64)

        if state.result_output and (state.mlator or oneshot or stopstart):
            state.result_output = False
        result_output = state.result_output  # Cache

        if result_output:
            state.max_count = None

        if (state.rtl_preload or state.rtl_preload_weights or state.result_output) \
           and not tc.dev.SUPPORT_SIM_PRELOAD:
            eprint('`--rtl-preload` and `--result-output` are not supported on this device.')

        if embedded_code and any(calcx4) and not state.new_kernel_loader:
            wprint('Enabling --new-kernel-loader since calcx4 is used.')
            state.new_kernel_loader = True
            state.compact_weights = False

        if not state.new_kernel_loader and state.mexpress:
            if any(calcx4):
                wprint('Ignoring --mexpress since calcx4 is used.')
                state.mexpress = False
            else:
                state.compact_weights = True

        mexpress = state.mexpress
        compact_weights = state.compact_weights

        # Check streaming and FIFO constraints
        fifo_group = fast_fifo

        if not fifo and state.synthesize_input is not None:
            eprint('`--synthesize-input` requires `--fifo`')
        if big_data[start_layer] and state.synthesize_input is not None:
            eprint('`--synthesize-input` requires `data_format: HWC`')

        if fifo:
            if start_layer != 0:
                eprint('`--start_layer` must be 0 when using a FIFO.')

            if input_chan[start_layer] > 16 \
               or big_data[start_layer] and input_chan[start_layer] > 4:
                eprint('Using the FIFO is restricted to a maximum of 4 input channels (CHW) or '
                       f'16 channels (HWC); this input is using {input_chan[start_layer]} '
                       'channels.')
            if big_data[start_layer] and processor_map[start_layer] & ~0x0001000100010001 != 0 \
               or not big_data[start_layer] \
               and processor_map[start_layer] & ~0x000f000f000f000f != 0:
                eprint('The FIFO is restricted to processors 0, 16, 32, 48 (CHW) or '
                       '0-3, 16-19, 32-35, 48-51 (HWC).')
            if fast_fifo:
                if big_data[start_layer] and input_chan[start_layer] > 1:
                    eprint('Fast FIFO supports only a single CHW input channel; '
                           f'this test is using {input_chan[start_layer]} channels.')
                elif not big_data[start_layer] and input_chan[start_layer] > 4:
                    eprint('Fast FIFO supports up to four HWC input channels; '
                           f'this test is using {input_chan[start_layer]} channels.')
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

        for ll in range(min(tc.dev.MAX_STREAM_LAYERS + 1, layers)):
            if next_sequence[ll] != -1 and next_sequence[ll] != ll + 1 and streaming[ll]:
                eprint(f'{layer_pfx(ll)}`next_sequence` must be {layer_str(ll+1)} when '
                       f'using streaming. Currently configured: {layer_str(next_sequence[ll])}')

            if tc.dev.EMULATE_1X1_STREAMING and kernel_size[ll] == [1, 1] \
               and operator[ll] in [op.CONV2D, op.CONVTRANSPOSE2D] \
               and (streaming[ll] or prev_sequence[ll] >= 0 and streaming[prev_sequence[ll]]):
                nprint(f'{layer_pfx(ll)}Using 3x3 kernel hardware for layer with 1x1 kernel due '
                       'to streaming.')
                # Create 3x3 weights from 1x1 weights and emulate using 3x3 kernels
                weight33 = np.zeros((kernel[ll].shape[0], 3, 3), dtype=np.int64)
                weight33[:, 1, 1] = kernel[ll][:, 0, 0]
                kernel[ll] = weight33
                assert padding[ll] == [0, 0]
                padding[ll] = [1, 1]
                hw_padding[ll] = [1, 1]
                kernel_size[ll][0] = kernel_size[ll][1] = 3

            if not tc.dev.SUPPORT_STREAM_NONPAD_FINAL and streaming[ll] \
               and (next_sequence[ll] == -1 or not streaming[next_sequence[ll]]) \
               and (padding[ll][0] == 0 or padding[ll][1] == 0):
                eprint(f'{layer_pfx(ll)}Padding for the final streaming layer must not '
                       'be zero.')

            if not tc.dev.SUPPORT_STREAMING_PASSTHROUGH \
               and operator[ll] == op.NONE and streaming[ll]:
                eprint(f'{layer_pfx(ll)}Passthrough operations are not supported for streaming '
                       'layers.')

        mlator = state.mlator

        if state.softmax and output_width[terminating_layer] == 8:
            wprint('--softmax should only be used with `output_width: 32`.')

        if fast_fifo and not riscv:
            eprint('--fast-fifo requires --riscv')

        if state.sleep and not riscv:
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

        for i, e in enumerate(quantization):
            if e is None:
                quantization[i] = 0  # Only in unused layers

        binary_quantization = any(quantization[ll] == -1 for ll in range(first_layer_used, layers))
        # Check we're not using binary weights on devices that don't support it
        if binary_quantization and not tc.dev.SUPPORT_BINARY_WEIGHTS:
            eprint('Binary weights (-1/+1) are not supported on this device.')

        # Account for extra transparently inserted hardware layers
        for ll in range(0, layers):
            if avgpool_reset_layer[ll]:
                sum_hw_layers += 1
            hw_add_layers[ll] = sum_hw_layers

        if repeat_layers * (final_layer + sum_hw_layers) > tc.dev.MAX_LAYERS:
            rep = '' if repeat_layers == 1 else f'When repeating {repeat_layers} times, '
            eprint(f'{rep}The adjusted layer count ({final_layer + sum_hw_layers}) '
                   f'exceeds the device maximum ({tc.dev.MAX_LAYERS}).')

        hw_operator = operator.copy()
        hw_input_dim = copy.deepcopy(input_dim)
        hw_pooled_dim = copy.deepcopy(pooled_dim)
        hw_output_dim = copy.deepcopy(output_dim)
        hw_kernel_size = copy.deepcopy(kernel_size)
        hw_kernel = copy.deepcopy(kernel)
        hw_dilation = copy.deepcopy(dilation)

        # Check that input channels are in separate memory instances if CHW (big) data format is
        # used, and calculate input and output expansion
        for ll in range(first_layer_used, layers):
            if quantization[ll] == 1 and binary_quantization:
                eprint(f'{layer_pfx(ll)}Cannot combine binary quantization with '
                       '1-bit quantization.')
            if output_shift[ll] is None:
                output_shift[ll] = 0 if not bypass[ll] else 7  # Set default

            if output_shift[ll] < -15 or output_shift[ll] > 15:
                implicit_shift = 8 - abs(quantization[ll]) if not bypass[ll] else 0
                eprint(f'{layer_pfx(ll)}{abs(quantization[ll])}-bit weight '
                       f'quantization supports an output_shift range of [{-15 - implicit_shift}, '
                       f'+{15 - implicit_shift}]. The specified value of output_shift is '
                       f'{output_shift[ll] - implicit_shift} which exceeds the system limits.')

            if big_data[ll]:
                p = processor_map[ll] >> (ffs(processor_map[ll]) & ~(tc.dev.P_SHARED-1))
                while p:
                    if popcount(p & (tc.dev.P_SHARED-1)) > 1:
                        eprint(f'{layer_pfx(ll)}CHW input format, but multiple '
                               'channels share the same memory instance. Modify the processor '
                               'map.')
                    p >>= tc.dev.P_SHARED

            out_expand[ll] = (output_chan[ll] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
            out_expand_thresh[ll] = (output_chan[ll] + out_expand[ll]-1) // out_expand[ll]
            if output_chan[ll] > tc.dev.MAX_PROC:
                out_expand_thresh[ll] = \
                    min((out_expand_thresh[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                        tc.dev.MAX_PROC)
            in_expand[ll] = (input_chan[ll] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
            if tcalc[ll] is None:
                tcalc[ll] = rd_ahead[ll] and in_expand[ll] > 1  # Set default
            in_expand_invol[ll] = (in_expand[ll] + 3) & ~3 if tcalc[ll] else in_expand[ll]
            in_expand_thresh[ll] = (input_chan[ll] + in_expand[ll] - 1) // in_expand[ll]

            if input_chan[ll] > tc.dev.MAX_PROC:
                in_expand_thresh[ll] = \
                    min((in_expand_thresh[ll] + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                        tc.dev.MAX_PROC)

            assert input_dim[ll][0] * input_dim[ll][1] * in_expand[ll] < tc.dev.FRAME_SIZE_MAX

            # Data memory size check - 4 channels share one instance unless CHW format
            in_size = input_dim[ll][0] * input_dim[ll][1] * in_expand[ll] * operands[ll] \
                * (1 if big_data[ll] else 4)
            if not streaming[ll] and in_size + in_offset[ll] > tc.dev.INSTANCE_WIDTH*16:
                eprint(f'{layer_pfx(ll)}{1 if big_data[ll] else 4} '
                       f'channel{"s" if not big_data[ll] else ""}/word {input_dim[ll][0]}x'
                       f'{input_dim[ll][1]} input (size {in_size}) '
                       f'with input offset 0x{in_offset[ll]:04x} and expansion {in_expand[ll]}x '
                       f'exceeds data memory instance size of {tc.dev.INSTANCE_WIDTH*16}.')

            if operator[ll] != op.CONV1D:
                input_dim_str[ll] = f'{input_dim[ll][0]}x{input_dim[ll][1]}'
                output_dim_str[ll] = f'{output_dim[ll][0]}x{output_dim[ll][1]}'
                kernel_size_str[ll] = f'{kernel_size[ll][0]}x{kernel_size[ll][1]}'
                pool_str[ll] = f'{pool[ll][0]}x{pool[ll][1]}' \
                    if pool[ll][0] > 1 or pool[ll][1] > 1 else '0x0'
                padding_str[ll] = f'{padding[ll][0]}/{padding[ll][1]}'
                pool_stride_str[ll] = f'{pool_stride[ll][0]}/{pool_stride[ll][1]}'
                pool_dilation_str[ll] = f'{pool_dilation[ll][0]}/{pool_dilation[ll][1]}'
                dilation_str[ll] = f'{dilation[ll][0]}/{dilation[ll][1]}'
                stride_str[ll] = f'{stride[ll][0]}/{stride[ll][1]}'
            else:
                input_dim_str[ll] = f'{input_dim[ll][0]}'
                output_dim_str[ll] = f'{output_dim[ll][0]}'
                kernel_size_str[ll] = f'{kernel_size[ll][0]}'
                pool_str[ll] = f'{pool[ll][0]}' \
                    if pool[ll][0] > 1 or pool[ll][1] > 1 else '0'
                padding_str[ll] = f'{padding[ll][0]}'
                pool_stride_str[ll] = f'{pool_stride[ll][0]}'
                pool_dilation_str[ll] = f'{pool_dilation[ll][0]}'
                dilation_str[ll] = f'{dilation[ll][0]}'
                stride_str[ll] = f'{stride[ll][0]}'

                if operands[ll] > 1:
                    eprint(f'{layer_pfx(ll)}Element-wise operations cannot be '
                           'combined with Conv1d.')

                if not tc.dev.SUPPORT_MULTIPASS_PADDED_CONV1D and padding[ll][0] > 0 \
                   and in_expand[ll] > 1:
                    eprint(f'{layer_pfx(ll)}This device does not support padded Conv1d '
                           'with input expansion > 1.', error=not state.ignore_hw_limits)

            if operator[ll] == op.NONE and next_sequence[ll] != -1 \
               and operator[next_sequence[ll]] == op.NONE \
               and not any(ll in c if c is not None else False for c in in_sequences):
                nprint(f'{layer_pfx(ll)}Passthrough layer is followed by passthrough '
                       f'layer {layer_str(next_sequence[ll])}. '
                       'These layers could potentially be combined.')
                if operands[ll] > 0 and pool[ll][0] == 1 and pool[ll][1] == 1 \
                   and (pool[next_sequence[ll]][0] > 1 or pool[next_sequence[ll]][1] > 1) \
                   and operands[next_sequence[ll]] == 1:
                    nprint('Use `pool_first: False` to combine element-wise and pooling layers '
                           'where pooling is executed after the element-wise operation.')

            if not pool_first[ll] and operands[ll] > tc.dev.MAX_POOL_LAST_ELEMENTS \
               and (pool[ll][0] > 1 or pool[ll][1] > 1):
                eprint(f'"pool last" supports a maximum of {tc.dev.MAX_POOL_LAST_ELEMENTS} '
                       'element-wise operands on this device.')

            if dilation[ll][0] > 1:
                if operator[ll] != op.CONV1D:
                    eprint(f'{layer_pfx(ll)}`dilation` > 1 is supported for Conv1d only.')

                if kernel_size[ll][0] == 1:
                    eprint(f'{layer_pfx(ll)}Kernel length must be greater than 1 to use '
                           '`dilation` > 1.')

                if (kernel_size[ll][0] - 1) * dilation[ll][0] < 9:
                    # Stretch existing kernel if we can
                    # 0 1 2 --> 0 X X X 1 X X X 2
                    kzeros = []
                    for s in range(1, kernel_size[ll][0]):
                        kzeros += [s] * (dilation[ll][0] - 1)
                    k = np.insert(kernel[ll], kzeros, 0, axis=1)
                    hw_kernel[ll] = k
                    hw_kernel_size[ll] = [k.shape[1], 1]
                elif kernel_size[ll][0] <= tc.dev.MAX_DILATION_1D_KERNEL:
                    # Use Conv2d
                    if pool[ll][0] != 1:
                        eprint(f'{layer_pfx(ll)}Pooling must be 1 to use `dilation` > 4.')
                    if padding[ll][0] > tc.dev.MAX_DILATION_1D_PAD:
                        eprint(f'{layer_pfx(ll)}Padding must be '
                               f'{tc.dev.MAX_DILATION_1D_PAD} or smaller to use `dilation` > 4.')
                    if operands[ll] != 1:
                        eprint(f'{layer_pfx(ll)}Operands must be 1 to use `dilation` > 4.')
                    if bypass[ll] or flatten[ll] or rd_ahead[ll] or streaming[ll]:
                        eprint(f'{layer_pfx(ll)}`bypass`, `flatten`, `rd_ahead`, '
                               '`streaming` must be False to use `dilation` > 4.')
                    if dilation[ll][0] > tc.dev.MAX_DILATION_1D:
                        eprint(f'{layer_pfx(ll)}`dilation` must be '
                               f'{tc.dev.MAX_DILATION_1D} or smaller for Conv1d operations.')

                    nprint(f'{layer_pfx(ll)}Using Conv2d hardware for dilated Conv1d.')
                    # Use the Conv1d hardware with 1 pad on 'dilation' columns using 3x3 kernels
                    hw_operator[ll] = op.CONV2D
                    hw_input_dim[ll][0] = (input_dim[ll][0] + dilation[ll][0] - 1) \
                        // dilation[ll][0]
                    hw_input_dim[ll][1] = dilation[ll][0]
                    hw_pooled_dim[ll] = hw_input_dim[ll]
                    hw_output_dim[ll] = hw_pooled_dim[ll]
                    hw_padding[ll] = [1, 1]
                    hw_kernel_size[ll] = [3, 3]
                    hw_dilation[ll] = [1, 1]
                    # 2D output size is equal to the 2D input size since the pad is fixed to 1.
                    # Subtract the original output dimensions to calculate the overage.
                    out_pad[ll] = hw_input_dim[ll][0] * hw_input_dim[ll][1] - output_dim[ll][0]

                    # Create 3x3 kernel from 3x1 kernel -- move original into center column
                    k = np.insert(kernel[ll].reshape(output_chan[ll],
                                                     input_chan[ll] // conv_groups[ll],
                                                     kernel_size[ll][0], -1),
                                  [0, 1], 0, axis=3)
                    if kernel_size[ll][0] == 2:
                        k = np.insert(k, 0, 0, axis=2)  # Insert at top - throw away the padding
                    elif kernel_size[ll][0] == 1:
                        k = np.insert(k, [0, 1], 0, axis=2)  # Use center
                    else:  # 3
                        out_ignore[ll] = 4 * dilation[ll][0] * out_expand[ll]
                    assert k.shape[2] == k.shape[3] == 3
                    hw_kernel[ll] = k.reshape(-1, k.shape[2], k.shape[3])

                    if out_offset[ll] < out_ignore[ll]:
                        eprint(f'{layer_pfx(ll)}`out_offset` used with dilation of '
                               f'{dilation[ll][0]} must be at least {out_ignore[ll]:04x}.')
                else:
                    eprint(f'{layer_pfx(ll)}Kernel length must be '
                           f'{tc.dev.MAX_DILATION_1D_KERNEL} or smaller to use `dilation` of '
                           f'{dilation[ll][0]}.')

            out_size = (output_dim[ll][0] * output_dim[ll][1] + out_pad[ll]) * out_expand[ll] \
                * 4 * output_width[ll] // 8
            if (not streaming[ll] or ll == terminating_layer) \
               and out_size + out_offset[ll] > tc.dev.INSTANCE_WIDTH*16:
                eprint(f'{layer_pfx(ll)}HWC (4 channels/word) '
                       f'{output_width[ll]}-bit {output_dim[ll][0]}x'
                       f'{output_dim[ll][1]} output (size {out_size}) '
                       f'with output offset 0x{out_offset[ll]:04x} and expansion '
                       f'{out_expand[ll]}x '
                       f'exceeds data memory instance size of {tc.dev.INSTANCE_WIDTH*16}.')

            if hw_operator[ll] == op.NONE:
                if activation[ll] is not None:
                    eprint(f'{layer_pfx(ll)}Pass-through layers must not use activation.')
                if padding[ll][0] != 0 or padding[ll][1] != 0:
                    eprint(f'{layer_pfx(ll)}Padding must be zero for passthrough layers.')
                if output_shift[ll] != 0 and output_shift[ll] is not None:
                    eprint(f'{layer_pfx(ll)}`output_shift` must be zero for passthrough '
                           'layers.')
                if (pool[ll][0] > 1 or pool[ll][1] > 1) \
                   and in_expand[ll] > tc.dev.MAX_POOL_PASSES \
                   and (hw_pooled_dim[ll][0] > 1 or hw_pooled_dim[ll][1] > 1):
                    eprint(f'{layer_pfx(ll)}pooling in passthrough layer uses '
                           f'{in_expand[ll]} {plural(in_expand[ll], "pass", "es")}, '
                           f'which exceeds the maximum of {tc.dev.MAX_POOL_PASSES} '
                           'on this device.')

                tram_max[ll] = 1
            else:
                if hw_operator[ll] == op.CONVTRANSPOSE2D:
                    # Flip padding around to match PyTorch conventions for ConvTranspose2d
                    hw_padding[ll] = (
                        hw_dilation[ll][0] * (hw_kernel_size[ll][0] - 1) - hw_padding[ll][0],
                        hw_dilation[ll][1] * (hw_kernel_size[ll][1] - 1) - hw_padding[ll][1]
                    )
                    if hw_padding[ll][0] not in tc.dev.SUPPORTED_X2D_PADS \
                       or hw_padding[ll][1] not in tc.dev.SUPPORTED_X2D_PADS:
                        eprint(f'{layer_pfx(ll)}The selected padding ({padding[ll]}) for '
                               'ConvTranspose2d is not supported on this device.')
                    if output_padding[ll][0] not in tc.dev.SUPPORTED_X2D_OUTPUT_PADS \
                       or output_padding[ll][1] not in tc.dev.SUPPORTED_X2D_OUTPUT_PADS:
                        eprint(f'{layer_pfx(ll)}The selected output padding '
                               f'({output_padding[ll]}) for ConvTranspose2d is not supported '
                               'on this device.')
                    tram_max[ll] = max(0, (hw_pooled_dim[ll][1] - 1) * stride[ll][1] + 1
                                       + output_padding[ll][1] + 2 * hw_padding[ll][1]
                                       - hw_kernel_size[ll][1]) + 1
                else:
                    tram_max[ll] = max(0, hw_pooled_dim[ll][1] + 2 * hw_padding[ll][1]
                                       - hw_kernel_size[ll][1]) + 1

            if hw_operator[ll] != op.CONVTRANSPOSE2D and (output_padding[ll][0] != 0
                                                          or output_padding[ll][1] != 0):
                eprint(f'{layer_pfx(ll)}Output padding must be 0 for this operator.')

            if input_chan[ll] % conv_groups[ll] != 0 or output_chan[ll] % conv_groups[ll] != 0:
                eprint(f'{layer_pfx(ll)}convolution groups ({conv_groups[ll]}) does not '
                       f'divide the input channels ({input_chan[ll]}) or '
                       f'output channels ({output_chan[ll]}).')

            if flatten[ll] and hw_operator[ll] == op.NONE:
                eprint(f'{layer_pfx(ll)}`flatten` is not compatible with passthrough '
                       'layers.')

            if flatten[ll] and (pool[ll][0] > 1 or pool[ll][1] > 1):
                eprint(f'{layer_pfx(ll)}`flatten` is not compatible with pooling.')

            if flatten[ll] and streaming[ll]:
                eprint(f'{layer_pfx(ll)}`flatten` is not compatible with streaming.')

            if conv_groups[ll] > 1:
                if not tc.dev.SUPPORT_DEPTHWISE:
                    eprint(f'{layer_pfx(ll)}Convolution groups ({conv_groups[ll]}) > 1 are not '
                           f' supported on this device.')
                if conv_groups[ll] != input_chan[ll] or conv_groups[ll] != output_chan[ll]:
                    eprint(f'{layer_pfx(ll)}Convolution groups ({conv_groups[ll]}) must be equal '
                           f'to the number of input channels ({input_chan[ll]}), and output '
                           f'channels ({output_chan[ll]}) must be equal to input channels.')
                if flatten[ll]:
                    eprint(f'{layer_pfx(ll)}Convolution groups ({conv_groups[ll]}) > 1 are not '
                           'supported when flattening.')
                if bias_group_map[ll] is not None:
                    eprint(f'{layer_pfx(ll)}`bias_group` is not supported for depth-wise layers.')
                # if output_width[ll] != 8:
                #     eprint(f'{layer_pfx(ll)}convolution groups ({conv_groups[ll]}) > 1 are not'
                #            f' supported when using `wide` output.')

            if input_skip[ll] != 0 and not tc.dev.SUPPORT_MULTIPASS_STRIDE:
                eprint(f'{layer_pfx(ll)}`read_gap` must be 0 for this device.')
            if input_skip[ll] != 0 and in_expand[ll] > 1 \
               and not tc.dev.SUPPORT_MULTIPASS_READ_STRIDE:
                eprint(f'{layer_pfx(ll)}`read_gap` must be 0 when using more than 64 channels or '
                       'multi-pass on this device.')

            # Conv1d pool_dilation
            if pool_dilation[ll][0] < 1 or pool_dilation[ll][1] < 1 \
               or pool_dilation[ll][0] > tc.dev.MAX_POOL_DILATION \
               or pool_dilation[ll][1] > tc.dev.MAX_POOL_DILATION:
                eprint(f'{layer_pfx(ll)}`pool_dilation` values must be 1 or greater, and '
                       f'{tc.dev.MAX_POOL_DILATION} or smaller on this device.')

            if in_sequences[ll] is not None:
                or_map = 0
                and_map = ~or_map
                for i, lt in enumerate(in_sequences[ll]):
                    emap = processor_map[0] if lt == -1 else output_processor_map[lt]
                    or_map |= emap
                    and_map &= emap
                if or_map != processor_map[ll]:
                    wprint(f'{layer_pfx(ll)}The combination of the `in_sequences` '
                           '`output_processor_map`s does not combine to `processor_map`.')
                if and_map != processor_map[ll]:
                    # Concatenate channels sequentially
                    if and_map != 0:
                        wprint(f'{layer_pfx(ll)}Channel concatenation has `output_processor_map` '
                               f'overlap for `in_sequences` {in_sequences[ll]}.')
                    for i, lt in enumerate(in_sequences[ll]):
                        if lt != -1 and in_offset[ll] != out_offset[lt]:
                            wprint(f'{layer_pfx(ll)}`in_offset` (0x{in_offset[ll]:04x}, for '
                                   f'input #{i}) does not match `out_offset` '
                                   f'(0x{out_offset[lt]:04x}) for layer {layer_str(lt)}.')
                else:
                    # Channel-wise concatenation or element-wise or interleaved concatenation
                    offs = 0
                    for i, lt in enumerate(in_sequences[ll]):
                        if lt != -1 and in_offset[ll] + 4 * i != out_offset[lt] \
                           and in_offset[ll] + offs != out_offset[lt]:
                            wprint(f'{layer_pfx(ll)}`in_offset` (0x{in_offset[ll]:04x}, for '
                                   f'input #{i}) does not match `out_offset` '
                                   f'(0x{out_offset[lt]:04x}) for layer {layer_str(lt)}.')
                        offs += 4 * output_dim[lt][0] * output_dim[lt][1]

                if operands[ll] == 1:  # cat
                    if write_gap[ll] == 0:
                        min_proc = -1
                        max_proc = -1
                        for lt in in_sequences[ll]:
                            first_proc = ffs(processor_map[0]) if lt == -1 \
                                else ffs(output_processor_map[lt])
                            last_proc = fls(processor_map[0]) if lt == -1 \
                                else fls(output_processor_map[lt])
                            if first_proc < min_proc:
                                wprint(f'{layer_pfx(ll)}In `in_sequences` {in_sequences[ll]}, '
                                       'an earlier layer in the sequence uses a higher first '
                                       f'processor ({min_proc}) than layer {layer_str(lt)} which '
                                       f'uses processor {first_proc}.')
                            if last_proc < max_proc:
                                wprint(f'{layer_pfx(ll)}In `in_sequences` {in_sequences[ll]}, '
                                       'an earlier layer in the sequence uses a higher last '
                                       f'processor ({max_proc}) than layer {layer_str(lt)} which '
                                       f'uses processor {last_proc}.')
                            min_proc = first_proc
                            max_proc = last_proc
                else:  # eltwise
                    eltwise_proc_map = 0
                    for lt in in_sequences[ll]:
                        emap = processor_map[0] if lt == -1 else output_processor_map[lt]
                        if eltwise_proc_map not in (0, emap):
                            eprint(f'{layer_pfx(ll)}In `in_sequences` {in_sequences[ll]}, '
                                   'an earlier layer in the sequence uses a different output '
                                   f'processor map (0x{eltwise_proc_map:016x}) than layer '
                                   f'{layer_str(lt)} which uses 0x{emap:016x}.')
                        eltwise_proc_map = emap

                # Merge the output of all processors of all input sequence members
                emap = 0
                for lt in in_sequences[ll]:
                    emap |= processor_map[0] if lt == -1 else output_processor_map[lt]
                # Check that all out input processors have data from somewhere in the merged map
                if processor_map[ll] & emap != processor_map[ll]:
                    wprint(f'{layer_pfx(ll)}The processor map {processor_map[ll]:016x} specifies '
                           'processors that have no data from any of the input sequences '
                           f'{in_sequences[ll]}.')

            if output_width[ll] != 8 and hw_operator[ll] == op.NONE:
                eprint(f'{layer_pfx(ll)}'
                       'The passthrough operator requires an output width of 8.')

        # Deduplicate kernels
        # Do this here since by now all modifications to the kernels have happened
        kernel_ptrs: List[int] = []  # Indirection for hw_kernel
        bias_ptrs: List[int] = []
        if state.deduplicate_weights:
            kernel_ptrs, hw_kernel = kdedup.deduplicate(
                hw_kernel,
                layers,
                quantization,
                processor_map,
                kind='kernels',
            )
            bias_ptrs, bias = kdedup.deduplicate(
                bias,
                layers,
                quantization,
                processor_map,
                kind='bias',
            )
        else:
            kernel_ptrs = list(range(len(hw_kernel)))
            bias_ptrs = list(range(len(bias)))
        state.weights = hw_kernel
        state.bias = bias

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
            if not overwrite:
                eprint('The target folder', target_dir, 'exists. Use --overwrite to proceed.')
            else:
                nprint('--overwrite specified, writing to', target_dir, 'even though it exists.')

        # Redirect stdout?
        if log:
            state.output_is_console = False
            sys.stdout = open(
                os.path.join(base_directory, test_name, log_filename),
                mode='w',
                encoding='utf-8',
            )
            print(f'{" ".join(str(x) for x in sys.argv)}')
            print(f'{tc.dev.partnum}\n')
            print(f'{test_name}')

        if block_mode:
            filename = state.input_filename + '.mem'
        else:
            filename = c_filename + ('_riscv' if riscv else '') + '.c'
        if not block_mode and (embedded_code or compact_data):
            sampledata_header = \
                open(
                    os.path.join(base_directory, test_name, state.sample_filename),
                    mode='w',
                    encoding='utf-8',
                )
            sampledata_header.write('// This file was @generated automatically\n\n')
            if state.generate_kat and state.result_filename is not None:
                sampleoutput_header = \
                    open(
                        os.path.join(base_directory, test_name, state.result_filename),
                        mode='w',
                        encoding='utf-8',
                    )
                sampleoutput_header.write('// This file was @generated automatically\n\n')
            else:
                sampleoutput_header = None
        else:
            sampledata_header = sampleoutput_header = None
        if not block_mode and not state.rtl_preload_weights:
            weight_header = \
                open(
                    os.path.join(base_directory, test_name, weight_filename),
                    mode='w',
                    encoding='utf-8',
                )
            weight_header.write('// This file was @generated automatically\n\n')
        else:
            weight_header = None

        # Calculate the groups needed, and groups and processors used overall
        processors_used = 0
        group_map = [None] * layers
        broadcast_mode = [False] * layers
        emulate_eltwise = [False] * layers
        for ll in range(first_layer_used, layers):
            bits = processor_map[ll]
            processors_used |= bits
            fl = ' (before flattening)' if flatten[ll] else ''

            if input_chan[ll] > tc.dev.MAX_CHANNELS:
                eprint(f'{layer_pfx(ll)}Configured for {input_chan[ll]} input channels{fl}, which '
                       f'exceeds the system maximum of {tc.dev.MAX_CHANNELS}.')
            if output_chan[ll] > tc.dev.MAX_CHANNELS:
                eprint(f'{layer_pfx(ll)}Configured for {output_chan[ll]} output channels, which '
                       f'exceeds the system maximum of {tc.dev.MAX_CHANNELS}.')
            if (ll != start_layer or not fast_fifo_quad) \
               and popcount(processor_map[ll]) != in_expand_thresh[ll]:
                eprint(f'{layer_pfx(ll)}{input_chan[ll]} input '
                       f'{plural(input_chan[ll], "channel")}{fl} '
                       f'using {in_expand[ll]} {plural(in_expand[ll], "pass", "es")}, '
                       f'and {operands[ll]} {plural(operands[ll], "operand")} '
                       f'({in_expand_thresh[ll]} processors '
                       f'per pass), but the enabled processor map 0x{processor_map[ll]:016x} '
                       f'has {popcount(processor_map[ll])} bits instead of the '
                       f'expected number of {in_expand_thresh[ll]}.')
            if ll == start_layer and fast_fifo_quad \
               and popcount(processor_map_0) != in_expand_thresh[ll]:
                eprint(f'{layer_pfx(ll)}{input_chan[ll]} input '
                       f'{plural(input_chan[ll], "channel")}{fl} '
                       f'using {in_expand[ll]} {plural(in_expand[ll], "pass", "es")} '
                       f'({in_expand_thresh[ll]} processors per pass), but the '
                       f'enabled processor map 0x{processor_map[ll]:016x} '
                       f'has {popcount(processor_map[ll])} bits instead of the '
                       f'expected number of {in_expand_thresh[ll]}.')
            if popcount(output_processor_map[ll]) != out_expand_thresh[ll]:
                eprint(f'{layer_pfx(ll)}{output_chan[ll]} output '
                       f'{plural(output_chan[ll], "channel")} using {out_expand[ll]} '
                       f'{plural(out_expand[ll], "pass", "es")} '
                       f'({out_expand_thresh[ll]} processors per pass), but the '
                       f'processor output map 0x{output_processor_map[ll]:016x} '
                       f'has {popcount(output_processor_map[ll])} bits instead of the '
                       f'expected number of {out_expand_thresh[ll]}.')
            this_map = []
            for group in range(tc.dev.P_NUMGROUPS):
                if (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
                    this_map.append(group)
            group_map[ll] = this_map

            # Ensure input and output map are the same for passthrough layers
            if hw_operator[ll] == op.NONE:
                for group in range(tc.dev.P_NUMGROUPS):
                    in_pro = 2**popcount(
                        (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                    ) - 1
                    out_pro = (output_processor_map[ll] >> group*tc.dev.P_NUMPRO) \
                        % 2**tc.dev.P_NUMPRO
                    if out_pro != 0:
                        out_pro >>= ffs(out_pro)
                    if out_pro != in_pro:
                        eprint(f'{layer_pfx(ll)}The output processors for a pass-through layer '
                               'must be a packed version of the input processors for each x16. '
                               f'Configured are: input {processor_map[ll]:016x}, output '
                               f'{output_processor_map[ll]:016x}.')

            # Ensure byte positions are the same in the input and output map for
            # depthwise convolutions
            if conv_groups[ll] > 1:
                if ffs(output_processor_map[ll]) % tc.dev.P_SHARED != 0:
                    eprint(f'{layer_pfx(ll)}Output processors for depth-wise convolutions '
                           'must be aligned to a multiple of 4. Configured for this layer: '
                           f'{output_processor_map[ll]:016x}.')
                if ffs(processor_map[ll]) % tc.dev.P_SHARED != 0 \
                   and (processor_map[ll] >> ffs(processor_map[ll])) // 2**tc.dev.P_NUMPRO > 0:
                    eprint(f'{layer_pfx(ll)}When spanning groups for depth-wise convolutions, '
                           'processors must be aligned to a multiple of 4. Configured for this '
                           f'layer: {processor_map[ll]:016x}.')
                if processor_map[ll] == output_processor_map[ll]:
                    broadcast_mode[ll] = True
                elif state.energy_warning:
                    nprint(f'{layer_pfx(ll)}depth-wise convolution moves data across processors. '
                           f'This has a performance impact. Input 0x{processor_map[ll]:016x}, '
                           f'output 0x{output_processor_map[ll]:016x}.')

            # Block certain element-wise operations when not using passthrough mode
            if tc.dev.EMULATE_ELTWISE_MP and operands[ll] > 1 and in_expand[ll] > 1 \
               and operands[ll] * in_expand[ll] != operands[ll] + in_expand[ll]:
                if hw_operator[ll] != op.NONE or pool[ll][0] > 1 or pool[ll][1] > 1 \
                   or pool_stride[ll][0] > 1 or pool_stride[ll][1] > 1:
                    eprint(f'{layer_pfx(ll)}The element-wise operation exceeds a multi-pass of 2 '
                           'and therefore does not support pooling or convolution.')
                emulate_eltwise[ll] = True

            # Warn if hidden layers use channel count that is not divisible by 4
            if ll != start_layer and input_chan[ll] % 4 != 0 and state.energy_warning:
                nprint(f'{layer_pfx(ll)}The input channel count ({input_chan[ll]}) is not '
                       'a multiple of 4. Best energy performance is achieved with multiples of 4.')

        groups_used = []
        for group in range(tc.dev.P_NUMGROUPS):
            if ((processors_used |
                 output_processor_map[final_layer]) >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO:
                groups_used.append(group)

        if 0 not in groups_used:
            eprint('Quadrant 0 is not used, this is currently unsupported.')

        for ll in range(first_layer_used, layers):
            if bias_group_map[ll] is not None:
                for e in bias_group_map[ll]:
                    if e not in groups_used:
                        eprint(f'{layer_pfx(ll)}`bias_quadrant` references unused quadrant {e}. '
                               f'Used x16 groups for this network are: {groups_used}.',
                               error=not ignore_bias_groups)

        # Create ARM code wrapper if needed
        if riscv and not block_mode:
            with open(
                os.path.join(base_directory, test_name, c_filename + '.c'),
                mode='w',
                encoding='utf-8',
            ) as f:
                apb = apbaccess.apbwriter(
                    f,
                    master=False,
                    riscv=False,
                    embedded_arm=embedded_code,
                    groups=list(set().union(groups_used)),
                    test_name=test_name,
                )
                apb.copyright_header()

                apb.output('// ARM wrapper code\n'
                           f'// {test_name}\n'
                           '// This file was @generated by '
                           f'{" ".join(str(x) for x in sys.argv)}\n\n')

                apb.header()
                apb.main()

        if input_csv is not None:
            csv = os.path.join(base_directory, test_name, input_csv)
        else:
            csv = None

        if embedded_code and api_filename.lower() != 'none':
            apifile = open(
                os.path.join(base_directory, test_name, api_filename),
                mode='w',
                encoding='utf-8',
            )
        else:
            apifile = None

        passfile = None
        if state.generate_kat and log_intermediate:
            memfile2 = open(os.path.join(base_directory, test_name,
                            f'{output_filename}.csv'),
                            mode='w', encoding='utf-8')
            if state.output_pass_filename is not None:
                passfile = open(os.path.join(base_directory, test_name,
                                f'{state.output_pass_filename}.csv'),
                                mode='w', encoding='utf-8')
            datafile = open(os.path.join(base_directory, test_name,
                            f'{state.output_data_filename}.npy'),
                            mode='wb')
            weightsfile = open(os.path.join(base_directory, test_name,
                               f'{state.output_weights_filename}.npy'),
                               mode='wb')
            biasfile = open(os.path.join(base_directory, test_name,
                            f'{state.output_bias_filename}.npy'),
                            mode='wb')
        else:
            memfile2 = None
            datafile = None
            weightsfile = None
            biasfile = None

        with open(os.path.join(base_directory, test_name, filename), mode='w',
                  encoding='utf-8') as memfile:
            apb = apbaccess.apbwriter(
                memfile,
                verify_writes=verify_writes,
                weight_header=weight_header,
                sampledata_header=sampledata_header,
                sampleoutput_header=sampleoutput_header,
                embedded_code=embedded_code,
                write_zero_registers=write_zero_regs,
                master=groups_used[0]
                if oneshot > 0 or stopstart or (apifile is not None) else False,
                riscv=True if riscv else None,
                fast_fifo=fast_fifo,
                input_chan=input_chan[start_layer],
                apifile=apifile,
                forever=forever,
                fifo=fifo,
                groups=list(set().union(groups_used)),
                oneshot=terminating_layer if oneshot else 0,
                num_classes=output_chan[terminating_layer],
                output_width=output_width[terminating_layer],
                bias=any(b is not None for b in bias),
                test_name=test_name,
            )

            apb.copyright_header()

            apb.output(f'// {test_name}\n'
                       '// This file was @generated by '
                       f'{" ".join(str(x) for x in sys.argv)}\n\n')
            if apifile is not None:
                apb.output(f'// {test_name}\n'
                           '// This file was @generated by '
                           f'{" ".join(str(x) for x in sys.argv)}\n\n'
                           '// DO NOT EDIT - regenerate this file instead!\n\n', True)

            # Human readable description of test
            apb.output(f'// Configuring {repeat_layers * layers} '
                       f'{plural(repeat_layers * layers, "layer")}\n'
                       f'// Input data: {"CHW" if big_data[first_layer_used] else "HWC"}\n',
                       embedded_code)

            for r in range(repeat_layers):
                for ll in range(first_layer_used, layers):
                    flatten_str = '' if not flatten[ll] else \
                        f' flattened to {input_chan[ll]*input_dim[ll][0]*input_dim[ll][1]}x1x1'
                    apb.output(f'// Layer {r * layers + ll}: '
                               f'{str(operands[ll])+"x" if operands[ll] > 1 else ""}'
                               f'{input_chan[ll]}x{input_dim_str[ll]}'
                               f'{" streaming" if streaming[ll] else ""}{flatten_str}, ',
                               embedded_code)
                    if not pool_first[ll] and operands[ll] > 1:
                        apb.output(f'{operands[ll]}-element {op.string(eltwise[ll], elt=True)}, ',
                                   embedded_code)
                    if pool[ll][0] > 1 or pool[ll][1] > 1:
                        apb.output(f'{"avg" if pool_average[ll] else "max"} pool {pool_str[ll]} '
                                   f'with stride {pool_stride_str[ll]}', embedded_code)
                        if pool_dilation[ll][0] > 1 or pool_dilation[ll][1] > 1:
                            apb.output(f' and dilation {pool_dilation_str[ll]}', embedded_code)
                    else:
                        apb.output('no pooling', embedded_code)
                    if pool_first[ll] and operands[ll] > 1:
                        apb.output(f', {operands[ll]}-element {op.string(eltwise[ll], elt=True)}',
                                   embedded_code)
                    if hw_operator[ll] != op.NONE:
                        conv_str = f', {op.string(operator[ll])}'
                        if operator[ll] != op.LINEAR:
                            conv_str += f' with kernel size {kernel_size_str[ll]}, ' \
                                        f'stride {stride_str[ll]}, ' \
                                        f'pad {padding_str[ll]}'
                        conv_str += f', {op.act_string(activation[ll])}, '
                        if dilation[ll][0] > 1 or dilation[ll][1] > 1:
                            conv_str += f'dilation {dilation_str[ll]}, '
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

            if state.input_fifo:
                apb.output('#define USE_FIFO\n')

            if embedded_code or compact_data or input_csv:
                # Pre-define data memory loader. Inline later when generating RTL sim.
                load.load(
                    True,
                    apb,
                    big_data[start_layer],
                    processor_map_0,
                    in_offset[start_layer],
                    [input_chan[start_layer], input_dim[start_layer][0],
                     input_dim[start_layer][1]],
                    in_expand[start_layer],
                    operands[start_layer],
                    in_expand_thresh[start_layer],
                    data,
                    hw_padding[start_layer],
                    csv_file=csv,
                )
            if not block_mode and (embedded_code or compact_weights):
                # Pre-define the kernels and bias values
                hw_kern_offs, hw_kern_len, hw_kern_count, hw_kern_ochan = kernels.load(
                    True,
                    apb,
                    layers,
                    hw_operator,
                    hw_kernel,
                    hw_kernel_size,
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
                    verify_kernels,
                    api=embedded_code,
                )
                hw_bias_offs, hw_bias_group, group_bias_max = kbias.load(
                    True,
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
                    list(set().union(groups_used)),
                    flatten,
                )

            apb.function_header(function='init')

            # Initialize CNN registers

            if verbose:
                print('\nGlobal registers:\n'
                      '-----------------')

            if tc.dev.REQUIRE_REG_CLEAR:
                val = 0 if not tc.dev.SUPPORT_PIPELINE or pipeline else 1 << 5
                for group in groups_used:
                    apb.write_ctl(group, tc.dev.REG_CTL, val | 1 << 3 | tc.dev.READY_SEL << 1,
                                  comment=' // Enable clocks', no_verify=True)
            # Reset
            apb.write_fifo_ctl(tc.dev.AON_CTL, tc.dev.AON_READY_SEL,
                               comment=' // AON control', force_write=True)

            if tc.dev.REQUIRE_REG_CLEAR:
                for group in groups_used:
                    apb.write_ctl(group, tc.dev.REG_SRAM, 0x40e,
                                  comment=' // SRAM control')
                bist_clear = tc.dev.BIST_ZERO_BOTH_EX if any(b is not None for b in bias) \
                    else tc.dev.BIST_ZERO_EX
                for group in groups_used:
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, bist_clear,
                                  comment=' // Clear registers', no_verify=True)
                for group in groups_used:
                    apb.wait_ctl(group, tc.dev.REG_SRAM_TEST,
                                 tc.dev.BIST_ZERO_WAIT, tc.dev.BIST_ZERO_WAIT,
                                 comment=' // Wait for clear')
                for group in groups_used:
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                                  comment=' // Reset BIST', force_write=True, no_verify=True)
                apb.output('\n', embedded_code)

            # Configure global control registers for used groups
            for group in groups_used:
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
                                       f'{tc.dev.TRAM_SIZE}); // Zero TRAM {group}\n',
                                       embedded_code)
                            apb.output('\n', embedded_code)

                # Stop state machine - will be overwritten later; enable FIFO
                val = tc.dev.READY_SEL << 1
                if fifo:
                    val |= 1 << 15
                val |= 1 << 3  # Enable clocks
                if mexpress:
                    val |= 1 << 20
                if tc.dev.SUPPORT_PIPELINE and not pipeline:
                    val |= 1 << 5
                apb.write_ctl(group, tc.dev.REG_CTL, val,
                              comment=' // Stop SM')
                # SRAM Control - does not need to be changed
                if not tc.dev.REQUIRE_REG_CLEAR:
                    apb.write_ctl(group, tc.dev.REG_SRAM, 0x40e,
                                  comment=' // SRAM control')
                # Number of layers and start layer
                val = (repeat_layers * (final_layer + sum_hw_layers)) \
                    | ((start_layer + hw_add_layers[start_layer]) << 8)
                apb.write_ctl(group, tc.dev.REG_LCNT_MAX, val,
                              comment=' // Layer count')

            if zero_sram:
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_DATA_EX,
                                  comment=' // Data SRAM BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.wait_ctl(group, tc.dev.REG_SRAM_TEST,
                                 tc.dev.BIST_DATA_WAIT, tc.dev.BIST_DATA_WAIT,
                                 comment=' // Wait for BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_DATA_ERR, 0,
                                   comment=' // Return on BIST error')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                                  comment=' // Reset BIST', force_write=True)
                apb.output('\n', embedded_code)
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_MASK_EX,
                                  comment=' // Mask SRAM BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.wait_ctl(group, tc.dev.REG_SRAM_TEST,
                                 tc.dev.BIST_MASK_WAIT, tc.dev.BIST_MASK_WAIT,
                                 comment=' // Wait for BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_MASK_ERR, 0,
                                   comment=' // Return on BIST error')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                                  comment=' // Reset BIST', force_write=True)
                apb.output('\n', embedded_code)
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_TRAM_EX,
                                  comment=' // Tornado SRAM BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.wait_ctl(group, tc.dev.REG_SRAM_TEST,
                                 tc.dev.BIST_TRAM_WAIT, tc.dev.BIST_TRAM_WAIT,
                                 comment=' // Wait for BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_TRAM_ERR, 0,
                                   comment=' // Return on BIST error')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                                  comment=' // Reset BIST', force_write=True)
                apb.output('\n', embedded_code)
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_BIAS_EX,
                                  comment=' // Bias Rfile BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.wait_ctl(group, tc.dev.REG_SRAM_TEST,
                                 tc.dev.BIST_BIAS_WAIT, tc.dev.BIST_BIAS_WAIT,
                                 comment=' // Wait for BIST')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.verify_ctl(group, tc.dev.REG_SRAM_TEST, tc.dev.BIST_BIAS_ERR, 0,
                                   comment=' // Return on BIST error')
                for group in range(tc.dev.P_NUMGROUPS):
                    apb.write_ctl(group, tc.dev.REG_SRAM_TEST, 0,
                                  comment=' // Reset BIST', force_write=True)
                apb.output('\n', embedded_code)

            apb.function_footer()

            if block_mode or not (embedded_code or compact_weights):
                hw_kern_offs, hw_kern_len, hw_kern_count, hw_kern_ochan = kernels.load(
                    embedded_code,
                    apb,
                    layers,
                    hw_operator,
                    hw_kernel,
                    hw_kernel_size,
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
                    verify_kernels,
                )
                hw_bias_offs, hw_bias_group, group_bias_max = kbias.load(
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
                    list(set().union(groups_used)),
                    flatten,
                )

            kern_offs = np.zeros((layers), dtype=np.int64)
            kern_len = np.zeros((layers), dtype=np.int64)
            kern_count = np.zeros((layers), dtype=np.int64)
            kern_ochan = np.zeros((layers), dtype=np.int64)
            bias_offs = [[None] * tc.dev.P_NUMGROUPS for _ in range(layers)]
            bias_group = [None] * layers
            for i, e in enumerate(kernel_ptrs):
                if i >= layers:
                    break
                if e is not None:
                    kern_offs[i] = hw_kern_offs[e]
                    kern_len[i] = hw_kern_len[e]
                    kern_count[i] = hw_kern_count[e]
                    kern_ochan[i] = hw_kern_ochan[e]
                else:
                    kernel_ptrs[i] = i
            for i, e in enumerate(bias_ptrs):
                if i >= layers:
                    break
                if e is not None:
                    bias_offs[i] = hw_bias_offs[e]
                    bias_group[i] = hw_bias_group[e]
                else:
                    bias_ptrs[i] = i

            if verbose:
                print('\nGlobal configuration:')
                print('---------------------')
                print(f'Used processors     = 0x{processors_used:016x}')
                print(f'Used quadrants      = {groups_used}')
                if start_layer > 0:
                    print(f'Starting layer      = {start_layer}')
                if any(s != i+1 and (s != -1 or i != final_layer)
                       for i, s in enumerate(next_sequence)):
                    print('Next layer sequence = [',
                          ', '.join(str(k) if k != -1 else 'stop' for k in next_sequence), ']',
                          sep='',)

                print('\nPer-quadrant configuration:')
                print('---------------------------')
                print(f'Used bias memory    = {group_bias_max}')

                print('\nPer-layer configuration:')
                print('------------------------')
                if repeat_layers > 1:
                    print(f'Layer repeat count  = {repeat_layers}')
                print(f'Quadrant map        = {group_map}')

                print('Input offset        = [',
                      ', '.join(f'0x{k:04x}' if k is not None
                                else 'N/A' for k in in_offset), ']', sep='',)
                print(f'Streaming           = {streaming}')
                print(f'Input channels      = {input_chan}')
                print(f'Input dimensions    = {input_dim}')
                print(f'Flatten             = {flatten}')
                if any(s > 0 for s in input_skip):
                    print(f'Input read gap      = {input_skip}')
                if any(s > 0 for s in input_channel_skip):
                    print(f'Input channel skip  = {input_channel_skip}')
                print(f'Input expansion     = {in_expand}')
                print(f'Expansion threshold = {in_expand_thresh}')

                print(f'Pooling             = {pool}')
                if any(h != 1 or w != 1 for h, w in pool_dilation):
                    print(f'Pooling dilation    = {pool_dilation}')
                print(f'Pooling stride      = {pool_stride}')
                print(f'Pooled dimensions   = {pooled_dim}')

                print('Processor map       = [',
                      ', '.join(f'0x{k:016x}' for k in processor_map), ']', sep='',)

                print('Element-wise op     = [',
                      ', '.join(op.string(k, elt=True) for k in eltwise), ']', sep='',)
                print(f'Operand expansion   = {operands}')

                print(f'Output channels     = {output_chan}')
                print(f'Output dimensions   = {output_dim}')
                if any(s > 0 for s in write_gap):
                    print(f'Output write gap    = {write_gap}')
                print(f'Output expansion    = {out_expand}')
                print(f'Expansion threshold = {out_expand_thresh}')
                print(f'Output shift        = {output_shift}')
                print('Output processors   = [',
                      ', '.join(f'0x{k:016x}' if k is not None
                                else 'N/A' for k in output_processor_map), ']', sep='',)
                print(f'Output data bits    = {output_width}')

                print(f'Quadrant with bias  = {bias_group}')
                print(f'Bias offset         = {bias_offs}')

                print('Output offset       = [',
                      ', '.join(f'0x{k:04x}' for k in out_offset), ']', sep='',)

                print('Operator            = [',
                      ', '.join(op.string(k) for k in operator), ']', sep='',)
                print('Activation          = [',
                      ', '.join(op.act_string(k) if k is not None
                                else 'no' for k in activation), ']', sep='',)
                print(f'Kernel offset       = {kern_offs.tolist()}')
                print(f'Kernel length       = {kern_len.tolist()}')
                print(f'Kernel count        = {kern_count.tolist()}')
                print(f'Kernel dimensions   = {kernel_size}')
                if any(h != 1 or w != 1 for h, w in dilation):
                    print(f'Dilation            = {dilation}')
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
                    hw_layer = r * (layers + sum_hw_layers) + ll + hw_add_layers[ll]

                    local_source = False
                    for group in groups_used:
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
                            output_processor_map[ll] & 2**tc.dev.P_NUMPRO - 1 << \
                            group*tc.dev.P_NUMPRO
                        if popcount(gmap) > 1:
                            p = ffs(gmap)
                            while p < fls(gmap):
                                gap = ffs(gmap & ~(2**(p+1) - 1)) - p - 1
                                gap_min, gap_max = min(gap, gap_min), max(gap, gap_max)
                                p += gap + 1
                            local_source = \
                                gap_min != gap_max or gap_max > 0 and hw_operator[ll] == op.NONE

                        # FIXME: Check that we don't overlap by-16 groups when in local_source mode
                        # FIXME: Non-uniform gaps are not supported

                    # For passthrough, determine time slot count (maximum across all used groups)
                    tscnt_max = 0
                    for group in groups_used:
                        if hw_operator[ll] == op.NONE:
                            if popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                        % 2**tc.dev.P_NUMPRO) != 0:
                                tscnt_max = max(
                                    tscnt_max,
                                    (popcount((processor_map[ll] >> group*tc.dev.P_NUMPRO)
                                              % 2**tc.dev.P_NUMPRO)
                                     * output_width[ll] // 8 - 1) // 4
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
                                              % 2**tc.dev.P_NUMPRO)
                                     * output_width[ll] + 7) // 8 - 1
                                )
                    timeslots[ll] = tscnt_max + 1
                    if flatten[ll]:
                        timeslots[ll] *= hw_pooled_dim[ll][0] * hw_pooled_dim[ll][1]

                    for gindex, group in enumerate(groups_used):
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            # Insert small single-quadrant passthrough layer
                            apb.output('  // Average pool accumulator reset layer and\n',
                                       embedded_code)
                        apb.output(f'  // Layer {r * layers + ll} quadrant {group}\n',
                                   embedded_code)

                        val = 0
                        if link_layer:
                            if ll != final_layer:
                                val = 1 << 7 | (ll + 1)
                            else:
                                val = 1 << 8  # Stop
                        else:
                            lt = next_sequence[ll]
                            if lt == -1:
                                if ll != layers - 1:  # Don't set stop bit unless required
                                    val = 1 << 8
                            elif lt != ll + 1:
                                val = 1 << 7 | lt + hw_add_layers[lt]
                            elif snoop_sequence[ll] is not None:
                                lt = snoop_sequence[ll]
                                assert lt >= 0
                                val = 1 << 7 | lt + hw_add_layers[lt]
                            if lt != -1 and gindex == 0:
                                if in_sequences[lt] is not None and ll in in_sequences[lt] \
                                   and operands[lt] == 1:
                                    ll_index = in_sequences[lt].index(ll)
                                    ll_offset = out_offset[ll] - ll_index * write_gap[ll] * 4
                                    offs = 0
                                    for e in in_sequences[lt][:ll_index]:
                                        offs += output_dim[e][0] * output_dim[e][1]
                                    if in_offset[lt] != ll_offset \
                                       and out_offset[ll] != offs * 4:
                                        wprint(f'{layer_pfx(ll)}The input offset '
                                               f'0x{in_offset[lt]:04x} of the next sequence '
                                               f'(layer {layer_str(lt)}) does not match the '
                                               'current layer\'s output offset '
                                               f'0x{out_offset[ll]:04x}, write gap '
                                               f'{write_gap[ll]}, input #{ll_index}.')
                                    pix = sum(output_dim[e][0] * output_dim[e][1]
                                              for e in in_sequences[lt])
                                    if (
                                        (input_chan[lt] != output_chan[ll] * len(in_sequences[lt])
                                         or input_dim[lt] != output_dim[ll])
                                        and (input_chan[lt] != output_chan[ll]
                                             or input_dim[lt][0] * input_dim[lt][1] != pix)
                                    ):
                                        wprint(f'{layer_pfx(ll)}The input dimensions of the next '
                                               f'sequence (layer {layer_str(lt)}, '
                                               f'{len(in_sequences[lt])} inputs, '
                                               f'{input_chan[lt]}x{input_dim_str[lt]}) do '
                                               "not match the current layer's output dimensions "
                                               f'({output_chan[ll]}x{output_dim_str[ll]}).')

                        if hasattr(tc.dev, 'LREG_NXTLYR'):
                            apb.write_lreg(group, hw_layer, tc.dev.LREG_NXTLYR, val,
                                           comment=' // Next Layer')

                        # Configure row count
                        if flatten[ll]:
                            in_row = pool[ll][0]
                            in_col = pool[ll][1]
                        else:
                            if hw_operator[ll] == op.CONVTRANSPOSE2D:
                                in_row = stride[ll][0] * hw_input_dim[ll][0]
                                in_col = stride[ll][1] * hw_input_dim[ll][1]
                            elif hw_operator[ll] == op.NONE and emulate_eltwise[ll]:
                                in_row = hw_input_dim[ll][0] * in_expand[ll]
                                in_col = hw_input_dim[ll][1]
                            else:
                                in_row = hw_input_dim[ll][0]
                                in_col = hw_input_dim[ll][1]
                        if hasattr(tc.dev, 'CNT_DIFF_OFFS'):
                            diff = (in_row - ((in_row - pool[ll][0] - pool_dilation[ll][0] + 1)
                                              // pool_stride[ll][0]) * pool_stride[ll][0])
                            val = in_row - diff  # Stop row, 0-based
                            assert val < 2**tc.dev.MAX_CNT_BITS

                            # Stop column
                            if hw_operator[ll] == op.CONV1D:
                                diff = 1
                            else:
                                diff = (in_col - ((in_col - pool[ll][1] - pool_dilation[ll][1] + 1)
                                                  // pool_stride[ll][1]) * pool_stride[ll][1])
                            # Bytes to next starting element
                            diff = (diff + (pool_stride[ll][0] - 1) * in_col) \
                                * (input_skip[ll] + 1) * operands[ll] * in_expand[ll]

                            val |= diff << tc.dev.CNT_DIFF_OFFS
                            if hw_padding[ll][0] > 0:
                                assert hw_padding[ll][0] - 1 < 2**2
                                val |= 1 << tc.dev.PAD_ENA_OFFS
                                val |= hw_padding[ll][0] - 1 << tc.dev.PAD_CNT_OFFS
                        else:
                            val = in_row - 1
                            assert hw_padding[ll][0] < 2**2
                            assert val + 2*hw_padding[ll][0] < 2**tc.dev.MAX_CNT_BITS
                            val |= hw_padding[ll][0] << tc.dev.PAD_CNT_OFFS
                            val += 2*hw_padding[ll][0]
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_RCNT, 1 << 16)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_RCNT, val,
                                       comment=' // Rows')

                        # Configure column count (evaluates to 0 for 1D convolutions)
                        if hasattr(tc.dev, 'CNT_DIFF_OFFS'):
                            # Calculate last pooling fetch before advancing to next row
                            diff = (in_col - ((in_col - pool[ll][1] - pool_dilation[ll][1] + 1)
                                              // pool_stride[ll][1]) * pool_stride[ll][1])
                            val = in_col - diff
                            assert val < 2**tc.dev.MAX_CNT_BITS
                            val |= diff << tc.dev.CNT_DIFF_OFFS
                            if hw_padding[ll][1] > 0:
                                assert hw_padding[ll][1] - 1 < 2**2
                                val |= 1 << tc.dev.PAD_ENA_OFFS
                                val |= hw_padding[ll][1] - 1 << tc.dev.PAD_CNT_OFFS
                        else:
                            val = in_col - 1
                            assert hw_padding[ll][1] < 2**2
                            assert val + 2 * hw_padding[ll][1] < 2**tc.dev.MAX_CNT_BITS
                            val |= hw_padding[ll][1] << tc.dev.PAD_CNT_OFFS
                            val += 2 * hw_padding[ll][1]
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_CCNT, 1 << 16)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_CCNT, val,
                                       comment=' // Columns')

                        # Configure pooling row count
                        val = (pool[ll][0] - 1) * pool_dilation[ll][0]
                        assert val < 2**4
                        if hasattr(tc.dev, 'CNT_INC_OFFS'):
                            val |= pool_dilation[ll][0] - 1 << tc.dev.CNT_INC_OFFS
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_PRCNT, val,
                                       comment=' // Pooling rows')

                        # Configure pooling column count
                        val = (pool[ll][1] - 1) * pool_dilation[ll][1]
                        assert val < 2**4
                        if hasattr(tc.dev, 'CNT_INC_OFFS'):
                            val |= pool_dilation[ll][1] - 1 << tc.dev.CNT_INC_OFFS
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_PCCNT, val,
                                       comment=' // Pooling columns')

                        # Configure pooling stride count
                        if hw_operator[ll] == op.CONVTRANSPOSE2D:
                            val = 0
                        elif pool_stride[ll][0] > 1:
                            val = pool_stride[ll][0]-1
                        else:
                            val = stride[ll][0]-1
                        assert val < 2**4
                        if hasattr(tc.dev, 'MP_STRIDE_OFFS'):  # Multipass stride
                            val |= pool_stride[ll][0] * operands[ll] * in_expand[ll] \
                                * (input_skip[ll] + 1) << tc.dev.MP_STRIDE_OFFS
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_STRIDE, 0x10)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_STRIDE, val,
                                       comment=' // Stride')

                        val = (out_offset[ll] - out_ignore[ll]) // 4
                        if not local_source:
                            # Configure SRAM write pointer -- write ptr is global
                            # (unless depth-wise w/o broadcast is used).
                            # Get offset to first available instance of the first used
                            # processor of the next layer.
                            if hw_operator[ll] != op.NONE \
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
                                if hw_operator[ll] != op.NONE:
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
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_WPTR_BASE, val)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_WPTR_BASE, val,
                                       comment=' // SRAM write ptr')

                        # Write Pointer Timeslot Offset Register
                        # Used for 1x1 convolution, and pooling without convolution
                        val = 0
                        if hw_operator[ll] in [op.CONV2D, op.LINEAR]:
                            if hw_kernel_size[ll] == [1, 1] and conv_groups[ll] == 1:
                                val = 1
                            elif conv_groups[ll] > 1 and not broadcast_mode[ll]:
                                val = tc.dev.INSTANCE_SIZE * 4
                        elif hw_operator[ll] == op.NONE:
                            if popcount(processor_map[ll]) > 4 \
                               or operands[ll] > 1 and in_expand[ll] > 1:
                                val = tc.dev.INSTANCE_SIZE * 4
                            else:
                                val = tc.dev.INSTANCE_SIZE
                        assert val < 2**tc.dev.MAX_PTR_BITS
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_WPTR_TOFFS, val)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_WPTR_TOFFS, val,
                                       comment=' // Write ptr time slot offs')

                        if hw_operator[ll] != op.NONE:
                            # [15:0] Write Pointer Mask Offset Register
                            val = 1 << tc.dev.WRITE_PTR_SHIFT
                            apb.write_lreg(group, hw_layer, tc.dev.LREG_WPTR_MOFFS, val,
                                           comment=' // Write ptr mask offs')

                        # [15:0] Write Pointer Multi-Pass Channel Offset Register
                        val = 0
                        if out_expand[ll] > 1:
                            val = (output_width[ll] // 8) * (write_gap[ll] + 1)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_WPTR_CHOFFS, val,
                                       comment=' // Write ptr multi-pass channel offs')

                        # Configure sram read ptr count -- read ptr is local
                        # Source address must match write pointer of previous layer (minus global
                        # offset)
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_RPTR_BASE,
                                           in_offset[ll] // 4)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_RPTR_BASE,
                                       in_offset[ll] // 4,
                                       comment=' // SRAM read ptr')

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
                           and hw_operator[ll] != op.NONE and group == groups_used[0] \
                           and conv_groups[ll] == 1:
                            # Set external source for other active processing groups (can be
                            # zero if no other groups are processing). Do not set the bit
                            # corresponding to this group (e.g., if group == 0, do not set bit 12)
                            sources = 0
                            for t in range(groups_used[0]+1, tc.dev.P_NUMGROUPS):
                                # See if any processors other than this one are operating
                                # and set the cnnsiena bit if true
                                if (processor_map[ll] >> (t * tc.dev.P_NUMPRO)) \
                                   % 2**tc.dev.P_NUMPRO:
                                    sources |= 1 << t

                            # Also set cnnsiena if we get the bias from that group
                            if bias_group[ll] is not None and bias_group[ll] != group:
                                sources |= 1 << bias_group[ll]
                            val |= sources << 12

                        if rd_ahead[ll] and hasattr(tc.dev, 'RD_AHEAD_OFFS'):
                            val |= 1 << tc.dev.RD_AHEAD_OFFS

                        if hasattr(tc.dev, 'CPRIME_MAX_OFFS') and hw_operator[ll] != op.NONE:
                            val |= hw_kernel_size[ll][0] - 1 << tc.dev.RPRIME_MAX_OFFS
                            val |= hw_kernel_size[ll][1] - 1 << tc.dev.CPRIME_MAX_OFFS

                        if rd_ahead[ll] and hasattr(tc.dev, 'SHIFT_CNT_OFFS'):
                            val |= ((in_expand[ll] - 1) // 4 if tcalc[ll] else in_expand[ll] - 1) \
                                << tc.dev.SHIFT_CNT_OFFS

                        if bypass[ll]:
                            val |= 1 << 30

                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_LCTL, 0x920)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_LCTL, val,
                                       comment=' // Layer control')

                        flatten_prod[ll] = 0
                        if flatten[ll]:
                            # Store all bits, top programmed in post processing register
                            flatten_prod[ll] = \
                                in_expand[ll] * hw_pooled_dim[ll][0] * hw_pooled_dim[ll][1] - 1
                            in_exp = flatten_prod[ll] & 0x0f  # Lower 4 bits only
                        elif hw_operator[ll] == op.NONE and emulate_eltwise[ll]:
                            in_exp = 0
                        else:
                            in_exp = in_expand[ll] - 1

                        if in_exp >= 2**4:
                            eprint(f'{layer_pfx(ll)}Input expansion of {in_exp+1} exceeds device '
                                   f'limit of {2**4}.')

                        quant = abs(quantization[ll]) if not bypass[ll] else 8
                        val = (fls(output_processor_map[ll])
                               - (ffs(output_processor_map[ll]) & ~(tc.dev.P_SHARED-1))) \
                            * quant << tc.dev.XPCH_MAX_OFFS | in_exp
                        if hw_operator[ll] != op.NONE:
                            wptr_skip = out_expand[ll] * (write_gap[ll] + 1) - 1
                        else:
                            wptr_skip = write_gap[ll]
                        assert wptr_skip < 2**tc.dev.MAX_WPTRINC_BITS
                        val |= wptr_skip << 4

                        apb.write_lreg(group, hw_layer, tc.dev.LREG_LCTL2, val,
                                       comment=' // Layer control 2')

                        # Configure mask start and end addresses
                        # Every mask memory starts from the same offset for all processors
                        oned_sad = 0
                        if hw_operator[ll] != op.NONE:
                            # FIXME: bypass corner cases
                            kc = kern_count[ll] if not bypass[ll] \
                                else output_chan[ll] // conv_groups[ll]
                            kl = (kc - 1) * quant

                            if ll == start_layer and calcx4[ll]:
                                # FIXME: Handle fast_fifo_quad and calcx4
                                if calcx4[ll]:
                                    kl += quant
                                kl = (kl + 3) // 4
                                if calcx4[ll]:
                                    kl -= quant
                            koffs, oned_sad = divmod(9 * kern_offs[ll],
                                                     hw_kernel_size[ll][0] * hw_kernel_size[ll][1])
                            if calcx4[ll]:
                                koffs = kernels.calcx4_index(koffs)
                            koffs *= 8
                        else:
                            kl = koffs = 0

                        if hasattr(tc.dev, 'LREG_MCNT1'):
                            if hw_operator[ll] != op.NONE:
                                assert koffs < 2**19
                                assert kl + koffs < 2**19
                                apb.write_lreg(group, hw_layer, tc.dev.LREG_MCNT1,
                                               kl + koffs,
                                               comment=' // Mask count')
                                apb.write_lreg(group, hw_layer, tc.dev.LREG_MCNT2, koffs,
                                               comment=' // Mask offset')
                            else:
                                val = (out_expand[ll] - 1) * 8
                                assert val < 2**19
                                apb.write_lreg(group, hw_layer, tc.dev.LREG_MCNT2, val,
                                               comment=' // Mask offset')
                        else:
                            if hw_operator[ll] != op.NONE:
                                assert koffs < 2**16
                                assert kl + koffs < 2**16
                                # kern_offs is always bytes
                                val = \
                                    koffs << tc.dev.MCNT_SAD_OFFS | kl + \
                                    koffs << tc.dev.MCNT_MAX_OFFS
                            elif emulate_eltwise[ll]:
                                val = 0
                            else:
                                val = (out_expand[ll] - 1) * 8
                                assert val < 2**16
                            apb.write_lreg(group, hw_layer, tc.dev.LREG_MCNT, val,
                                           comment=' // Mask offset and count')

                        if hasattr(tc.dev, 'LREG_OCHAN'):
                            if bypass[ll]:
                                val = output_chan[ll] - 1
                            elif hw_operator[ll] != op.NONE and conv_groups[ll] == 1:
                                val = kern_ochan[ll] - 1
                                if calcx4[ll]:
                                    val //= 4
                            elif conv_groups[ll] > 1:
                                val = (tscnt_max + 1) * in_expand[ll] - 1
                            else:
                                val = tscnt_max
                            apb.write_lreg(group, hw_layer, tc.dev.LREG_OCHAN, val,
                                           comment=' // Output channel count')

                        val = tscnt_max
                        assert 0 <= val < 2**4
                        if hw_operator[ll] == op.CONV1D:
                            val |= hw_kernel_size[ll][0] << 8 | 1 << 12
                            assert hw_kernel_size[ll][0] < 2**4
                        elif (hw_operator[ll] in [op.CONV2D, op.LINEAR]
                              and hw_kernel_size[ll] == [1, 1]
                              or hw_operator[ll] == op.NONE and operands[ll] == 1):
                            val |= 1 << 8
                        if operands[ll] > 1:
                            val |= \
                                1 << 13 | op.eltwise_fn(eltwise[ll]) << 14 | operands[ll] - 1 << 18
                            if (pool[ll][0] > 1 or pool[ll][1] > 1) \
                               and pool_first[ll]:
                                val |= 1 << 16
                            if hw_operator[ll] != op.NONE:  # CONV2D, LINEAR, CONVTRANSPOSE2D
                                val |= 1 << 17
                        assert 0 <= oned_sad < 2**4
                        val |= oned_sad << 4

                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_ONED, 0x100)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_ONED, val,
                                       comment=' // 1D')

                        # Configure tram pointer max
                        if hw_operator[ll] == op.CONV1D or \
                           hw_operator[ll] in [op.CONV2D, op.LINEAR] \
                           and hw_kernel_size[ll] == [1, 1] \
                           and (ll == 0 or not streaming[ll]):
                            if flatten_prod[ll] >= 2**4:
                                assert flatten_prod[ll] < 2**16
                                val = flatten_prod[ll] << 16 | (2 * flatten_prod[ll] + 1)
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
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_TPTR, val,
                                       comment=' // TRAM ptr max')

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
                        assert hw_operator[ll] != op.NONE or output_shift[ll] == 0
                        if output_shift[ll] < 0:
                            val |= (-output_shift[ll] | 2**4) << 13
                        else:
                            val |= output_shift[ll] << 13

                        # [24] ts_ena
                        # [25] onexone_ena

                        if conv_groups[ll] == 1 and group == bias_group[ll]:
                            # Enable bias only for one group
                            offs = bias_offs[ll][group]
                            if calcx4[ll]:
                                offs //= 4
                            assert offs < 2**12
                            val |= 1 << 12 | offs
                        elif bias_offs[ll][group] is not None and (
                            conv_groups[ll] > 1 or fast_fifo_quad and ll == 0
                        ):
                            # Enable bias for all groups
                            offs = bias_offs[ll][group]
                            if broadcast_mode[ll]:
                                offs //= 4
                            assert offs < 2**12
                            val |= 1 << 12 | offs

                        if not tc.dev.SUPPORT_MULTIPASS_ELTWISE_CONV_BIAS \
                           and group == bias_group[ll] \
                           and operator[ll] != op.NONE \
                           and operands[ll] > 1:
                            eprint(f'{layer_pfx(ll)}On this device, multi-pass element-wise '
                                   'operations must be in a separate layer from convolutions with '
                                   'bias.')

                        if hw_operator[ll] == op.NONE:
                            if operands[ll] == 1:
                                val |= 3 << 24
                            else:
                                val |= 1 << 24

                        if activation[ll] == op.ACT_ABS:
                            val |= 1 << 26

                        if flatten_prod[ll] >= 2**4:
                            hw_flatten[ll] = True
                            val |= 1 << 27 | (flatten_prod[ll] >> 4) << 18  # flatten_ena, xpmp_cnt

                        if hw_operator[ll] == op.CONVTRANSPOSE2D:
                            val |= 1 << 28

                        if conv_groups[ll] > 1:
                            val |= 1 << 30 | 1 << 24  # depthwise_ena, ts_ena

                        if calcx4[ll]:
                            val |= 1 << 29

                            if not tc.dev.SUPPORT_MULTIPASS_X4_PARTIALQUAD \
                               and out_expand[ll] > 1 and tc.dev.MAX_PROC != 64:
                                eprint(f'{layer_pfx(ll)}This device does not support `calcx4` '
                                       'with multi-pass when writing to fewer than 4 quadrants.',
                                       error=not state.ignore_hw_limits)

                        if tcalc[ll]:
                            val |= 1 << 31
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_POST, 0x03000000)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_POST, val,
                                       comment=' // Post processing register')

                        # Configure mask and processor enables
                        # Enable at most 16 processors and masks
                        val = (processor_map[ll] >> group*tc.dev.P_NUMPRO) % 2**tc.dev.P_NUMPRO
                        if hw_operator[ll] != op.NONE and not bypass[ll]:
                            val = val << 16 | val  # Mask enables
                        if avgpool_reset_layer[ll] and group == groups_used[0]:
                            apb.write_lreg(group, hw_layer - 1, tc.dev.LREG_ENA, 1)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_ENA, val,
                                       comment=' // Mask and processor enables')

                        delta1 = delta2 = stream_start = invol = 0
                        last_layer = False
                        if not tc.dev.REQUIRE_NEW_STREAMING:
                            if ll == start_layer and fifo:
                                # Start: 1
                                if override_start is not None:
                                    stream_start = override_start
                                elif streaming[ll]:
                                    stream_start = (pool[ll][0] - 1) * hw_input_dim[ll][1] \
                                        + pool[ll][1]
                                else:
                                    stream_start = hw_input_dim[ll][0] * hw_input_dim[ll][1]
                                    if big_data[ll]:
                                        stream_start = (stream_start + 3) // 4
                                if override_start is None:
                                    stream_start *= pool[ll][0]

                                if streaming[ll]:
                                    # Delta 1: This layer's pooling stride
                                    if override_delta1 is not None:
                                        delta1 = override_delta1
                                    else:
                                        delta1 = (pool_stride[ll][1] - 1) * operands[ll]
                                    if override_delta2 is not None:
                                        delta2 = override_delta2
                                    else:
                                        delta2 = (pool[ll][0] - 1) * hw_input_dim[ll][1] \
                                            * operands[ll]

                            elif ll > start_layer and streaming[ll]:
                                # Start: Prior layer's padded pooled row width * prior layer's
                                # kernel height + prior layer's kernel width + prior layer's pad
                                stream_start = (hw_pooled_dim[prev_sequence[ll]][1]
                                                + 2 * hw_padding[prev_sequence[ll]][1]) \
                                    * (hw_kernel_size[prev_sequence[ll]][0] - 1
                                       + pool[ll][0] - 1) \
                                    + hw_kernel_size[prev_sequence[ll]][1] - 1 + pool[ll][1] \
                                    + increase_start

                                # Delta 1: This layer's pooling stride
                                delta1 = pool_stride[ll][1] * operands[ll] + increase_delta1

                                # Delta 2: (This layer's pooling - 1) * full prior layer's padded
                                # rows + prior layer's pad
                                delta2 = (pool_stride[ll][0] - 1) \
                                    * (hw_pooled_dim[prev_sequence[ll]][1]
                                        + 2 * hw_padding[prev_sequence[ll]][1]) \
                                    + pool[ll][1] * operands[ll] + increase_delta2
                        else:
                            # MAX78002
                            # =IF(Pad=0,Stride,Pad)
                            row_prim = hw_padding[ll][1] or pool_stride[ll][0]
                            # =IF((Row_Pool*Row_Dilation_Stride)>Stride,
                            #     Row_Pool*Row_Dilation_Stride,Stride)
                            row_inc = max(pool[ll][0] * pool_dilation[ll][0], pool_stride[ll][0])
                            # =IF((Col_Pool*Col_Dilation_Stride)>Stride,
                            #     (Col_Pool*Col_Dilation_Stride),Stride)
                            col_inc = max(pool[ll][1] * pool_dilation[ll][1], pool_stride[ll][1])

                            if streaming[ll]:
                                if ll == final_layer or not streaming[next_sequence[ll]]:
                                    last_layer = True
                                if debug_new_streaming and ll > 0:
                                    # Cols_Stride=ROUNDDOWN(Cols/Col_Inc,0)
                                    # =IF(Cols_Stride*Stride+Pool>Cols,Cols_Stride-1,Cols_Stride)
                                    effective_cols = hw_input_dim[ll][1] // col_inc
                                    if effective_cols * pool_stride[ll][1] + pool[ll][1] > \
                                       hw_input_dim[ll][1]:
                                        effective_cols -= 1
                                    # =IF(Col_Inc=1,0,Cols-(Effective_Cols*Col_Inc))
                                    col_adjust = 0 if col_inc == 1 \
                                        else hw_input_dim[ll][1] - effective_cols * col_inc
                                else:
                                    # =IF(((ROUNDDOWN(Cols/Stride,0)*Stride)+(Pool-1))>Cols,
                                    #     (ROUNDDOWN((Cols/Stride),0))-1,
                                    #     (ROUNDDOWN((Cols/Stride),0)))
                                    effective_cols = hw_input_dim[ll][1] // pool_stride[ll][1]
                                    if effective_cols * pool_stride[ll][1] + pool[ll][1] - 1 > \
                                       hw_input_dim[ll][1]:
                                        effective_cols -= 1
                                    # =(Cols-(Effective_Cols*Col_Inc))
                                    col_adjust = hw_input_dim[ll][1] - effective_cols * col_inc

                            # Prefill
                            if ll == start_layer and override_start is not None:
                                stream_start_hwc = stream_start = override_start
                            elif ll == start_layer and fifo:
                                if streaming[ll]:
                                    # =IF(AND(Stride=1,Row_Pool=1),Col_Inc,
                                    #     ((((Row_Inc)*(Cols+(Pad*2)))
                                    #                       +(Pad+(2*Col_Inc)))*Elementwise) +
                                    #      (Read_Ahead*Stride))
                                    if pool_stride[ll][0] == 1 and pool[ll][0] == 1:
                                        stream_start = col_inc
                                    else:
                                        stream_start = (
                                            row_inc * (hw_input_dim[ll][1] + hw_padding[ll][1] * 2)
                                            + (hw_padding[ll][1] + 2 * col_inc)
                                        ) * operands[ll]
                                        if rd_ahead[ll]:
                                            stream_start += pool_stride[ll][1]
                                else:  # fifo only
                                    stream_start = hw_input_dim[ll][0] * hw_input_dim[ll][1]
                                stream_start_hwc = stream_start
                                if big_data[ll]:
                                    stream_start = (stream_start + 3) // 4
                            elif ll > start_layer and streaming[ll]:
                                if debug_new_streaming and ll > 0:
                                    # =(Row_Prim*(Cols+(Pad*2)))
                                    #   +(((Row_Inc*(Cols+(Pad*2)))
                                    #   +(Pad+(2*Col_Inc)))*Elementwise)
                                    #   +(Read_Ahead*Stride)
                                    # if last_layer:
                                    #   += (Col_Adj*Stride)
                                    stream_start = \
                                        row_prim * \
                                        (hw_input_dim[ll][1] + 2 * hw_padding[ll][1]) \
                                        + (row_inc * (hw_input_dim[ll][1]
                                                      + 2 * hw_padding[ll][1])
                                           + hw_padding[ll][1] + 2 * col_inc) * operands[ll]
                                else:
                                    # =(Pad*(Cols+(Pad*2)))
                                    #   +(((Row_Inc*(Cols+(Pad*2)))
                                    #   +(Pad+(2*Col_Inc)))*Elementwise)
                                    #   +(Read_Ahead*Stride)
                                    # if last_layer:
                                    #   += (Col_Adj*Stride)
                                    stream_start = \
                                        hw_padding[ll][1] * \
                                        (hw_input_dim[ll][1] + 2 * hw_padding[ll][1]) \
                                        + (row_inc * (hw_input_dim[ll][1]
                                                      + 2 * hw_padding[ll][1])
                                           + hw_padding[ll][1] + 2 * col_inc) * operands[ll]
                                if rd_ahead[ll]:
                                    stream_start += pool_stride[ll][1]
                                if last_layer and debug_new_streaming:
                                    stream_start += col_adjust * pool_stride[ll][1]
                                stream_start_hwc = stream_start
                                if big_data[ll]:
                                    # =(ROUNDUP(Prefill/4,0))
                                    stream_start = (stream_start + 3) // 4

                            # Delta 1 Count, Delta 2 Count
                            if streaming[ll]:
                                # =IF((Cols-(Effective_Cols*Stride))<0,0,
                                #      (Cols-(Effective_Cols*Stride)))
                                skipped_cols = max(
                                    0,
                                    (hw_input_dim[ll][1] - (effective_cols * pool_stride[ll][1]))
                                )

                                if ll == start_layer:
                                    if override_delta1 is not None:
                                        delta1 = override_delta1
                                    else:
                                        # =IF(AND(Stride=1,Row_Pool=1),(Stride*Elementwise)-1,
                                        #     (Stride*Elementwise))
                                        delta1 = pool_stride[ll][1] * operands[ll]
                                        if pool_stride[ll][0] == 1 and pool[ll][0] == 1:
                                            delta1 -= 1
                                        if big_data[ll]:
                                            # =(ROUNDUP(Delta1_0/4,0))
                                            delta1 = (delta1 + 3) // 4

                                    if override_delta2 is not None:
                                        delta2 = override_delta2
                                    else:
                                        if debug_new_streaming and ll > 0:
                                            # =IF(Stride=1,Delta1_0+Col_Adj,
                                            #     (((Stride-1)*Cols))+Col_Adj)
                                            if pool_stride[ll][0] == 1:
                                                delta2 = delta1 + col_adjust
                                            else:
                                                delta2 = (pool_stride[ll][0] - 1) \
                                                    * hw_input_dim[ll][1] + col_adjust
                                        else:
                                            # =IF(Stride=1,Delta1_0+Skipped_Cols,
                                            #            (((Stride-1)*Cols))+Skipped_Cols
                                            #              +(Col_Pool-1))
                                            if pool_stride[ll][0] == 1:
                                                delta2 = delta1 + skipped_cols
                                            else:
                                                delta2 = (pool_stride[ll][0] - 1) \
                                                    * hw_input_dim[ll][1] \
                                                    + skipped_cols + pool[ll][1] - 1
                                        if big_data[ll]:
                                            # =(ROUNDUP(Delta2_0/4,0))
                                            delta2 = (delta2 + 3) // 4
                                        if pipeline and delta2 > 0:
                                            delta2 += 1
                                else:  # != start_layer
                                    # =Stride*Elementwise
                                    delta1 = pool_stride[ll][1] * operands[ll]
                                    if big_data[ll]:
                                        # =(ROUNDUP(Delta1/4,0))
                                        delta1 = (delta1 + 3) // 4
                                    delta1 += increase_delta1

                                    if debug_new_streaming and ll > 0:
                                        # =IF(Stride=1,Delta1+Col_Adj,((Stride-1)*Cols)+Col_Adj)
                                        if pool_stride[ll][0] == 1:
                                            delta2 = delta1 + col_adjust
                                        else:
                                            delta2 = (pool_stride[ll][0] - 1) \
                                                * hw_input_dim[ll][1] \
                                                + col_adjust
                                    else:
                                        # =IF(Stride=1,Delta1+Skipped_Cols,
                                        #     ((Stride-1)*Cols)+Skipped_Cols)
                                        if pool_stride[ll][0] == 1:
                                            delta2 = delta1 + skipped_cols
                                        else:
                                            delta2 = (pool_stride[ll][0] - 1) \
                                                * hw_input_dim[ll][1] \
                                                + skipped_cols
                                    if big_data[ll]:
                                        # =(ROUNDUP(Delta2/4,0))
                                        delta2 = (delta2 + 3) // 4
                                    delta2 += increase_delta2

                        # strm_invol[3:0]: Per stream invol offset - based on stream count
                        if ll > start_layer and streaming[ll]:
                            invol = sum(in_expand_invol[:ll])

                        assert stream_start < 2**tc.dev.MAX_ISVAL_BITS
                        val = stream_start
                        if streaming[ll] and group == groups_used[0]:
                            state.stream_start.append(stream_start + 1)
                        if state.fifo_go and ll == start_layer:
                            val |= 1 << 25
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_STREAM1, val,
                                       comment=' // Stream processing start')

                        assert invol < 2**4, \
                            f'{layer_pfx(ll)}invol ({invol:04x}) exceeds supported range.'
                        assert delta1 < 2**5
                        if delta2 >= 2**tc.dev.MAX_DSVAL2_BITS:
                            eprint(f'Layer {ll}: delta2 ({delta2}) exceeds device maximum '
                                   f'({2**tc.dev.MAX_DSVAL2_BITS}). Reduce pooling.')
                        val = delta2 << 16 | delta1 << 4 | invol
                        if streaming[ll] and group == groups_used[0]:
                            state.delta1.append(delta1)
                            state.delta2.append(delta2)
                        apb.write_lreg(group, hw_layer, tc.dev.LREG_STREAM2, val,
                                       comment=' // Stream processing delta')

                        if fifo and streaming[ll]:
                            if ll == start_layer and override_rollover is not None:
                                val = override_rollover
                            elif not tc.dev.REQUIRE_NEW_STREAMING:
                                if big_data[ll]:
                                    # FIXME: stream_start + max(stride[ll][1], pool_stride[ll][1])
                                    val = 12
                                else:
                                    val = stream_start + (pool[ll][0] - 1) * hw_input_dim[ll][1] \
                                        + max(stride[ll][1], pool_stride[ll][1], pool[ll][1])
                                # Rollover must be multiple of multi-pass:
                                rem = val % in_expand[ll]
                                if rem > 0:
                                    val = val + in_expand[ll] - rem
                            else:
                                # MAX78002 - Buffer
                                if ll == start_layer:
                                    # =Prefill0 + ((Stride - 1) * (Cols + (Pad * 2))) + Col_Inc
                                    val = stream_start_hwc + col_inc \
                                        + (pool_stride[ll][1] - 1) * (hw_input_dim[ll][1]
                                                                      + hw_padding[ll][1] * 2)
                                    if big_data[ll]:
                                        # =Buffer0*4*Stride
                                        val *= pool_stride[ll][1] * 4
                                else:
                                    # =(MROUND(Prefill+((Passes-1)*((Row_Inc*Cols)+Pad+Col_Inc))
                                    #   +Col_Inc,Passes))
                                    val = stream_start_hwc \
                                        + (in_expand[ll] - 1) * (row_inc * hw_input_dim[ll][1]
                                                                 + hw_padding[ll][1]
                                                                 + col_inc) \
                                        + col_inc
                                    val += (in_expand[ll] - val % in_expand[ll]) % in_expand[ll]
                                    if big_data[ll]:
                                        # =(ROUNDUP(Buffer/4,0))
                                        val = (val + 3) // 4

                            assert val < 2**tc.dev.MAX_FBUF_BITS

                            if group == groups_used[0]:  # Run checks just once
                                # Check rollover vs available data memory
                                if output_processor_map[ll] & processor_map[ll] != 0:  # Overlap?
                                    if in_offset[ll] < out_offset[ll] - out_ignore[ll]:
                                        if in_offset[ll] + val * 4 \
                                           >= out_offset[ll] - out_ignore[ll]:
                                            eprint(
                                                f'{layer_pfx(ll)}Overlapping input and output: '
                                                f'in_offset 0x{in_offset[ll]:08x} + '
                                                f'rollover 0x{val:08x} * 4 >= '
                                                f'out_offset 0x{out_offset[ll]:08x} - '
                                                f'out_ignore 0x{out_ignore[ll]:08x}.',
                                                error=not no_error_stop,
                                            )
                                    else:
                                        if out_offset[ll] + val * 4 >= in_offset[ll]:
                                            eprint(
                                                f'{layer_pfx(ll)}Overlapping input and output: '
                                                f'out_offset 0x{out_offset[ll]:08x} + '
                                                f'rollover 0x{val:08x} * 4 >= '
                                                f'in_offset 0x{in_offset[ll]:08x}.',
                                                error=not no_error_stop,
                                            )
                                        if ll == terminating_layer:
                                            osize = \
                                                output_dim[ll][0] * output_dim[ll][1] + out_pad[ll]
                                            if out_offset[ll] + osize * out_expand[ll] * 4 >= \
                                               in_offset[ll]:
                                                eprint(
                                                    f'{layer_pfx(ll)}Overlapping input and '
                                                    f'output: out_offset 0x{out_offset[ll]:08x} + '
                                                    f'output of size {osize} '
                                                    f'({output_dim_str[ll]}) '
                                                    f'* {out_expand[ll]} * 4 >= '
                                                    f'in_offset 0x{in_offset[ll]:08x}.',
                                                    error=not no_error_stop,
                                                )
                                if in_offset[ll] + val * 4 >= tc.dev.INSTANCE_WIDTH \
                                   * tc.dev.P_SHARED * 4:
                                    eprint(
                                        'Input plus rollover exceeds instance size: '
                                        f'in_offset 0x{in_offset[ll]:08x}, '
                                        f'out_offset 0x{out_offset[ll]:08x}, '
                                        f'rollover 0x{val:08x}, '
                                        f'instance size 0x{tc.dev.INSTANCE_WIDTH*4:08x}.',
                                        error=not no_error_stop,
                                    )

                                # Check streaming buffers for overlap across all streaming layers
                                # and the data memories used by the processors in the streaming
                                # layers, as well as the output of the last streaming layer.
                                dmap = tc.dev.datamem_map(processor_map[ll],
                                                          fast_fifo_quad and ll == 0)
                                stream_buf[ll] = (in_offset[ll], in_offset[ll] + val * 4, dmap)
                                for pl in range(ll):
                                    if stream_buf[pl] is None:
                                        continue
                                    if stream_buf[pl][2] & dmap != 0 \
                                       and overlap(stream_buf[ll], stream_buf[pl]):
                                        eprint(
                                            f'{layer_pfx(ll)}Streaming buffer '
                                            f'({stream_buf[ll][0]:04x}-{stream_buf[ll][1]:04x}, '
                                            f'processors {processor_map[ll]:016x}) '
                                            f'overlaps layer {layer_str(pl)} '
                                            f'({stream_buf[pl][0]:04x}-{stream_buf[pl][1]:04x}, '
                                            f'processors {processor_map[pl]:016x}).',
                                            error=not overwrite_ok,
                                        )
                                    if rd_ahead[ll] \
                                       and tc.dev.datainstance_from_offs(stream_buf[ll][0]) \
                                       == tc.dev.datainstance_from_offs(stream_buf[pl][0]):
                                        eprint(
                                            f'{layer_pfx(ll)}In streaming mode with read-ahead, '
                                            'all streaming read-ahead layers must use separate '
                                            'memory instances. The layer conflicts with layer '
                                            f'{layer_str(pl)}; both use instance '
                                            f'{tc.dev.datainstance_from_offs(stream_buf[pl][0])}.',
                                        )

                                if ll == final_layer or not streaming[next_sequence[ll]]:
                                    dmap = tc.dev.datamem_map(output_processor_map[ll])
                                    for pl in range(ll + 1):
                                        if stream_buf[pl] is None:
                                            continue
                                        if stream_buf[pl][2] & dmap != 0 \
                                           and overlap((out_offset[ll], out_offset[ll]
                                                       + (output_dim[ll][0] * output_dim[ll][1]
                                                          + out_pad[ll]) * 4
                                                       * output_width[ll] // 8), stream_buf[pl]):
                                            eprint(
                                                f'{layer_pfx(ll)}The output '
                                                f'({out_offset[ll]:04x}-{stream_buf[ll][1]:04x}, '
                                                'output processors '
                                                f'{output_processor_map[ll]:016x}) overlaps '
                                                f'streaming buffer for layer {layer_str(pl)} '
                                                f'({stream_buf[pl][0]:04x}-{stream_buf[pl][1]:04x}'
                                                f', processors {processor_map[pl]:016x}).',
                                                error=not overwrite_ok,
                                            )

                            apb.write_lreg(group, hw_layer, tc.dev.LREG_FMAX, val,
                                           no_verify=not tc.dev.SUPPORT_ROLLOVER_READ,
                                           comment=' // Rollover')
                            rollover[ll] = val

                        # In read-ahead mode, ensure that input and output use separate
                        # instances. First, check the start addresses, then the end addresses.
                        if rd_ahead[ll]:
                            in_instance = (
                                tc.dev.datainstance_from_offs(in_offset[ll]),
                                tc.dev.datainstance_from_offs(in_offset[ll] + 4 * operands[ll]
                                                              * in_expand[ll] * hw_input_dim[ll][0]
                                                              * hw_input_dim[ll][1] - 1)
                            )
                            out_instance = (
                                tc.dev.datainstance_from_offs(out_offset[ll]),
                                tc.dev.datainstance_from_offs(out_offset[ll] + 4 * out_expand[ll]
                                                              * (output_dim[ll][0]
                                                                 * output_dim[ll][1]
                                                                 + out_pad[ll]) - 1)
                            )
                            if in_instance[0] == out_instance[0] \
                               or in_instance[1] == out_instance[1]:
                                eprint(f'{layer_pfx(ll)}Input and output cannot use the same data '
                                       'memory instances in read-ahead mode. '
                                       f'in_offset: {in_offset[ll]:04x}/instance(s) '
                                       f'{in_instance}, out_offset: {out_offset[ll]:04x}/'
                                       f'instance(s) {out_instance}.')

                        if ll == start_layer and fifo:
                            val = hw_input_dim[ll][0] * hw_input_dim[ll][1]
                            if big_data[ll]:
                                val = (val + 3) // 4
                            assert val < 2**tc.dev.MAX_IFRM_BITS
                            apb.write_ctl(group, tc.dev.REG_IFRM, val,
                                          comment=' // Input frame size')

                        apb.output('\n', embedded_code)  # End of group

            if zero_unused:
                for r in range(repeat_layers):
                    for ll in range(first_layer_used, layers, tc.dev.MAX_LAYERS):
                        for group in groups_used:
                            for reg in range(tc.dev.MAX_LREG+1):
                                if reg == tc.dev.LREG_RFU:  # Register 2 not implemented
                                    continue
                                apb.write_lreg(group, hw_layer, reg, 0,
                                               force_write=True,
                                               comment=f' // Zero unused layer {ll} registers')
                    if hasattr(tc.dev, 'MIN_STREAM_LREG'):
                        for ll in range(first_layer_used, layers, tc.dev.MAX_STREAM_LAYERS):
                            for group in groups_used:
                                for reg in range(tc.dev.MIN_STREAM_LREG, tc.dev.MAX_STREAM_LREG+1,
                                                 tc.dev.MAX_STREAM_LAYERS):
                                    apb.write_lreg(group, hw_layer, reg, 0,
                                                   force_write=True,
                                                   comment=f' // Zero unused layer {ll} registers')

            if snoop is not None:
                apb.output('  // Configure conditional execution\n', embedded_code)
                for group in groups_used:
                    assert len(snoop) == 32
                    apb.write_ctl(group, tc.dev.REG_SNP1_A1, snoop[0],
                                  comment=' // Address snoop 1 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_A2, snoop[1],
                                  comment=' // Address snoop 1 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP1_D1, snoop[2],
                                  comment=' // Data snoop 1 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_D2, snoop[3],
                                  comment=' // Data snoop 1 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP1_X1, snoop[4],
                                  comment=' // Count snoop 1 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_X2, snoop[5],
                                  comment=' // Count snoop 1 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP1_C1, snoop[6],
                                  comment=' // Snoop 1 control register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_C2, snoop[7],
                                  comment=' // Snoop 1 control register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP1_ACC, snoop[8],
                                  comment=' // Snoop 1 data accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP1_HIT, snoop[9],
                                  comment=' // Snoop 1 match hit accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP1_MAX, snoop[10],
                                  comment=' // Snoop 1 match max accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP1_AM, snoop[11],
                                  comment=' // Snoop 1 match address register')
                    apb.write_ctl(group, tc.dev.REG_SNP2_A1, snoop[12],
                                  comment=' // Address snoop 2 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP2_A2, snoop[13],
                                  comment=' // Address snoop 2 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP2_D1, snoop[14],
                                  comment=' // Data snoop 2 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP2_D2, snoop[15],
                                  comment=' // Data snoop 2 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP2_X1, snoop[16],
                                  comment=' // Count snoop 2 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP2_X2, snoop[17],
                                  comment=' // Count snoop 2 register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP2_C1, snoop[18],
                                  comment=' // Snoop 2 control register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP2_C2, snoop[19],
                                  comment=' // Snoop 2 control register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP2_ACC, snoop[20],
                                  comment=' // Snoop 2 data accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP2_HIT, snoop[21],
                                  comment=' // Snoop 2 match hit accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP2_MAX, snoop[22],
                                  comment=' // Snoop 2 match max accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP2_AM, snoop[23],
                                  comment=' // Snoop 2 match address register')

                    apb.output('\n', embedded_code)

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
                        hw_padding[start_layer],
                        csv_file=csv,
                    )

            if verbose:
                print('\nGlobal registers:')
                print('-----------------')

            # Configure the FIFOs when we're using them
            if fifo:
                apb.output('\n', embedded_code)

                # FIFO control
                if not fast_fifo:
                    val = 0x02 << 2 | 0x02 << 7 | tc.dev.FIFO_READY_SEL
                    if tc.dev.REQUIRE_FIFO_CPL:
                        val |= 1 << 11
                    for i in range(input_chan[start_layer]):
                        if processor_map_0 & 1 << (i % tc.dev.P_NUMGROUPS) * tc.dev.P_NUMPRO != 0:
                            val |= 1 << i % tc.dev.P_NUMGROUPS + 12
                    apb.write_fifo_ctl(tc.dev.FIFO_CTL, val,
                                       comment=' // FIFO control')
                else:
                    apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_IE, 0,
                                            comment=' // Fast FIFO interrupt enable')
                    val = 10 << 4  # Async, threshold 10
                    apb.write_fast_fifo_ctl(tc.dev.FAST_FIFO_CR, val,
                                            comment=' // Fast FIFO control')

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
            if snoop is not None:
                val |= 1 << 7
            if tc.dev.SUPPORT_PIPELINE and not pipeline:
                val |= 1 << 5

            if embedded_code:
                apb.function_footer()
                apb.function_header(function='start')

            if embedded_code or tc.dev.MODERN_SIM:
                apb.output('  cnn_time = 0;\n\n', embedded_code)

            # Enable all needed groups except the first one
            rdy_sel = tc.dev.READY_SEL if not pipeline else tc.dev.PIPELINE_READY_SEL
            for group in groups_used:
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
                if state.snoop_loop:
                    fval |= 1 << 7  # apbclkena
                else:
                    if group != groups_used[0]:
                        fval |= 0x01
                    fval |= 1 << 11  # ext_sync
                apb.write_ctl(group, tc.dev.REG_CTL, val | rdy_sel << 1
                              | fval | groups_used[0] << 9,
                              comment=f' // Enable quadrant {group}')

            if powerdown:
                unused_groups = [group for group in list(range(tc.dev.P_NUMGROUPS))
                                 if group not in groups_used]
                val2 = 0
                for group in unused_groups:
                    val2 |= 1 << 12 + group
                apb.write_fifo_ctl(tc.dev.AON_CTL, val2 | tc.dev.AON_READY_SEL,
                                   comment=' // AON control')

            if state.snoop_loop:
                for group in groups_used:
                    apb.output('\n', embedded_code)
                    apb.write_lreg(group, hw_layer, tc.dev.LREG_NXTLYR, 0x80,
                                   force_write=True, comment=' // Link Layer')
                    apb.write_ctl(group, tc.dev.REG_SNP1_HIT, 0, force_write=True,
                                  comment=' // Clear match hit accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP1_A1, 0x00200000 | (out_offset[ll] >> 2),
                                  force_write=True,
                                  comment=' // Address snoop 1 register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_X1,
                                  0x000084d0 if pipeline else 0x00002134, force_write=True,
                                  comment=' // Snoop 1 match hit accumulator')
                    apb.write_ctl(group, tc.dev.REG_SNP1_C2, 0x00004000, force_write=True,
                                  comment=' // Snoop 1 control register 2')
                    apb.write_ctl(group, tc.dev.REG_SNP1_C1, 0x8a412014, force_write=True,
                                  comment=' // Snoop 1 control register 1')
                    apb.write_ctl(group, tc.dev.REG_SNP1_C1, 0x8a412015, force_write=True,
                                  comment=' // Snoop 1 control register 1')

                apb.output('\n', embedded_code)
                for group in groups_used:
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
                    fval |= (1 << 11) | (1 << 7)
                    if group != groups_used[0]:
                        fval |= 0x01

                    apb.write_ctl(group, tc.dev.REG_CTL, val | rdy_sel << 1
                                  | fval | groups_used[0] << 9,
                                  comment=f' // Enable quadrant {group}')
                apb.output('\n', embedded_code)

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
            if state.snoop_loop:
                val |= 1 << 7
            apb.write_ctl(groups_used[0], tc.dev.REG_CTL, val | rdy_sel << 1 | 0x01,
                          comment=f' // Master enable quadrant {groups_used[0]}')

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
                        hw_padding[start_layer],
                        csv_file=csv,
                    )

            apb.function_footer()
            # End of input

        # ----------------------------------------------------------------------------------------

        in_map = apb.get_mem()
        latency_data: List[Tuple[int, str, str]] = [(1, 'Startup', '')]

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
                data[0].shape,
                output_shift[ll],
                data,
                output_width=o_width,
                operands=operands[ll],
            )
            assert out_size[0] == d_shape[1] \
                and out_size[1] == d_shape[2] and out_size[2] == d_shape[3]

            return data

        # The data_buf list contains the output of each layer, with the exception of the
        # first element which is the input to layer 0 (so everything is shifted right by one):
        # data_buf[0]: Input to layer 0
        # data_buf[1]: Output of layer 0
        # data_buf[2]: Output of layer 1
        # data_buf[ll + 1]: Output of layer ll
        data_buf = [None] * (layers + 1)
        ll = start_layer
        data_buf[ll] = data

        with console.Progress(start=True) as progress:
            task = progress.add_task(description='Creating network... ', total=layers)
            # Compute layer-by-layer output and chain results into input
            while ll < layers:
                progress.update(task, completed=ll)

                if verbose and tc.dev.SUPPORT_LATENCY_CALC:
                    if not flatten[ll]:
                        hw_in_dim = hw_input_dim[ll]
                        hw_in_chan = input_chan[ll]
                        hw_out_chan = kern_count[ll] // in_expand[ll]
                        hw_out_dim = hw_output_dim[ll]
                    else:
                        hw_in_dim = (hw_input_dim[ll][0] * pool[ll][0],
                                     hw_input_dim[ll][1] * pool[ll][1])
                        hw_in_chan = input_chan[ll]
                        hw_out_chan = kern_count[ll] \
                            // (in_expand[ll] * hw_pooled_dim[ll][0] * hw_pooled_dim[ll][1])
                        hw_out_dim = (hw_output_dim[ll][0] * hw_pooled_dim[ll][0],
                                      hw_output_dim[ll][1] * hw_pooled_dim[ll][1])
                    if tc.dev.REQUIRE_2X_MP_PASSTHROUGH and hw_operator[ll] == op.NONE \
                       and out_expand[ll] > 1 and (pool[ll][0] > 1 or pool[ll][1] > 1):
                        multipass = 2 * in_expand[ll] - 1
                    else:
                        multipass = in_expand[ll]
                    layer_lat, layer_comment = latency.calculate(
                        input_chan=hw_in_chan,
                        input_dim=hw_in_dim,
                        pool=pool[ll],
                        pool_stride=pool_stride[ll],
                        pooled_dim=hw_pooled_dim[ll] if operator[ll] != op.CONVTRANSPOSE2D
                        else (hw_pooled_dim[ll][0] * stride[ll][0],
                              hw_pooled_dim[ll][1] * stride[ll][1]),
                        multipass=multipass,
                        output_chan=hw_out_chan if hw_operator[ll] != op.NONE else output_chan[ll],
                        output_dim=hw_out_dim,
                        kernel_size=hw_kernel_size[ll],
                        padding=hw_padding[ll],
                        num_elements=operands[ll],
                        pool_first=pool_first[ll],
                        passthrough=hw_operator[ll] == op.NONE,
                        pass_out_chan=timeslots[ll],
                        flatten=hw_flatten[ll],
                        streaming=streaming[ll],
                        kern_offs=kern_offs[ll],
                    )
                    if streaming[ll]:
                        layer_lat *= -1
                    latency_data.append((layer_lat, f'Layer {layer_str(ll)}', layer_comment))

                compute.debug_open(ll, base_directory, test_name, log_filename)

                # Concatenate input data if needed
                if in_sequences[ll] is not None:
                    if len(in_sequences[ll]) > 1:
                        err_concat = None
                        try:
                            data = np.concatenate([data_buf[i + 1] for i in in_sequences[ll]],
                                                  axis=0)
                        except ValueError as err:
                            err_concat = err
                        if err_concat is not None:
                            try:
                                data = np.hstack(
                                    [data_buf[i + 1].reshape(data_buf[i + 1].shape[0], -1)
                                     for i in in_sequences[ll]]
                                ).reshape(data_buf[in_sequences[ll][0] + 1].shape[0],
                                          input_dim[ll][0], input_dim[ll][1])
                            except ValueError as err:
                                eprint(f'{layer_pfx(ll)}Input data concatenation unsuccessful: ',
                                       err_concat, err)
                    else:
                        data = data_buf[in_sequences[ll][0] + 1]
                else:
                    data = data_buf[ll]

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

                if datafile is not None:
                    # Log input to npy
                    np.save(datafile, data, allow_pickle=False, fix_imports=False)

                show_data(
                    ll,
                    data.shape,
                    data,
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
                    data[0].shape,
                    pool[ll],
                    pool_stride[ll],
                    pool_average[ll],
                    data,
                    dilation=pool_dilation[ll],
                    expand=in_expand[ll],
                    expand_thresh=in_expand_thresh[ll],
                    operation=operator[ll],
                    operands=data.shape[0],
                    rounding=avg_pool_rounding,
                    debug_data=None if not log_pooling else os.path.join(base_directory,
                                                                         test_name),
                )

                if datafile is not None:
                    # Pooling output (pre-elementwise)
                    if pool[ll][0] > 1 or pool[ll][1] > 1 \
                       or pool_stride[ll][0] > 1 or pool_stride[ll][1] > 1 \
                       or pool_dilation[ll][0] > 1 or pool_dilation[ll][1] > 1:
                        np.save(datafile, data, allow_pickle=False, fix_imports=False)
                    else:
                        np.save(datafile, np.empty((0)), allow_pickle=False, fix_imports=False)

                if operator[ll] == op.CONV1D:
                    if out_size[0] != in_chan \
                       or out_size[1] != pooled_dim[ll][0] or pooled_dim[ll][1] != 1:
                        eprint(f'{layer_pfx(ll)}Input dimensions do not match. '
                               f'Expected: {in_chan}x{pooled_dim[ll][0]}, '
                               f'got {out_size[0]}x{out_size[1]}.')
                else:
                    if out_size[0] != in_chan \
                       or out_size[1] != pooled_dim[ll][0] or out_size[2] != pooled_dim[ll][1]:
                        eprint(f'{layer_pfx(ll)}Input dimensions do not match. '
                               f'Expected: {in_chan}x{pooled_dim[ll][0]}x{pooled_dim[ll][1]}, '
                               f'got {out_size[0]}x{out_size[1]}x{out_size[2]}.')

                if operands[ll] > 1 and pool_first[ll]:
                    data = run_eltwise(data, ll)
                else:
                    data = np.squeeze(data, axis=0)

                if datafile is not None:
                    # if operands[ll] > 1 and pool_first[ll]:
                    np.save(datafile, data, allow_pickle=False, fix_imports=False)
                    # else:
                    #    np.save(datafile, np.empty((0)), allow_pickle=False, fix_imports=False)

                # Convolution or passthrough
                if operator[ll] in [op.CONV2D, op.LINEAR]:
                    if flatten[ll]:
                        in_chan *= pooled_dim[ll][0] * pooled_dim[ll][1]
                        data = data.reshape(in_chan, 1, 1)
                        if verbose:
                            print_data(
                                verbose,
                                f'FLATTEN TO {in_chan}x1x1',
                                data,
                                data.shape,
                                1,
                                in_chan,
                            )

                    if not bypass[ll]:
                        k = kernel[kernel_ptrs[ll]].reshape(
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

                    out_buf, out_size = conv2d_layer(
                        ll,
                        data.shape,
                        kernel_size[ll],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll],
                        dilation[ll],
                        stride[ll],
                        activation[ll],
                        k,
                        bias[bias_ptrs[ll]],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                        bypass=bypass[ll],
                        datafile=datafile,
                    )
                elif operator[ll] == op.CONVTRANSPOSE2D:
                    if not bypass[ll]:
                        k = kernel[kernel_ptrs[ll]].reshape(
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
                        data.shape,
                        kernel_size[ll],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll],
                        dilation[ll],
                        stride[ll],
                        output_padding[ll],
                        activation[ll],
                        k,
                        bias[bias_ptrs[ll]],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                        bypass=bypass[ll],
                        datafile=datafile,
                    )
                elif operator[ll] == op.CONV1D:
                    if not bypass[ll]:
                        k = kernel[kernel_ptrs[ll]].reshape(
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
                        data.shape,
                        kernel_size[ll][0],
                        output_shift[ll],
                        output_chan[ll],
                        padding[ll][0],
                        dilation[ll][0],
                        stride[ll][0],
                        activation[ll],
                        k,
                        bias[bias_ptrs[ll]],
                        data,
                        output_width=output_width[ll],
                        groups=conv_groups[ll],
                        bypass=bypass[ll],
                        datafile=datafile,
                    )
                elif operator[ll] == op.NONE:  # '0'D (pooling only or passthrough)
                    out_buf, out_size = passthrough_layer(
                        ll,
                        data.shape,
                        data,
                        datafile=datafile,
                    )
                else:
                    eprint(f'Unknown operator `{op.string(operator[ll])}`.')

                if operator[ll] in [op.CONV2D, op.LINEAR, op.CONVTRANSPOSE2D, op.CONV1D]:
                    if weightsfile is not None:
                        np.save(
                            weightsfile,
                            hw_kernel[ll].reshape(
                                output_chan[ll],
                                input_chan[ll] // conv_groups[ll],
                                hw_kernel_size[ll][0],
                                -1,
                            ),
                            allow_pickle=False,
                            fix_imports=False,
                        )
                    if biasfile is not None:
                        if bias[bias_ptrs[ll]] is not None:
                            np.save(biasfile, bias[bias_ptrs[ll]], allow_pickle=False,
                                    fix_imports=False)
                        else:
                            np.save(biasfile, np.empty((0)), allow_pickle=False, fix_imports=False)
                else:
                    if weightsfile is not None:
                        np.save(weightsfile, np.empty((0)), allow_pickle=False, fix_imports=False)
                    if biasfile is not None:
                        np.save(biasfile, np.empty((0)), allow_pickle=False, fix_imports=False)

                if datafile is not None:
                    # Operator output
                    np.save(datafile, out_buf, allow_pickle=False, fix_imports=False)

                assert out_size[0] == output_chan[ll] \
                    and out_size[1] == output_dim[ll][0] and out_size[2] == output_dim[ll][1]
                assert out_size[0] == output_size[ll][0] \
                    and out_size[1] == output_size[ll][1] and out_size[2] == output_size[ll][2]

                # Write .mem file for output or create the C check_output() function to
                # verify the output
                out_map = datamem.allocate()
                if block_mode:
                    if ll == terminating_layer:
                        filename = output_filename + '.mem'  # Final output
                    else:
                        filename = f'{output_filename}-{ll}.mem'  # Intermediate output
                    filemode = 'w'
                else:
                    if output_layer[ll] or ll == terminating_layer:
                        filename = c_filename + ('_riscv' if riscv else '') + '.c'  # Final output
                    else:
                        filename = None  # Intermediate output - used for layer overwrite check
                    filemode = 'a'

                try:
                    if filename:
                        memfile = open(os.path.join(base_directory, test_name, filename),
                                       mode=filemode, encoding='utf-8')
                    else:
                        memfile = None
                    apb.set_memfile(memfile)

                    if state.generate_kat:
                        if log_intermediate:
                            apb2 = apbaccess.apbwriter(
                                memfile2,
                                verify_writes=False,
                                embedded_code=False,
                                write_zero_registers=True,
                                master=groups_used[0] if oneshot > 0 or stopstart else False,
                                riscv=None,
                                fast_fifo=False,
                                input_chan=input_chan[start_layer],
                                debug_mem=True,
                                test_name=test_name,
                                passfile=passfile,
                            )
                            out_map2 = datamem.allocate()
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
                                mlator=False,  # mlator is a function of read, not write
                                write_gap=write_gap[ll],
                                rollover=rollover[ll + 1] if ll < layers - 1 else None,
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
                            overwrite_ok,
                            mlator=mlator and output_layer[ll],
                            write_gap=write_gap[ll],
                            unload_layer=output_layer[ll],
                            streaming=streaming[ll],
                        )
                        if debug_snoop:
                            apb.verify_ctl(group, tc.dev.REG_SNP1_ACC, None, snoop[24],
                                           comment=' // Verify snoop 1 data accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP1_HIT, None, snoop[25],
                                           comment=' // Verify snoop 1 match hit accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP1_MAX, None, snoop[26],
                                           comment=' // Verify snoop 1 match max accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP1_AM, None, snoop[27],
                                           comment=' // Verify snoop 1 match address register')
                            apb.verify_ctl(group, tc.dev.REG_SNP2_ACC, None, snoop[28],
                                           comment=' // Verify snoop 2 data accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP2_HIT, None, snoop[29],
                                           comment=' // Verify snoop 2 match hit accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP2_MAX, None, snoop[30],
                                           comment=' // Verify snoop 2 match max accumulator')
                            apb.verify_ctl(group, tc.dev.REG_SNP2_AM, None, snoop[31],
                                           comment=' // Verify snoop 2 match address register')
                finally:
                    if memfile:
                        memfile.close()

                if not np.any(out_buf):
                    wprint(f'{layer_pfx(ll)}All output values for the given sample input are '
                           'zero. The generated known-answer test for this network may not be '
                           'meaningful. See the log file for details.')

                if next_sequence[ll] != -1 and streaming[next_sequence[ll]]:
                    # When streaming, the output should not overwrite the input of prior layers
                    # since these layers are still needed.
                    datamem.combine(in_map, out_map)
                else:
                    # Else, preserve the output map of all prior layers marked 'output' plus
                    # the current layer.
                    if output_layer[ll]:
                        # Add this output to the map of all output layers
                        if all_outputs_map is None:
                            all_outputs_map = out_map
                        else:
                            datamem.combine(all_outputs_map, out_map)
                        # Since this layer is an output layer, in_map is the same
                        in_map = all_outputs_map
                    else:
                        # Take the map of all previous output layers, and add this layer to in_map
                        if all_outputs_map is None:
                            in_map = out_map
                        else:
                            in_map = np.array(all_outputs_map, copy=True)
                            datamem.combine(in_map, out_map)

                compute.debug_close()

                if simulated_sequence[ll] is not None:
                    if simulated_sequence[ll] == -1:
                        break
                    ll = simulated_sequence[ll]
                else:
                    if next_sequence[ll] == -1:
                        break
                    ll = next_sequence[ll]

                data_buf[ll] = out_buf.reshape(out_size)

            progress.update(task, completed=layers)

        data = data_buf[ll]

        try:
            if filename:
                memfile = open(os.path.join(base_directory, test_name, filename),
                               mode=filemode, encoding='utf-8')
            else:
                memfile = None
            apb.set_memfile(memfile)

            if state.generate_kat:
                l_str = ", ".join([layer_str(i) for i, e in enumerate(output_layer) if e])
                apb.output(f'// Expected output of layer {l_str} for {test_name} '
                           'given the sample input (known-answer test)\n'
                           '// Delete this function for production code\n')
                if sampleoutput_header is not None:
                    apb.output('static const uint32_t sample_output[] = SAMPLE_OUTPUT;\n')
                apb.function_header(dest='wrapper', prefix='', function='check_output')

                apb.verify_unload_finalize()
                apb.function_footer(dest='wrapper')  # check_output()
        finally:
            if memfile:
                memfile.close()
            if memfile2:
                memfile2.close()
            if datafile:
                datafile.close()
            if weightsfile:
                weightsfile.close()
            if biasfile:
                biasfile.close()
            if passfile:
                passfile.close()

        if log_intermediate:
            with open(os.path.join(base_directory, test_name,
                                   f'{state.output_config_filename}.csv'),
                      mode='w', encoding='utf-8') as configfile:
                # Save layer configuration to debug file
                for ll in range(first_layer_used, layers):
                    if pool[ll][0] <= 1 and pool[ll][1] <= 1:
                        pool_type = 'no'
                    else:
                        pool_type = 'avg' if pool_average[ll] else 'max'

                    configfile.write(
                        f'l,{ll},'
                        f'{tc.dev.INSTANCE_SIZE:x},'
                        f'{1 if streaming[ll] else 0},'
                        f'{0 if rollover[ll] is None else rollover[ll]:x},'
                        f'{output_width[ll]},'
                        f'{processor_map[ll]:x},'
                        f'{flatten_prod[ll]:x},'
                        f'{1 if hw_flatten[ll] else 0},'
                        f'{1 if flatten[ll] else 0},'
                        f'{op.string(operator[ll])},'
                        f'{op.string(hw_operator[ll])},'
                        f'{op.string(eltwise[ll], elt=True)},'
                        f'{in_expand[ll]},'
                        f'{input_chan[ll]},'
                        f'{hw_input_dim[ll][0]}x{hw_input_dim[ll][1]},'
                        f'{hw_pooled_dim[ll][0]}x{hw_pooled_dim[ll][1]},'
                        f'{hw_output_dim[ll][0]}x{hw_output_dim[ll][1]},'
                        f'{pool_type},'
                        f'{1 if pool_first[ll] else 0},'
                        f'{pool[ll][0]}x{pool[ll][1]},'
                        f'{pool_stride[ll][0]}x{pool_stride[ll][1]},'
                        f'{pool_dilation[ll][0]}x{pool_dilation[ll][1]},'
                        f'{conv_groups[ll]},'
                        f'{hw_padding[ll][0]}x{hw_padding[ll][1]},'
                        f'{hw_kernel_size[ll][0]}x{hw_kernel_size[ll][1]},'
                        f'{stride[ll][0]}x{stride[ll][1]},'
                        f'{hw_dilation[ll][0]}x{hw_dilation[ll][1]},'
                        f'{out_pad[ll]},'
                        f'{output_padding[ll][0]}x{output_padding[ll][1]}\n'
                    )

        # ----------------------------------------------------------------------------------------
        total = 0
        lat_unknown = False
        if tc.dev.SUPPORT_LATENCY_CALC:
            if verbose:
                print('ESTIMATED LATENCY')
            for layer_lat, layer_name, layer_comment in latency_data:
                total += abs(layer_lat)
                layer_lat_str = f'{abs(layer_lat):18,}'
                if layer_lat <= 0:
                    lat_unknown = True
                    layer_lat_str += ' (est)'
                if verbose:
                    print(f'{layer_name:9}{layer_lat_str}')
                    if state.debug_latency and layer_comment != '':
                        print(f'\n{layer_comment}')
            total_str = f'{total:22,} cycles'
            if lat_unknown:
                total_str += ' (est)'
            if verbose:
                print('                 ==========\n'
                      f'Total{total_str}\n')

            if not (embedded_code or block_mode or any(streaming)):
                rtlsim.write_latency(
                    test_name,
                    total,
                    [x for x, _, _ in latency_data],
                )
        # ----------------------------------------------------------------------------------------

        if not block_mode:
            with open(os.path.join(base_directory, test_name, filename), mode=filemode,
                      encoding='utf-8') as memfile:
                apb.set_memfile(memfile)

                if state.softmax or embedded_code and state.unload:
                    apb.unload(
                        output_layer=output_layer,
                        processor_map=output_processor_map,
                        input_shape=output_size,
                        output_offset=out_offset,
                        out_expand=out_expand,
                        out_expand_thresh=out_expand_thresh,
                        output_width=output_width,
                        write_gap=write_gap,
                    )

                if state.softmax:
                    apb.softmax_layer(
                        output_width=output_width[terminating_layer],
                        shift=8 - abs(quantization[terminating_layer])
                        if not bypass[terminating_layer] else 0,
                    )

                summary_stats = '/*\n' + \
                                stats.summary(factor=repeat_layers, spaces=2,
                                              group_bias_max=group_bias_max) + \
                                '*/\n'
                apb.main()
                apb.output(summary_stats + '\n')

        # Close header files
        if sampledata_header is not None:
            sampledata_header.close()
        if sampleoutput_header is not None:
            sampleoutput_header.close()
        if apifile is not None:
            apifile.close()
        if state.rtl_preload or state.rtl_preload_weights \
           or result_output or state.new_kernel_loader:
            apb.write_mem(base_directory, test_name)
        if weight_header is not None:
            weight_header.close()

        # Create run_test.sv
        if not embedded_code and not block_mode:
            rtlsim.create_runtest_sv(
                test_name,
                timeout,
                groups_used=groups_used,
                cnn_cycles=total,
                apb=apb,
                input_dim=hw_input_dim,
                in_expand=in_expand,
            )
            assets.copy('assets', 'rtlsim-ai' + str(device), base_directory, test_name)
            if riscv_cache:
                assets.copy('assets', 'rtlsim-riscv-cache-ai' + str(device), base_directory,
                            test_name)
            elif riscv_flash:
                assets.copy('assets', 'rtlsim-riscv-flash-ai' + str(device), base_directory,
                            test_name)
            elif riscv:
                assets.copy('assets', 'rtlsim-riscv-ai' + str(device), base_directory, test_name)
            if result_output:
                assets.copy('assets', 'rtlsim-verify-output', base_directory, test_name)
        elif block_mode:
            assets.copy('assets', 'blocklevel-ai' + str(device), base_directory, test_name)
        elif embedded_code:
            output_count = output_chan[terminating_layer] \
                * output_dim[terminating_layer][0] * output_dim[terminating_layer][1]
            insert = summary_stats + \
                '\n/* Number of outputs for this network */\n' \
                f'#define CNN_NUM_OUTPUTS {output_count}'
            if timer is not None:
                insert += '\n\n/* Use this timer to time the inference */\n' \
                          f'#define CNN_INFERENCE_TIMER MXC_TMR{timer}'

            if riscv:
                assets.from_template('assets', 'embedded-riscv-ai' + str(device), base_directory,
                                     test_name, board_name)
            else:
                assets.from_template('assets', 'embedded-ai' + str(device), base_directory,
                                     test_name, board_name)

                assets.makefile(
                    base_directory,
                    test_name,
                    tc.dev.partnum,
                    board_name,
                    overwrite=overwrite,
                )

            assets.from_template('assets', 'eclipse', base_directory, test_name, board_name)

            # Generate VS Code project files
            if riscv:
                # RISC-V projects need some slight tweaking.
                # The combined .elf file that is created doesn't
                # have any debug symbols.  The M4 and RISC-V
                # "split" files do have symbols, so we point
                # the debugger to those and flash the combined file.
                assets.vscode(
                    base_directory,
                    test_name,
                    tc.dev.partnum,
                    board_name,
                    program_file="${config:project_name}-combined.elf",
                    symbol_file="${config:project_name}.elf",
                    overwrite=overwrite,
                )

            else:
                assets.vscode(
                    base_directory,
                    test_name,
                    tc.dev.partnum,
                    board_name,
                    overwrite=overwrite,
                )

            assets.from_template('assets', 'device-all', base_directory,
                                 test_name, board_name, insert=insert)
            assets.from_template('assets', 'device-ai' + str(device), base_directory,
                                 test_name, board_name)

        print(stats.summary(factor=repeat_layers, group_bias_max=group_bias_max))

        return test_name
