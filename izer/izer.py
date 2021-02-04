###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Embedded network and simulation test generator program for Tornado CNN
"""
import os

import numpy as np

from . import (checkpoint, cmsisnn, commandline, devices, max7800x, onnxcp, op, rtlsim, sampledata,
               sampleweight, stats)
from . import tornadocnn as tc
from . import yamlcfg
from .eprint import eprint, wprint


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=np.inf, linewidth=190)

    args = commandline.get_parser()

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
    cfg, cfg_layers, params = yamlcfg.parse(args.config_file)

    # If not using test data, load weights and biases
    # This also configures the network's output channels
    if cfg['arch'] != 'test':
        if not args.checkpoint_file:
            eprint("--checkpoint-file is a required argument.")
        fext = args.checkpoint_file.rsplit(sep='.', maxsplit=1)[1].lower()
        if fext == 'onnx':
            # ONNX file selected
            layers, weights, bias, output_shift, \
                input_channels, output_channels = \
                onnxcp.load(
                    args.checkpoint_file,
                    cfg['arch'],
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
            layers, weights, bias, output_shift, \
                input_channels, output_channels = \
                checkpoint.load(
                    args.checkpoint_file,
                    cfg['arch'],
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
        layers, weights, output_shift, \
            input_channels, output_channels = \
            sampleweight.load(
                cfg['dataset'],
                params['quantization'],
                params['output_shift'],
                cfg_layers,
                cfg['weights'] if 'weights' in cfg else None,
                params['conv_groups'],
                params['operator'],
            )
        bias = sampleweight.load_bias(
            cfg_layers,
            cfg['bias'] if 'bias' in cfg else None,
            args.no_bias,
        )

    if cfg_layers > layers:
        # Add empty weights/biases and channel counts for layers not in checkpoint file.
        # The checkpoint file does not contain weights for non-convolution operations.
        # Insert empty input channels/output channels/weights/biases and increase `layers`
        # accordingly.
        for ll in range(cfg_layers):
            operator = params['operator'][ll]

            if operator == op.NONE or op.eltwise(operator) or params['bypass'][ll]:
                weights.insert(ll, None)
                if not params['bypass'][ll]:
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

    processor_map = params['processor_map'][:layers]
    output_processor_map = params['output_processor_map'][:layers]
    in_sequences = params['in_sequences'][:layers]
    next_sequence = params['next_sequence'][:layers]
    prev_sequence = [-1] * layers

    # Override channels, establish sequence
    for ll in range(layers - 1):
        if next_sequence[ll] is None:
            next_sequence[ll] = ll + 1  # Assign default next layer as sequential
    if next_sequence[layers - 1] is None:
        next_sequence[layers - 1] = -1
    prev_ll = -1
    final_layer = None
    max_layer = args.start_layer
    min_layer = args.start_layer
    ll = args.start_layer
    while ll < layers:
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
            input_channels[ll] = output_channels[prev_ll]
        if params['input_chan'][ll] is not None:
            input_channels[ll] = params['input_chan'][ll]
        if output_channels[ll] <= 0:
            output_channels[ll] = input_channels[ll]
        if params['output_chan'][ll] is not None:
            output_channels[ll] = params['output_chan'][ll]

        # Fix up default output maps
        if output_processor_map[ll] is None \
           and next_sequence[ll] != -1 and next_sequence[ll] < layers:
            output_processor_map[ll] = processor_map[next_sequence[ll]]

        if args.stop_after is not None and ll == args.stop_after:
            next_sequence[ll] = -1

        prev_sequence[ll] = prev_ll
        prev_ll = ll
        max_layer = max(ll, max_layer)
        min_layer = min(ll, min_layer)
        if next_sequence[ll] != -1 and next_sequence[ll] != ll + 1 \
           and not tc.dev.SUPPORT_LINK_LAYER:
            eprint(f"Layer {ll}: `next_sequence` is not supported on this device.")
        elif next_sequence[ll] > layers:
            wprint(f"Layer {ll}: `next_sequence` exceeds available layers, setting to `stop`.")
            next_sequence[ll] = -1
        if next_sequence[ll] == -1:
            final_layer = ll
            break

        ll = next_sequence[ll]

    layers = max_layer + 1
    if final_layer is None:
        final_layer = max_layer

    if tc.dev.USE_PROCESSORS:
        if 'output_map' in cfg:
            # Use optional configuration value if it's specified
            output_processor_map[final_layer] = cfg['output_map']
        elif output_processor_map[final_layer] is None:
            # Default to packed, 0-aligned output map
            expand = (output_channels[final_layer] + tc.dev.MAX_PROC-1) // tc.dev.MAX_PROC
            expand_chunk = (output_channels[final_layer] + expand-1) // expand
            if output_channels[final_layer] > tc.dev.MAX_PROC:
                expand_chunk = min((expand_chunk + tc.dev.P_SHARED-1) & ~(tc.dev.P_SHARED-1),
                                   tc.dev.MAX_PROC)
            output_processor_map[final_layer] = 2**expand_chunk-1

    # Remove extraneous layer configuration values (when --stop-after is used)
    processor_map = processor_map[:layers]
    output_processor_map = output_processor_map[:layers]
    next_sequence = next_sequence[:layers]
    prev_sequence = prev_sequence[:layers]

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
    bypass = params['bypass'][:layers]
    bias_group_map = params['bias_group_map'][:layers]
    calcx4 = [True] * layers if args.calcx4 else params['calcx4'][:layers]
    readahead = [True] * layers if args.rd_ahead else params['readahead'][:layers]
    pool_dilation = params['pool_dilation'][:layers]

    # Command line override
    if args.input_offset is not None:
        input_offset[args.start_layer] = args.input_offset

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

    ll = args.start_layer
    auto_input_dim[ll] = [input_size[1], input_size[2]]
    if conf_input_dim[ll] is None:
        input_dim[ll] = auto_input_dim[ll]
    else:
        input_dim[ll] = conf_input_dim[ll]
    if input_offset[ll] is None:
        input_offset[ll] = 0

    if not tc.dev.SUPPORT_ARBITRARY_OUTPUT_WIDTH:
        # Check last layer
        if output_width[final_layer] != 8 and activation[final_layer] is not None:
            eprint(f'`output_width` must be 8 when activation is used in (layer {ll}).')

    while ll < layers:
        if input_channels[ll] <= 0:
            eprint(f'Must specify `in_channels` for layer {ll}.')
        if quantization[ll] is None:
            quantization[ll] = 8 if not bypass[ll] and operator[ll] != op.NONE else 0  # Defaults
        if operator[ll] != op.NONE and not bypass[ll]:
            if quantization[ll] == -1:
                w = np.abs(weights[ll])
                assert w.min() == w.max() == 1
            else:
                assert weights[ll].min() >= -1 << quantization[ll] - 1
                assert weights[ll].max() <= (1 << quantization[ll] - 1) - 1

        # Check all but first layer
        if ll != args.start_layer:
            # Fix up default input maps
            if input_offset[ll] is None:
                input_offset[ll] = output_offset[prev_sequence[ll]]
            # Check we don't turn on streaming too late
            if streaming[ll] and not streaming[prev_sequence[ll]]:
                eprint(f'Enable streaming from the first layer on (found in layer {ll}.')
            if big_data[ll]:
                eprint(f'`data_format` in layer {ll}: CHW can only be configured for the '
                       'first layer.')

        # Check all but last layer
        if ll != final_layer:
            if output_width[ll] != 8:
                eprint(f'`output_width` must be 8 for intermediate layer {ll}.')

        if in_sequences[ll] is not None:
            if tc.dev.SUPPORT_LINK_LAYER:
                if isinstance(in_sequences[ll], list) \
                   and any([(i > len(in_sequences)) for i in in_sequences[ll]]) \
                   or not isinstance(in_sequences[ll], list) \
                   and in_sequences[ll] > final_layer:
                    eprint(f'`in_sequences` in layer {ll} cannot be greater than the last layer.')
            else:
                if isinstance(in_sequences[ll], list) \
                   and any([(i >= ll) for i in in_sequences[ll]]) \
                   or not isinstance(in_sequences[ll], list) \
                   and in_sequences[ll] >= ll:
                    eprint(f'`in_sequences` in layer {ll} cannot be greater than layer sequence '
                           'on this device')

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
                auto_input_dim[ll] = output_dim[prev_sequence[ll]]
            if conf_input_dim[ll] is None:
                input_dim[ll] = auto_input_dim[ll]
            else:
                input_dim[ll] = conf_input_dim[ll]
        if operator[ll] != op.CONV1D:
            if pool_stride[ll][0] != pool_stride[ll][1]:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support non-square '
                       f'pooling stride (currently set to '
                       f'{pool_stride[ll][0]}x{pool_stride[ll][1]}).')
            pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0]
                            - pool_dilation[ll][0] + 1) // pool_stride[ll][0],
                           (input_dim[ll][1] + pool_stride[ll][1] - pool[ll][1]
                            - pool_dilation[ll][1] + 1) // pool_stride[ll][1]]
        else:
            pooled_size = [(input_dim[ll][0] + pool_stride[ll][0] - pool[ll][0]
                            - pool_dilation[ll][0] + 1) // pool_stride[ll][0],
                           1]

        pooled_dim[ll] = pooled_size
        if any(dim == 0 for dim in pooled_dim[ll]):
            eprint(f'Pooling or zero-padding in layer {ll} results in a zero data dimension '
                   f'(input {input_dim[ll]}, result {pooled_dim[ll]}).')

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
            if padding[ll][0] >= 3 and not tc.dev.SUPPORT_ARBITRARY_PADDING:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                       f'(currently set to {padding[ll][0]}).')
        else:
            # We don't have to consider padding for the width calculation,
            # since padding has to be a multiple of 3 and we check for that.
            if padding[ll][0] >= 3 and not tc.dev.SUPPORT_ARBITRARY_PADDING:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support `pad` >= 3 '
                       f'(currently set to {padding[ll][0]}).')
            if stride[ll][0] != 1 and not tc.dev.SUPPORT_ARBITRARY_STRIDE:
                eprint(f'{op.string(operator[ll])} in layer {ll} does not support stride other '
                       f'than 1 (currently set to {stride[ll][0]}).')
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

        ll = next_sequence[ll]
        if ll == -1:
            break

    if args.riscv and not args.riscv_cache and args.embedded_code:
        eprint("Embedded code on RISC-V requires --riscv-cache.")

    if tc.dev.device != devices.CMSISNN:
        tn = max7800x.create_net(
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
            args.input_split,
            input_offset,
            output_offset,
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
            verify_writes=args.verify_writes,
            verify_kernels=args.verify_kernels,
            embedded_code=args.embedded_code,
            compact_weights=args.compact_weights,
            compact_data=args.compact_data and not args.rtl_preload,
            write_zero_regs=args.write_zero_registers,
            weight_filename=args.weight_filename,
            sample_filename=args.sample_filename,
            init_tram=args.init_tram,
            avg_pool_rounding=args.avg_pool_rounding,
            fifo=args.fifo,
            fast_fifo=args.fast_fifo,
            fast_fifo_quad=args.fast_fifo_quad,
            zero_sram=args.zero_sram,
            mlator=args.mlator,
            oneshot=args.one_shot,
            ext_rdy=args.ext_rdy,
            stopstart=args.stop_start,
            mexpress=args.mexpress,
            riscv=args.riscv,
            riscv_exclusive=args.riscv_exclusive,
            riscv_flash=args.riscv_flash,
            riscv_cache=args.riscv_cache,
            riscv_debug=args.riscv_debug,
            debugwait=args.debugwait,
            override_start=args.override_start,
            increase_start=args.increase_start,
            override_rollover=args.override_rollover,
            override_delta1=args.override_delta1,
            increase_delta1=args.increase_delta1,
            override_delta2=args.override_delta2,
            increase_delta2=args.increase_delta2,
            slow_load=args.slow_load,
            synthesize_input=args.synthesize_input,
            mlator_noverify=args.mlator_noverify,
            input_csv=args.input_csv,
            input_csv_period=args.input_csv_period,
            input_csv_format=args.input_csv_format,
            input_csv_retrace=args.input_csv_retrace,
            input_fifo=args.input_fifo,
            input_sync=args.input_sync,
            sleep=args.deepsleep,
            powerdown=args.powerdown,
            simple1b=args.simple1b,
            legacy_test=args.legacy_test,
            legacy_kernels=args.legacy_kernels,
            log_intermediate=args.log_intermediate,
            log_pooling=args.log_pooling,
            allow_streaming=args.allow_streaming,
            softmax=args.softmax,
            clock_trim=args.clock_trim,
            repeat_layers=args.repeat_layers,
            fixed_input=args.fixed_input,
            max_count=args.max_count,
            boost=args.boost,
            forever=args.forever,
            write_gap=write_gap,
            start_layer=args.start_layer,
            first_layer_used=min_layer,
            final_layer=final_layer,
            pipeline=args.pipeline,
            pll=args.pll,
            reshape_inputs=args.reshape_inputs,
            link_layer=args.link_layer,
            measure_energy=args.energy,
            timer=args.timer,
            board_name=args.board_name,
            rd_ahead=readahead,
            calcx4=calcx4,
            rtl_preload=args.rtl_preload,
            result_output=args.result_output,
            weight_start=args.weight_start,
            wfi=args.wfi,
            bypass=bypass,
            bias_group_map=bias_group_map,
            pool_dilation=pool_dilation,
            input_pix_clk=args.input_pix_clk,
            fifo_go=args.fifo_go,
            pretend_zero_sram=args.pretend_zero_sram,
        )
        if not args.embedded_code and args.autogen.lower() != 'none':
            rtlsim.append_regression(
                args.top_level,
                tn,
                args.queue_name,
                args.autogen,
                args.autogen_list,
            )
    else:
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
            args.legacy_test,
        )

        print(stats.summary(debug=args.debug, weights=weights, w_size=quantization, bias=bias))
