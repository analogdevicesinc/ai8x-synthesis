###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Embedded network and simulation test generator program for Tornado CNN
"""
import os
import sys
import time
from pydoc import locate

import numpy as np

import rich.console

from . import checkpoint, commandline, console, onnxcp, op, rtlsim, sampledata, sampleweight, state
from . import tornadocnn as tc
from . import versioncheck, yamlcfg
from .eprint import eprint, wprint
from .names import layer_pfx, layer_str
from .utils import plural


def main():
    """
    Command line wrapper
    """
    np.set_printoptions(threshold=sys.maxsize, linewidth=190)

    # Save stdout before colorama potentially wraps it
    state.output_is_console = sys.stdout is not None and sys.stdout.isatty()
    saved_stdout = sys.stdout
    console.stderr = rich.console.Console(stderr=True)

    args = commandline.get_parser()

    # Check whether code is up-to-date
    if not args.no_version_check:
        now = round(time.time())
        last_check = versioncheck.get_last_check()
        if now - last_check >= args.version_check_interval * 60 * 60:
            if versioncheck.check_repo(args.upstream):
                # Check succeeded, don't check again for a while
                versioncheck.set_last_check(now)

    # Configure device and set device dependent state
    tc.dev = tc.get_device(args.device)

    # Manipulate device defaults based on command line (FIXME: this should go into state)
    if args.max_proc:
        tc.dev.MAX_PROC = args.max_proc
        tc.dev.P_NUMPRO = min(args.max_proc, tc.dev.P_NUMPRO)
        tc.dev.P_NUMGROUPS = (tc.dev.MAX_PROC + tc.dev.P_NUMPRO - 1) // tc.dev.P_NUMPRO
    if args.ready_sel:
        tc.dev.READY_SEL = args.ready_sel
    if args.ready_sel_fifo:
        tc.dev.FIFO_READY_SEL = args.ready_sel_fifo
    if args.ready_sel_aon:
        tc.dev.AON_READY_SEL = args.ready_sel_aon
    if args.new_kernel_loader:
        args.compact_weights = False
    if args.enable_delay is None:
        args.enable_delay = tc.dev.DEFAULT_SWITCH_DELAY if args.embedded_code else 0

    # Change global state based on command line
    commandline.set_state(args)

    # Load configuration file
    cfg, cfg_layers, params = yamlcfg.parse(args.config_file, args.skip_yaml_layers)
    state.layer_name = params['layer_name']

    # If not using test data, load weights and biases
    # This also configures the network's output channels
    if cfg['arch'] != 'test':
        if not args.checkpoint_file:
            eprint('--checkpoint-file is a required argument.')
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
                    params['bypass'],
                    params['weight_source'],
                    args.skip_checkpoint_layers,
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
                params['bypass'],
                filename=args.weight_input,
            )
        bias = sampleweight.load_bias(
            cfg_layers,
            cfg['bias'] if 'bias' in cfg else None,
            args.no_bias,
            params['operator'],
            params['bypass'],
            filename=args.bias_input,
        )
    if cfg_layers > layers:
        # Add empty weights/biases and channel counts for layers not in checkpoint file.
        # The checkpoint file does not contain weights for non-convolution operations.
        # Insert empty input channels/output channels/weights/biases and increase layers
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
            else:
                val = params['weight_source'][ll]
                if val is not None:
                    weights.insert(ll, weights[val])
                    bias.insert(ll, bias[val])
                    input_channels.insert(ll, input_channels[val])
                    output_channels.insert(ll, output_channels[val])
                    output_shift[ll] = output_shift[val]
                    layers += 1

    if layers != cfg_layers:
        eprint(f'Number of layers in the YAML configuration file ({cfg_layers}) '
               f'does not match the checkpoint file ({layers}).')

    if any(p < 0 or p >= 4*tc.dev.MEM_SIZE for p in params['output_offset']):
        eprint('Unsupported value for `out_offset` in YAML configuration. Supported on this '
               f'device: 0 to 0x{4*tc.dev.MEM_SIZE:04x}.')

    if any(q != 8 for q in params['bias_quantization']):
        eprint('All bias quantization configuration values must be 8.')

    print(f'Configuring data set: {cfg["dataset"]}.')
    if args.sample_input is None:
        sampledata_file = os.path.join('tests', f'sample_{cfg["dataset"].lower()}.npy')
    else:
        sampledata_file = args.sample_input
    data = sampledata.get(
        sampledata_file,
        synthesize_input=args.synthesize_input,
        synthesize_words=args.synthesize_words,
    )
    if np.max(data) > 127 or np.min(data) < -128:
        eprint(f'Input data {sampledata_file} contains values that are outside the limits of '
               f'signed 8-bit (data min={np.min(data)}, max={np.max(data)})!')
    # Work with 1D input data
    if data.ndim < 3:
        data = np.expand_dims(data, axis=2)

    input_size = list(data.shape)

    processor_map = params['processor_map'][:layers]
    output_processor_map = params['output_processor_map'][:layers]
    in_sequences = params['in_sequences'][:layers]
    next_sequence = params['next_sequence'][:layers]
    prev_sequence = [-1] * layers
    write_gap = params['write_gap'][:layers]
    input_skip = params['input_skip'][:layers]
    operands = params['operands'][:layers]

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
            assert isinstance(in_sequences[ll], list)
            if params['eltwise'][ll] == op.NONE:
                # Concatenate
                input_channels[ll] = sum(input_size[0] if i == -1 else output_channels[i]
                                         for i in in_sequences[ll])
            else:
                # Element-wise operation
                input_channels[ll] = input_size[0] if in_sequences[ll][0] == -1 \
                    else output_channels[in_sequences[ll][0]]
            gap = write_gap[in_sequences[ll][0]]
            chan = output_channels[in_sequences[ll][0]]
            l_str = ", ".join([layer_str(e) for e in in_sequences[ll]])
            l_inseq = len(in_sequences[ll])
            for e in in_sequences[ll]:
                if chan != output_channels[e]:
                    ochan_str = ', '.join(str(output_channels[e]) for e in in_sequences[ll])
                    eprint(f'{layer_pfx(ll)}`in_sequences` [{l_str}] for the element-wise '
                           'operation includes inputs with non-matching channel counts '
                           f'({ochan_str}).')
                if gap != write_gap[e]:
                    wprint(f'{layer_pfx(ll)}`in_sequences` [{l_str}] for the element-wise '
                           'operation includes inputs with non-matching `write_gap` values.')
            if operands[ll] > 1:
                if gap != operands[ll] - 1:
                    wprint(f'{layer_pfx(ll)}The element-wise operation has {operands[ll]} '
                           f'operands, but the `write_gap` is {gap} for the `in_sequences` '
                           f'[{l_str}] instead of {operands[ll] - 1}.')
                if input_skip[ll] != gap // operands[ll]:
                    wprint(f'{layer_pfx(ll)}`read_gap` {input_skip[ll]} does not match the '
                           f'operand count ({operands[ll]}) and `write_gap` {gap} of the '
                           f'{plural(operands[ll], "input")} for this layer.')
            elif gap > 0:
                if gap != l_inseq - 1:
                    wprint(f'{layer_pfx(ll)}The channel interleave operation has {l_inseq} '
                           f'inputs, but the `write_gap` is {gap} for the `in_sequences` '
                           f'[{l_str}] instead of {l_inseq - 1}.')
                if input_skip[ll] != (gap + 1) // l_inseq - 1:
                    wprint(f'{layer_pfx(ll)}`read_gap` {input_skip[ll]} does not match the '
                           f'input count ({l_inseq}) and `write_gap` {gap} of the '
                           f'{plural(l_inseq, "input")} for this layer.')
        else:
            gap = 0 if prev_ll == -1 else write_gap[prev_ll]
            if gap != input_skip[ll]:
                wprint(f'{layer_pfx(ll)}`read_gap` {input_skip[ll]} does not match the '
                       f'`write_gap` {gap} of layer {layer_str(prev_ll)} that created the input '
                       'for this layer.')

        if input_channels[ll] <= 0:
            if ll == 0:
                input_channels[ll] = input_size[0] // params['operands'][0]
            else:
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

        if args.stop_after is not None and ll == args.stop_after - args.skip_yaml_layers:
            next_sequence[ll] = -1

        prev_sequence[ll] = prev_ll
        prev_ll = ll
        max_layer = max(ll, max_layer)
        min_layer = min(ll, min_layer)
        if next_sequence[ll] != -1 and next_sequence[ll] != ll + 1 \
           and not tc.dev.SUPPORT_LINK_LAYER:
            eprint(f'{layer_pfx(ll)}`next_sequence` is not supported on this device.')
        elif next_sequence[ll] > layers:
            wprint(f'{layer_pfx(ll)}`next_sequence` exceeds available layers, '
                   'setting to `stop`.')
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
    output_padding = params['output_padding'][:layers]
    stride = params['stride'][:layers]
    dilation = params['dilation'][:layers]
    big_data = params['big_data'][:layers]
    output_width = params['output_width'][:layers]
    if args.output_width is not None:
        output_width[final_layer] = args.output_width
    operator = params['operator'][:layers]
    if args.ignore_streaming:
        streaming = [False] * layers
    else:
        streaming = params['streaming'][:layers]
    if args.streaming_layers is not None:
        # Additional (or only) streaming layers from command line
        for e in args.streaming_layers:
            streaming[e] = True
    flatten = params['flatten'][:layers]
    eltwise = params['eltwise'][:layers]
    pool_first = params['pool_first'][:layers]
    activation = params['activation'][:layers]
    conv_groups = params['conv_groups'][:layers]
    bypass = params['bypass'][:layers]
    bias_group_map = params['bias_group_map'][:layers]
    calcx4 = [True] * layers if args.calcx4 else params['calcx4'][:layers]
    readahead = [True] * layers if args.rd_ahead else params['readahead'][:layers]
    pool_dilation = params['pool_dilation'][:layers]
    tcalc = params['tcalc'][:layers]
    snoop_sequence = params['snoop_sequence'][:layers]
    simulated_sequence = params['simulated_sequence'][:layers]
    output_layer = params['output_layer'][:layers]
    if not any(output_layer):
        output_layer[final_layer] = True
    elif not output_layer[final_layer]:
        wprint(f'The final layer {layer_str(ll)} is not designated as an `output` layer.')
    # Set this since 'layers' may have changed
    operands = params['operands'][:layers]

    # Command line override
    if args.input_offset is not None:
        input_offset[args.start_layer] = args.input_offset

    # Derived configuration options
    pool_average = [bool(x) for x in params['average']]

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
    output_dim = [(0, 0)] * layers

    avgpool_reset_layer = [False] * layers

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
            eprint(f'{layer_pfx(ll)}`output_width` must be 8 when activation is used.')

    if ll == 0 and (pool[0][0] != 1 or pool[0][1] != 1):
        wprint(f'{layer_pfx(0)}The input layer uses pooling. Consider downscaling the input data '
               'at the source instead.')

    while ll < layers:
        if input_channels[ll] <= 0:
            eprint(f'{layer_pfx(ll)}`in_channels` is required.')
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
            elif in_sequences[ll] is None and input_offset[ll] != output_offset[prev_sequence[ll]]:
                wprint(f'{layer_pfx(ll)}Non-default `in_offset: 0x{input_offset[ll]:04x}`'
                       f', but no `in_sequences` given. Assuming `in_sequences: '
                       f'{layer_str(prev_sequence[ll])}` which may be incorrect.')
            # Check we don't turn on streaming too late
            if streaming[ll] and not streaming[prev_sequence[ll]]:
                eprint(f'Layer {layer_str(ll)} is a streaming layer, but the previous layer '
                       f'{layer_str(prev_sequence[ll])} is non-streaming. This is not supported.')
            if big_data[ll]:
                eprint(f'{layer_pfx(ll)}`data_format` CHW can only be configured for the '
                       'first layer.')

        # Check all but last layer
        next_seq = next_sequence[ll]
        if output_width[ll] != 8 and next_seq != -1 \
           and (in_sequences[next_seq] is None or ll in in_sequences[next_seq]):
            wprint(f'{layer_pfx(ll)}`output_width` should be 8 for intermediate layers.')

        if in_sequences[ll] is not None:
            if tc.dev.SUPPORT_LINK_LAYER:
                if any(i > len(in_sequences) for i in in_sequences[ll]):
                    eprint(f'{layer_pfx(ll)}`in_sequences` cannot be greater than the '
                           'last layer.')
            else:
                if any(i >= ll for i in in_sequences[ll]):
                    eprint(f'{layer_pfx(ll)}`in_sequences` cannot be greater than layer '
                           'sequence on this device')

        if input_dim[ll] is None:
            pixels = 0
            if in_sequences[ll] is not None:
                dim = auto_input_dim[0] if in_sequences[ll][0] == -1 \
                    else output_dim[in_sequences[ll][0]]
                for e in in_sequences[ll]:
                    odim = auto_input_dim[0] if e == -1 else output_dim[e]
                    if odim != dim and conf_input_dim[ll] is None:
                        eprint(f'{layer_pfx(ll)}Cannot concatenate outputs of different '
                               f'dimensions without specifying `in_dim`: {dim} vs {odim}.')
                    pixels += odim[0] * odim[1]
                auto_input_dim[ll] = dim
                prev_op = operator[in_sequences[ll][0]]
            else:
                auto_input_dim[ll] = output_dim[prev_sequence[ll]]
                prev_op = operator[prev_sequence[ll]]
            if conf_input_dim[ll] is None:
                input_dim[ll] = auto_input_dim[ll]
                # Print warning when going from 1D to 2D without explicitly reformatting the input
                if input_dim[ll] is None:
                    eprint(f'{layer_pfx(ll)}No input dimension information. Use '
                           '`in_dim` to explicitly set dimensions.')
                if input_dim[ll][1] == 1 and operator[ll] in [op.CONV2D, op.CONVTRANSPOSE2D] \
                   and prev_op == op.CONV1D:
                    wprint(f'{layer_pfx(ll)}Using 1-dimensional data {input_dim[ll][0]}x'
                           f'{input_dim[ll][1]} with a {op.string(operator[ll])} operator. '
                           'Use `in_dim:` to reshape the data to two dimensions, or to silence '
                           'this message.')
            else:
                if pixels > 0 and conf_input_dim[ll][0] * conf_input_dim[ll][1] != pixels:
                    eprint(f'{layer_pfx(ll)}The configured `in_dim` does not match the '
                           f'sum of all pixels ({pixels}) in the layer\'s `in_sequences`.')
                input_dim[ll] = conf_input_dim[ll]
        if operator[ll] != op.CONV1D:
            if pool_stride[ll][0] != pool_stride[ll][1]:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support '
                       f'non-square pooling stride (currently set to '
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
            eprint(f'{layer_pfx(ll)}Pooling or zero-padding results in a zero data '
                   f'dimension (input {input_dim[ll]}, result {pooled_dim[ll]}).')

        if operator[ll] != op.CONV1D:
            if stride[ll][0] != stride[ll][1]:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support '
                       f'non-square stride (currently set to {stride[ll][0]}x{stride[ll][1]}).')
            if operator[ll] != op.CONVTRANSPOSE2D and stride[ll][0] != 1:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support stride '
                       f'other than 1 (currently set to {stride[ll][0]}x{stride[ll][1]}).')
            if operator[ll] in [op.NONE, op.CONV2D, op.LINEAR]:
                output_dim[ll] = [(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1)
                                   - 1 + 2 * padding[ll][0]) // stride[ll][0] + 1,
                                  (pooled_size[1] - dilation[ll][1] * (kernel_size[ll][1] - 1)
                                   - 1 + 2 * padding[ll][1]) // stride[ll][1] + 1]
            elif operator[ll] == op.CONVTRANSPOSE2D:
                output_dim[ll] = [(pooled_size[0] - 1) * stride[ll][0] - 2 * padding[ll][0]
                                  + dilation[ll][0] * (kernel_size[ll][0] - 1)
                                  + output_padding[ll][0] + 1,
                                  (pooled_size[1] - 1) * stride[ll][1] - 2 * padding[ll][1]
                                  + dilation[ll][1] * (kernel_size[ll][1] - 1)
                                  + output_padding[ll][1] + 1]
            else:  # Element-wise
                output_dim[ll] = [pooled_size[0], pooled_size[1]]
            if flatten[ll]:
                if pooled_dim[ll][0] * pooled_dim[ll][1] > 256:
                    eprint(f'{layer_pfx(ll)}`flatten` exceeds supported input dimensions '
                           f'({pooled_dim[ll][0]} * {pooled_dim[ll][1]} > 256)).')
                if pooled_dim[ll][0] * pooled_dim[ll][1] == 1:
                    wprint(f'{layer_pfx(ll)}`flatten` is not needed since input '
                           'dimensions are 1x1.')
                output_dim[ll] = [1, 1]
                input_channels[ll] //= pooled_dim[ll][0] * pooled_dim[ll][1]
                assert input_channels[ll] > 0
            if padding[ll][0] >= 3 and not tc.dev.SUPPORT_ARBITRARY_PADDING:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support '
                       f'`pad` >= 3 (currently set to {padding[ll][0]}).')
        else:
            if padding[ll][0] >= 3 and not tc.dev.SUPPORT_ARBITRARY_PADDING:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support '
                       f'`pad` >= 3 (currently set to {padding[ll][0]}).')
            if stride[ll][0] != 1 and not tc.dev.SUPPORT_ARBITRARY_STRIDE:
                eprint(f'{layer_pfx(ll)}{op.string(operator[ll])} does not support stride '
                       f'other than 1 (currently set to {stride[ll][0]}).')
            output_dim[ll] = [(pooled_size[0] - dilation[ll][0] * (kernel_size[ll][0] - 1) - 1 +
                               2 * padding[ll][0]) // stride[ll][0] + 1,
                              1]

        # No padding when no convolution...
        if operator[ll] == op.NONE and (padding[ll][0] > 0 or padding[ll][1] > 0):
            eprint(f'{layer_pfx(ll)}Padding ({padding[ll]}) can only be used with a convolution '
                   'operation.')

        # Prohibit pad greater than or equal to kernel size
        if padding[ll][0] >= kernel_size[ll][0] or padding[ll][1] >= kernel_size[ll][1]:
            eprint(f'{layer_pfx(ll)}Pad size ({padding[ll]}) is greater than or equal to'
                   f' kernel size ({kernel_size[ll]}).')

        # Check for max dimensions
        if any(dim > tc.dev.MAX_ROW_COL for dim in input_dim[ll]):
            eprint(f'{layer_pfx(ll)}Input dimension {input_dim[ll]} exceeds system '
                   f'maximum of {tc.dev.MAX_ROW_COL}.')
        if any(dim > tc.dev.MAX_ROW_COL for dim in output_dim[ll]):
            eprint(f'{layer_pfx(ll)}Output dimension {output_dim[ll]} exceeds system '
                   f'maximum of {tc.dev.MAX_ROW_COL}.')
        if any(dim == 0 for dim in output_dim[ll]):
            eprint(f'{layer_pfx(ll)}Output dimension {output_dim[ll]} is zero.')

        assert input_channels[ll] > 0

        if activation[ll] is not None and operator[ll] == op.NONE:
            eprint(f'{layer_pfx(ll)}Activation {op.act_string(activation[ll])} cannot '
                   'be used in a passthrough layer.')

        # On MAX78002, if an average pool directly follows an element-wise operation, we need
        # to insert a small layer to reset the average pool logic
        if tc.dev.REQUIRE_PASSTHROUGH_AVGPOOL_AFTER_ELTWISE and ll > 0 \
           and operands[prev_sequence[ll]] > 1 \
           and pool_average[ll]:
            avgpool_reset_layer[ll] = True

        ll = next_sequence[ll]
        if ll == -1:
            break

    if args.riscv and not args.riscv_cache and args.embedded_code:
        eprint('Embedded code on RISC-V requires --riscv-cache.')

    # Modify global state based on locally calculated variables
    state.activation = activation
    state.auto_input_dim = auto_input_dim
    state.avgpool_reset_layer = avgpool_reset_layer
    state.bias = bias
    state.bias_group_map = bias_group_map
    state.big_data = big_data
    state.bypass = bypass
    state.calcx4 = calcx4
    state.conv_groups = conv_groups
    state.data = data
    state.dilation = dilation
    state.eltwise = eltwise
    state.final_layer = final_layer
    state.first_layer_used = min_layer
    state.flatten = flatten
    state.in_offset = input_offset
    state.in_sequences = in_sequences
    state.input_channel_skip = input_channel_skip
    state.input_channels = input_channels
    state.input_dim = input_dim
    state.input_offset = input_offset
    state.input_skip = input_skip
    state.kernel_size = kernel_size
    state.layers = layers
    state.next_sequence = next_sequence
    state.operands = operands
    state.operator = operator
    state.out_offset = output_offset
    state.output_channels = output_channels
    state.output_dim = output_dim
    state.output_layer = output_layer
    state.output_offset = output_offset
    state.output_padding = output_padding
    state.output_processor_map = output_processor_map
    state.output_shift = output_shift
    state.output_width = output_width
    state.padding = padding
    state.pool = pool
    state.pool_average = pool_average
    state.pool_dilation = pool_dilation
    state.pool_first = pool_first
    state.pool_stride = pool_stride
    state.pooled_dim = pooled_dim
    state.prev_sequence = prev_sequence
    state.processor_map = processor_map
    state.quantization = quantization
    state.read_ahead = readahead
    state.simulated_sequence = simulated_sequence
    state.snoop = cfg['snoop'] if 'snoop' in cfg else None
    state.snoop_sequence = snoop_sequence
    state.streaming = streaming
    state.stride = stride
    state.tcalc = tcalc
    state.unload_custom = params['unload_custom']
    state.weights = weights
    state.write_gap = write_gap

    # Implied states
    if state.riscv_debug:
        state.riscv = True
    if state.riscv_cache:
        state.riscv = True
        state.riscv_flash = True
    if state.riscv_flash or state.riscv_exclusive:
        state.riscv = True

    if state.fast_fifo_quad:
        state.fast_fifo = True
    if state.fast_fifo:
        state.fifo = True

    # Instantiate backend
    module = locate('izer.backend.' + tc.dev.backend)
    assert module is not None
    be = module.Backend()

    tn = be.create_net()
    if not args.embedded_code and args.autogen.lower() != 'none':
        rtlsim.append_regression(
            args.top_level,
            tn,
            args.queue_name,
            args.autogen,
            args.autogen_list,
        )

    # Restore stdout in case we're wrapped in cProfile
    sys.stdout = saved_stdout
