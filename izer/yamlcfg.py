###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
YAML Configuration Routines
"""
import os

import yaml
import yamllint
import yamllint.config
import yamllint.linter

from . import devices, names, op, state
from . import tornadocnn as tc
from .eprint import eprint, nprint, wprint

DEFAULT_2D_KERNEL = [3, 3]
DEFAULT_1D_KERNEL = [9, 1]
FC_KERNEL = [1, 1]


class UniqueKeyLoader(yaml.Loader):
    """
    Throw an error when encountering duplicate YAML keys.
    """
    def construct_mapping(self, node, deep=False):
        """Construct a key/value mapping"""
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None,
                f"Expected a mapping node, but found {node.id}",
                node.start_mark
            )

        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as exc:
                eprint(f'Found unacceptable key {exc} {key_node.start_mark} '
                       f'while constructing a mapping {node.start_mark}')

            # check for duplicate keys
            if key in mapping:
                eprint(f'Found duplicate key {key} '
                       f'while constructing a mapping {node.start_mark}')
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value

        return mapping


def parse(
        config_file,
        skip_layers=0,
):
    """
    Configure network parameters from the YAML configuration file `config_file`.
    `max_conv` can be set to force an early termination of the parser.
    The function returns both YAML dictionary, the length of the processor map,
    as well as a settings dictionary.
    """

    def error_exit(message, sequence):
        """
        Print error message `message` for layer sequence `sequence` and exit.
        """
        eprint(f'{message} (found in layer sequence {sequence} in YAML configuration).')

    print(f'Reading {config_file} to configure network...')

    # Run yamllint first
    if not os.path.exists(config_file):
        eprint(f'YAML configuration file {config_file} does not exist!')

    yaml_config = yamllint.config.YamlLintConfig('extends: relaxed')
    with open(config_file, mode='r', encoding='utf-8') as cfg_file:
        for p in yamllint.linter.run(cfg_file, yaml_config):
            eprint(f'{config_file} line {p.line}, col {p.column}: {p.desc}',
                   error=p.level == 'error')

    # Load configuration file
    with open(config_file, mode='r', encoding='utf-8') as cfg_file:
        cfg = yaml.load(cfg_file, Loader=UniqueKeyLoader)

    cfg_set = set(cfg) - set(['bias', 'dataset', 'layers', 'unload',
                              'output_map', 'arch', 'weights', 'snoop'])
    if bool(cfg_set):
        eprint(f'Configuration file {config_file} contains unknown key(s): {cfg_set}.')

    if 'layers' not in cfg or 'arch' not in cfg or 'dataset' not in cfg:
        eprint(f'Configuration file {config_file} does not contain '
               '`layers`, `arch`, or `dataset`.')

    # These are initialized with 'None'. Use this to see whether a layer was configured,
    # will be auto-initialized to previous layer's value or a default.
    processor_map = [None] * tc.dev.MAX_LAYERS
    output_map = [None] * tc.dev.MAX_LAYERS
    input_offset = [None] * tc.dev.MAX_LAYERS
    input_chan = [None] * tc.dev.MAX_LAYERS
    input_skip = [0] * tc.dev.MAX_LAYERS
    input_chan_skip = [0] * tc.dev.MAX_LAYERS
    input_dim = [None] * tc.dev.MAX_LAYERS
    output_chan = [None] * tc.dev.MAX_LAYERS
    # All other variables are initialized with the default values
    padding = [[1, 1]] * tc.dev.MAX_LAYERS
    output_padding = [[0, 0]] * tc.dev.MAX_LAYERS
    pool = [[1, 1]] * tc.dev.MAX_LAYERS
    pooling_enabled = [False] * tc.dev.MAX_LAYERS
    average = [0] * tc.dev.MAX_LAYERS
    pool_stride = [[None, None]] * tc.dev.MAX_LAYERS
    quantization = [None] * tc.dev.MAX_LAYERS
    bias_quantization = [8] * tc.dev.MAX_LAYERS
    output_shift = [None] * tc.dev.MAX_LAYERS
    output_offset = [0] * tc.dev.MAX_LAYERS
    activation = [None] * tc.dev.MAX_LAYERS
    big_data = [False] * tc.dev.MAX_LAYERS
    output_width = [8] * tc.dev.MAX_LAYERS
    operator = [op.CONV2D] * tc.dev.MAX_LAYERS
    dilation = [[1, 1]] * tc.dev.MAX_LAYERS
    kernel_size = [DEFAULT_2D_KERNEL] * tc.dev.MAX_LAYERS
    conv_groups = [1] * tc.dev.MAX_LAYERS
    stride = [[1, 1]] * tc.dev.MAX_LAYERS
    streaming = [False] * tc.dev.MAX_LAYERS
    flatten = [False] * tc.dev.MAX_LAYERS
    operands = [1] * tc.dev.MAX_LAYERS
    eltwise = [op.NONE] * tc.dev.MAX_LAYERS
    pool_first = [True] * tc.dev.MAX_LAYERS
    in_sequences = [None] * tc.dev.MAX_LAYERS
    next_sequence = [None] * tc.dev.MAX_LAYERS
    simulated_sequence = [None] * tc.dev.MAX_LAYERS
    snoop_sequence = [None] * tc.dev.MAX_LAYERS
    write_gap = [0] * tc.dev.MAX_LAYERS
    bypass = [False] * tc.dev.MAX_LAYERS
    bias_group_map = [None] * tc.dev.MAX_LAYERS
    calcx4 = [False] * tc.dev.MAX_LAYERS
    readahead = [False] * tc.dev.MAX_LAYERS
    pool_dilation = [[1, 1]] * tc.dev.MAX_LAYERS
    tcalc = [None] * tc.dev.MAX_LAYERS
    output_layer = [False] * tc.dev.MAX_LAYERS
    unload_custom = []
    layer_name = [None] * tc.dev.MAX_LAYERS
    weight_source = [None] * tc.dev.MAX_LAYERS

    sequence = 0
    skip = skip_layers
    for ll in cfg['layers']:
        if skip > 0:
            skip -= 1
            continue

        cfg_set = set(ll) - set(['max_pool', 'avg_pool', 'convolution', 'conv_groups', 'dilation',
                                 'groups', 'in_channels', 'in_dim', 'in_sequences', 'in_skip',
                                 'in_channel_skip', 'in_offset', 'kernel_size', 'pool_stride',
                                 'out_channels', 'out_offset', 'activate', 'activation',
                                 'data_format', 'eltwise', 'flatten', 'op', 'operands',
                                 'operation', 'operator', 'output_processors', 'output_width',
                                 'output_shift', 'pool_first', 'processors', 'pad', 'quantization',
                                 'next_sequence', 'snoop_sequence', 'simulated_sequence',
                                 'sequence', 'streaming', 'stride', 'write_gap', 'bypass',
                                 'bias_group', 'bias_quadrant', 'calcx4', 'readahead', 'name',
                                 'pool_dilation', 'output_pad', 'tcalc', 'read_gap', 'output',
                                 'weight_source'])
        if bool(cfg_set):
            eprint(f'Configuration file {config_file} contains unknown key(s) for `layers`: '
                   f'{cfg_set}.')

        if 'sequence' in ll:
            sequence = ll['sequence'] - skip_layers  # Override sequence information

        if sequence >= tc.dev.MAX_LAYERS:
            error_exit(f'This device supports up to {tc.dev.MAX_LAYERS} layers', sequence)

        if processor_map[sequence]:
            error_exit('Layer was already specified', sequence)

        if tc.dev.device != devices.CMSISNN:
            pmap = ll['processors'] if 'processors' in ll else None
            if isinstance(pmap, str):
                try:
                    pmap = int(pmap.replace('.', '').replace('_', ''), 16)
                except ValueError:
                    pass
            if pmap is None:
                error_exit('`processors` must not be zero or missing', sequence)
            if not isinstance(pmap, int) or pmap < 1 or pmap >= 2**tc.dev.MAX_PROC:
                error_exit('`processors` must be an int from 1 to '
                           f'2**{tc.dev.MAX_PROC}-1', sequence)
            processor_map[sequence] = pmap

            if 'output_processors' in ll:
                pmap = ll['output_processors']
                if isinstance(pmap, str):
                    try:
                        pmap = int(pmap.replace('.', '').replace('_', ''), 16)
                    except ValueError:
                        pass
                if not isinstance(pmap, int) or pmap < 1 or pmap >= 2**tc.dev.MAX_PROC:
                    error_exit('`output_processors` must be an int from 1 to '
                               f'2**{tc.dev.MAX_PROC}-1', sequence)
                output_map[sequence] = pmap
        else:
            processor_map[sequence] = 1

        if 'max_pool' in ll:
            val = ll['max_pool']
            if not isinstance(val, list):
                pool[sequence] = [val, val]
            else:
                pool[sequence] = val
            pooling_enabled[sequence] = True
        elif 'avg_pool' in ll:
            val = ll['avg_pool']
            if not isinstance(val, list):
                pool[sequence] = [val, val]
            else:
                pool[sequence] = val
            pooling_enabled[sequence] = True
            average[sequence] = 1

        if 'pool_stride' in ll:
            val = ll['pool_stride']
            if not isinstance(val, list):
                pool_stride[sequence] = [val, val]
            else:
                pool_stride[sequence] = val

        if 'quantization' in ll:
            val = ll['quantization']
            if isinstance(val, str):
                val = val.lower()
            if val not in [1, 2, 4, 8, 'bin', 'binary']:
                error_exit('`quantization` must be 1, 2, 4, 8 or bin/binary', sequence)
            if val in ['bin', 'binary']:
                val = -1
            quantization[sequence] = val

        if 'output_shift' in ll:
            val = ll['output_shift']
            output_shift[sequence] = val
            # The implicit shift for quantization is added later

        if 'in_channels' in ll:
            input_chan[sequence] = ll['in_channels']
        if 'in_dim' in ll:
            val = ll['in_dim']
            if isinstance(val, int):
                val = [val, 1]
            else:
                if not isinstance(val, list) or len(val) > 2:
                    error_exit('`in_dim` must be an integer or list not exceeding two dimensions',
                               sequence)
            input_dim[sequence] = val
        if 'in_skip' in ll and 'read_gap' in ll:
            error_exit('Duplicate key for `in_skip`/`read_gap`', sequence)
        if 'in_skip' in ll or 'read_gap' in ll:
            key = 'in_skip' if 'in_skip' in ll else 'read_gap'
            input_skip[sequence] = ll[key]
        if 'in_channel_skip' in ll:
            input_chan_skip[sequence] = ll['in_channel_skip']
        if 'in_offset' in ll:
            input_offset[sequence] = ll['in_offset']
        if 'out_channels' in ll:
            output_chan[sequence] = ll['out_channels']
        if 'out_offset' in ll:
            output_offset[sequence] = ll['out_offset']
        else:
            wprint('Defaulting to `out_offset = 0` for '
                   f'layer sequence {sequence} in YAML configuration.')

        if 'activate' in ll and 'activation' in ll:
            error_exit('Duplicate key for `activation`/`activate`', sequence)
        if 'activate' in ll or 'activation' in ll:
            key = 'activate' if 'activate' in ll else 'activation'
            if ll[key].lower() == 'relu':
                activation[sequence] = op.ACT_RELU
            elif ll[key].lower() == 'abs':
                activation[sequence] = op.ACT_ABS
            elif ll[key].lower() == 'none':
                activation[sequence] = None
            else:
                error_exit(f'Unknown value "{ll[key]}" for `{key}`', sequence)
            if state.ignore_activation:
                activation[sequence] = None

        if 'convolution' in ll and 'operation' in ll \
           or 'convolution' in ll and 'operator' in ll \
           or 'convolution' in ll and 'op' in ll \
           or 'operation' in ll and 'operator' in ll \
           or 'operation' in ll and 'op' in ll \
           or 'operator' in ll and 'op' in ll:
            error_exit('Duplicate key for `convolution`/`operation`/`operator`/`op`', sequence)
        if 'convolution' in ll or 'operation' in ll or 'op' in ll or 'operator' in ll:
            key = 'convolution' if 'convolution' in ll else \
                  'operation' if 'operation' in ll else \
                  'operator' if 'operator' in ll else \
                  'op'
            conv = ll[key].lower()
            if conv == 'conv1d':
                operator[sequence] = op.CONV1D
            elif conv == 'conv2d':
                operator[sequence] = op.CONV2D
            elif conv == 'convtranspose2d':
                operator[sequence] = op.CONVTRANSPOSE2D
            elif conv in ['none', 'passthrough']:
                operator[sequence] = op.NONE
                padding[sequence] = [0, 0]
            elif conv == 'add':
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_ADD
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv in ['or', 'bitwiseor', 'bitwise_or']:
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_OR
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv == 'sub':
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_SUB
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv in ['xor', 'bitwisexor', 'bitwise_xor']:
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_XOR
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv in ['linear', 'fc', 'mlp']:
                # Emulate using Conv2D with 1x1 kernels and 1x1 data
                operator[sequence] = op.LINEAR
                kernel_size[sequence] = FC_KERNEL
                padding[sequence] = [0, 0]
            else:
                error_exit(f'Unknown value "{ll[key]}" for `{key}`', sequence)
        else:
            wprint('Defaulting to `op: Conv2d` for '
                   f'layer sequence {sequence} in YAML configuration.')

        if operator[sequence] in [op.CONV1D, op.CONV2D, op.CONVTRANSPOSE2D, op.LINEAR] \
           and not ('activate' in ll or 'activation' in ll):
            nprint(f'Defaulting to "no activation" for {conv} in '
                   f'layer sequence {sequence} in YAML configuration.')

        if 'pad' in ll:
            val = ll['pad']
            if val < 0:
                error_exit(f'Unsupported value {val} for `pad`', sequence)
            padding[sequence] = [val, val]

        if 'dilation' in ll:
            val = ll['dilation']
            if not isinstance(val, list):
                dilation[sequence] = [val, val]
            else:
                dilation[sequence] = val
            if operator[sequence] == op.NONE:
                error_exit('`dilation` requires a convolution operator', sequence)

        if 'eltwise' in ll:
            conv = ll['eltwise'].lower()
            if conv == 'add':
                eltwise[sequence] = op.ELTWISE_ADD
                operands[sequence] = 2
            elif conv == 'or':
                eltwise[sequence] = op.ELTWISE_OR
                operands[sequence] = 2
            elif conv == 'sub':
                eltwise[sequence] = op.ELTWISE_SUB
                operands[sequence] = 2
            elif conv == 'xor':
                eltwise[sequence] = op.ELTWISE_XOR
                operands[sequence] = 2
            else:
                error_exit(f'Unknown value "{ll["eltwise"]}" for `eltwise`', sequence)

        if 'pool_first' in ll:
            val = ll['pool_first']
            try:
                pool_first[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `pool_first`', sequence)

        if 'operands' in ll:
            if not op.eltwise(eltwise[sequence]):
                error_exit('`operands` can only be used with element-wise operations', sequence)
            val = ll['operands']
            if val < 2 or val > 16:
                error_exit('`operands` has to be 2..16', sequence)
            operands[sequence] = val

        if 'data_format' in ll:
            val = ll['data_format'].lower()
            if val in ['chw', 'big']:
                big_data[sequence] = True
            elif val in ['hwc', 'little']:
                pass
            else:
                error_exit('Unknown value for `data_format`', sequence)

        if 'output_width' in ll:
            val = ll['output_width']
            if val not in [8, 32]:
                error_exit('`output_width` must be 8 or 32', sequence)
            output_width[sequence] = val

        if 'kernel_size' in ll:
            if kernel_size[sequence] != DEFAULT_2D_KERNEL:
                error_exit('Cannot configure `kernel_size` for fully connected layers', sequence)

            val = str(ll['kernel_size']).lower()
            if operator[sequence] == op.CONV2D:
                if val not in ['1x1', '3x3']:
                    error_exit(f'Unsupported value `{val}` for `kernel_size`', sequence)
                kernel_size[sequence] = [int(val[0]), int(val[2])]
            elif operator[sequence] == op.CONVTRANSPOSE2D:
                if val not in ['3x3']:
                    error_exit(f'Unsupported value `{val}` for `kernel_size`', sequence)
                kernel_size[sequence] = [int(val[0]), int(val[2])]
            else:
                try:
                    val = int(val)
                except ValueError:
                    error_exit(f'Unsupported value `{val}` for `kernel_size`', sequence)
                if val < 1 or val > 9:
                    error_exit(f'Unsupported value `{val}` for `kernel_size`', sequence)
                kernel_size[sequence] = [val, 1]
        elif operator[sequence] == op.CONV1D:  # Set default for 1D convolution
            kernel_size[sequence] = DEFAULT_1D_KERNEL

        if 'stride' in ll:
            val = ll['stride']
            if isinstance(val, list):
                if val[0] != val[1]:
                    error_exit('`stride` values must be equal in both dimensions', sequence)
                val = val[0]
            if pooling_enabled[sequence]:
                # Must use the default stride when pooling, otherwise stride can be set
                if val != 1:
                    error_exit('Cannot set `stride` to non-default value when pooling', sequence)
            else:
                if operator[sequence] == op.CONVTRANSPOSE2D and val != 2:
                    error_exit('Cannot set `stride` to non-default value for ConvTranspose2D',
                               sequence)
                # Stride can be set
                stride[sequence] = [val, val]

        if 'streaming' in ll:
            val = ll['streaming']
            try:
                streaming[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `streaming`', sequence)

        if 'flatten' in ll:
            val = ll['flatten']
            try:
                flatten[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `flatten`', sequence)

        if 'in_sequences' in ll:
            val = ll['in_sequences']
            if not isinstance(val, list):
                val = [val]
            in_sequences[sequence] = [
                x - skip_layers if not isinstance(x, str) else x for x in val
            ]

        if 'next_sequence' in ll:
            val = ll['next_sequence']
            if isinstance(val, str):
                val = val.lower()
                if val == 'stop':
                    val = -1
                next_sequence[sequence] = val
            else:
                next_sequence[sequence] = val - skip_layers

        if 'simulated_sequence' in ll:
            val = ll['simulated_sequence']
            if isinstance(val, str):
                val = val.lower()
                if val == 'stop':
                    val = -1
                simulated_sequence[sequence] = val
            else:
                simulated_sequence[sequence] = val - skip_layers

        if 'snoop_sequence' in ll:
            snoop_sequence[sequence] = ll['snoop_sequence'] - skip_layers

        if 'conv_groups' in ll and 'groups' in ll:
            error_exit('Duplicate key for `conv_groups`/`groups`', sequence)
        if 'conv_groups' in ll or 'groups' in ll:
            key = 'conv_groups' if 'conv_groups' in ll else 'groups'
            conv_groups[sequence] = ll[key]

        if 'write_gap' in ll:
            write_gap[sequence] = ll['write_gap']

        if 'bypass' in ll:
            val = ll['bypass']
            try:
                bypass[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `bypass`', sequence)

        if 'bias_group' in ll and 'bias_quadrant' in ll:
            error_exit('Duplicate key for `bias_quadrant`/`bias_group`', sequence)
        if 'bias_group' in ll or 'bias_quadrant' in ll:
            key = 'bias_quadrant' if 'bias_quadrant' in ll else 'bias_group'
            val = ll[key]
            if isinstance(val, int):
                bias_group_map[sequence] = [val]
            else:
                bias_group_map[sequence] = val

        if 'calcx4' in ll:
            val = ll['calcx4']
            try:
                calcx4[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `calcx4`', sequence)

        if 'tcalc' in ll:
            val = ll['tcalc']
            try:
                tcalc[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `tcalc`', sequence)

        if 'readahead' in ll:
            val = ll['readahead']
            try:
                readahead[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `readahead`', sequence)

        if 'pool_dilation' in ll:
            val = ll['pool_dilation']
            if not isinstance(val, list):
                pool_dilation[sequence] = [val, val]
            else:
                pool_dilation[sequence] = val
            if not pooling_enabled[sequence]:
                error_exit('`pool_dilation` requires pooling', sequence)

        if 'output_pad' in ll:
            val = ll['output_pad']
            if val < 0:
                error_exit(f'Unsupported value {val} for `output_pad`', sequence)
            output_padding[sequence] = [val, val]
        elif operator[sequence] == op.CONVTRANSPOSE2D:
            output_padding[sequence] = [1, 1]

        if 'output' in ll:
            val = ll['output']
            try:
                output_layer[sequence] = bool(val)
            except ValueError:
                error_exit(f'Unsupported value `{val}` for `output`', sequence)

        if 'name' in ll:
            val = ll['name']
            if names.find_layer(layer_name, sequence, val.lower(), 'name', False) is not None:
                error_exit(f'Duplicate layer name {val} for `name`', sequence)
            if val.lower() in ['stop', 'input']:
                error_exit(f'Using reserved name {val} for `name`', sequence)
            layer_name[sequence] = val

        if 'weight_source' in ll:
            val = ll['weight_source']
            if isinstance(val, str):
                val = val.lower()
                weight_source[sequence] = val
            else:
                weight_source[sequence] = val - skip_layers

        # Fix up values for 1D convolution or no convolution
        if operator[sequence] == op.CONV1D:
            padding[sequence][1] = 0
            pool[sequence][1] = 1
            pool_stride[sequence][1] = 1
            stride[sequence][1] = 1
            dilation[sequence][1] = 1
            pool_dilation[sequence][1] = 1
        elif operator[sequence] == op.NONE:
            kernel_size[sequence] = [1, 1]
        elif operator[sequence] == op.CONVTRANSPOSE2D:
            stride[sequence] = [2, 2]

        sequence += 1

    # Sequence specification may have holes. Contract to the used layers.
    for ll in range(tc.dev.MAX_LAYERS-1, -1, -1):
        if processor_map[ll] is None:
            del processor_map[ll]
            del padding[ll]
            del pool[ll]
            del pooling_enabled[ll]
            del pool_stride[ll]
            del input_chan[ll]
            del input_chan_skip[ll]
            del input_skip[ll]
            del input_dim[ll]
            del input_offset[ll]
            del output_chan[ll]
            del output_offset[ll]
            del average[ll]
            del activation[ll]
            del big_data[ll]
            del quantization[ll]
            del bias_quantization[ll]
            del output_shift[ll]
            del output_map[ll]
            del output_width[ll]
            del operator[ll]
            del dilation[ll]
            del kernel_size[ll]
            del stride[ll]
            del streaming[ll]
            del flatten[ll]
            del operands[ll]
            del eltwise[ll]
            del pool_first[ll]
            del in_sequences[ll]
            del next_sequence[ll]
            del simulated_sequence[ll]
            del snoop_sequence[ll]
            del conv_groups[ll]
            del write_gap[ll]
            del bypass[ll]
            del bias_group_map[ll]
            del calcx4[ll]
            del readahead[ll]
            del pool_dilation[ll]
            del output_padding[ll]
            del tcalc[ll]
            del output_layer[ll]
            del weight_source[ll]

    for ll, _ in enumerate(operator):
        # Convert string layer names to sequences
        if isinstance(next_sequence[ll], str):
            next_sequence[ll] = names.find_layer(layer_name, ll, next_sequence[ll],
                                                 'next_sequence')
        if isinstance(simulated_sequence[ll], str):
            simulated_sequence[ll] = names.find_layer(layer_name, ll, simulated_sequence[ll],
                                                      'simulated_sequence')
        if isinstance(weight_source[ll], str):
            weight_source[ll] = names.find_layer(layer_name, ll, weight_source[ll],
                                                 'weight_source')
        if in_sequences[ll] is not None:
            new_in_sequences = []
            for e in in_sequences[ll]:
                if isinstance(e, str):
                    new_in_sequences.append(names.find_layer(layer_name, ll, e, 'in_sequences'))
                else:
                    new_in_sequences.append(e)
            in_sequences[ll] = new_in_sequences

        # Warn when using default pool stride of 1, 1
        if pool_stride[ll][0] is None:
            if pooling_enabled[ll]:
                wprint(f'Using default pool stride of 1 in layer {ll}.')
            pool_stride[ll] = [1, 1]

        # Check that pooling isn't set for ConvTranspose2d:
        if not pool_first[ll] and (operands[ll] == 1 or pool[ll][0] == 1 and pool[ll][1] == 1):
            error_exit('`pool_first: False` requires both pooling and element-wise operations', ll)

    if 'unload' in cfg:
        for ll in cfg['unload']:
            cfg_set = set(ll)

            if bool(cfg_set - set(['processors', 'dim', 'offset', 'width',
                                   'write_gap', 'channels'])):
                eprint(f'Configuration file {config_file} contains unknown key(s) for `unload`.')

            if 'processors' not in cfg_set or 'dim' not in cfg_set or 'channels' not in cfg_set \
               or 'offset' not in cfg_set:
                eprint(f'`unload` sequence in configuration file {config_file} does not contain '
                       '`processors`, `channels`, `dim`, or `offset`.')

            unload_proc = ll['processors']
            if isinstance(unload_proc, str):
                try:
                    unload_proc = int(unload_proc.replace('.', '').replace('_', ''), 16)
                except ValueError:
                    pass
            val = ll['dim']
            unload_dim = val if isinstance(val, list) else [val, 1]
            unload_channels = ll['channels']
            unload_offset = ll['offset']
            unload_width = ll['width'] if 'with' in ll else 8
            unload_write_gap = ll['write_gap'] if 'write_gap' in ll else 0

            unload_custom.append({
                'proc': unload_proc,
                'dim': (unload_channels, unload_dim[0], unload_dim[1]),
                'offset': unload_offset,
                'width': unload_width,
                'write_gap': unload_write_gap,
            })

    settings = {}
    settings['activation'] = activation
    settings['average'] = average
    settings['bias_group_map'] = bias_group_map
    settings['bias_quantization'] = bias_quantization
    settings['big_data'] = big_data
    settings['bypass'] = bypass
    settings['calcx4'] = calcx4
    settings['conv_groups'] = conv_groups
    settings['dilation'] = dilation
    settings['eltwise'] = eltwise
    settings['flatten'] = flatten
    settings['in_sequences'] = in_sequences
    settings['input_chan_skip'] = input_chan_skip
    settings['input_chan'] = input_chan
    settings['input_dim'] = input_dim
    settings['input_offset'] = input_offset
    settings['input_skip'] = input_skip
    settings['kernel_size'] = kernel_size
    settings['layer_name'] = layer_name
    settings['next_sequence'] = next_sequence
    settings['operands'] = operands
    settings['operator'] = operator
    settings['output_chan'] = output_chan
    settings['output_layer'] = output_layer
    settings['output_offset'] = output_offset
    settings['output_padding'] = output_padding
    settings['output_processor_map'] = output_map
    settings['output_shift'] = output_shift
    settings['output_width'] = output_width
    settings['padding'] = padding
    settings['pool_dilation'] = pool_dilation
    settings['pool_first'] = pool_first
    settings['pool_stride'] = pool_stride
    settings['pool'] = pool
    settings['pooling_enabled'] = pooling_enabled
    settings['processor_map'] = processor_map
    settings['quantization'] = quantization
    settings['readahead'] = readahead
    settings['simulated_sequence'] = simulated_sequence
    settings['snoop_sequence'] = snoop_sequence
    settings['streaming'] = streaming
    settings['stride'] = stride
    settings['tcalc'] = tcalc
    settings['unload_custom'] = unload_custom if len(unload_custom) > 0 else None
    settings['write_gap'] = write_gap
    settings['weight_source'] = weight_source

    return cfg, len(processor_map), settings
