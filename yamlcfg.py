###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
YAML Configuration Routines
"""
import yaml

import devices
import op
import tornadocnn as tc
from eprint import eprint, wprint

DEFAULT_2D_KERNEL = [3, 3]
DEFAULT_1D_KERNEL = [9, 1]
FC_KERNEL = [1, 1]


class UniqueKeyLoader(yaml.Loader):
    """
    Throw an error when encountering duplicate YAML keys.
    """
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None,
                "Expected a mapping node, but found %s" % node.id,
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
                       f'while constructing a mapping{node.start_mark}')
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value

        return mapping


def parse(config_file, max_conv=None, device=84):  # pylint: disable=unused-argument
    """
    Configure network parameters from the YAML configuration file `config_file`.
    `max_conv` can be set to force an early termination of the parser.
    `device` is `84`, `85`, etc.
    The function returns both YAML dictionary, the length of the processor map,
    as well as a settings dictionary.
    """

    def error_exit(message, sequence):
        """
        Print error message `message` for layer sequence `sequence` and exit.
        """
        eprint(f'{message} (found in layer sequence {sequence} in YAML configuration).')

    # Load configuration file
    with open(config_file) as cfg_file:
        print(f'Reading {config_file} to configure network...')
        cfg = yaml.load(cfg_file, Loader=UniqueKeyLoader)

    if bool(set(cfg) - set(['bias', 'dataset', 'layers', 'output_map', 'arch', 'weights'])):
        eprint(f'Configuration file {config_file} contains unknown key(s).')

    if 'layers' not in cfg or 'arch' not in cfg or 'dataset' not in cfg:
        eprint(f'Configuration file {config_file} does not contain '
               f'`layers`, `arch`, or `dataset`.')

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
    # We don't support changing the following (yet), but leave as parameters:
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
    write_gap = [0] * tc.dev.MAX_LAYERS

    sequence = 0
    for ll in cfg['layers']:
        if bool(set(ll) - set(['max_pool', 'avg_pool', 'convolution', 'conv_groups',
                               'groups', 'in_channels', 'in_dim', 'in_sequences', 'in_skip',
                               'in_channel_skip', 'in_offset', 'kernel_size', 'pool_stride',
                               'out_channels', 'out_offset', 'activate', 'activation',
                               'data_format', 'eltwise', 'flatten', 'op', 'operands', 'operation',
                               'operator', 'output_processors', 'output_width', 'output_shift',
                               'pool_first', 'processors', 'pad', 'quantization',
                               'sequence', 'streaming', 'stride', 'write_gap'])):
            eprint(f'Configuration file {config_file} contains unknown key(s) for `layers`.')

        if 'sequence' in ll:
            sequence = ll['sequence']  # Override sequence information

        if sequence >= tc.dev.MAX_LAYERS:
            error_exit(f'This device supports up to {tc.dev.MAX_LAYERS} layers', sequence)

        if processor_map[sequence]:
            error_exit('Layer was already specified', sequence)

        if device != devices.CMSISNN:
            if 'processors' in ll:
                processor_map[sequence] = ll['processors']
            if not processor_map[sequence]:
                error_exit('`processors` must not be zero or missing', sequence)
            if not isinstance(processor_map[sequence], int) \
               or processor_map[sequence] >= 2**tc.dev.MAX_PROC:
                error_exit(f'`processors` must be an int from 0 to 2**{tc.dev.MAX_PROC}-1',
                           sequence)

            if 'output_processors' in ll:
                output_map[sequence] = ll['output_processors']
                if not output_map[sequence]:
                    error_exit('output_processors` cannot be zero', sequence)
                if not isinstance(output_map[sequence], int) \
                   or output_map[sequence] >= 2**tc.dev.MAX_PROC:
                    error_exit('`output_processors` must be an int from 0 to '
                               f'2**{tc.dev.MAX_PROC}-1', sequence)
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
            if val not in [1, 2, 4, 8]:
                error_exit('`quantization` must be 1, 2, 4, or 8', sequence)
            quantization[sequence] = val

        if 'output_shift' in ll:
            val = ll['output_shift']
            output_shift[sequence] = val
            # The implicit shift for quantization is added later

        if 'in_channels' in ll:
            input_chan[sequence] = ll['in_channels']
        if 'in_dim' in ll:
            if isinstance(ll['in_dim'], list) and len(ll['in_dim']) > 2:
                error_exit('`in_dim` must not exceed two dimensions', sequence)
            input_dim[sequence] = ll['in_dim']
        if 'in_skip' in ll:
            input_skip[sequence] = ll['in_skip']
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
            elif conv == 'or':
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_OR
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv == 'sub':
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_SUB
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv == 'xor':
                operator[sequence] = op.NONE
                eltwise[sequence] = op.ELTWISE_XOR
                operands[sequence] = 2
                padding[sequence] = [0, 0]
            elif conv in ['linear', 'fc', 'mlp']:
                # Emulate using Conv2D with 1x1 kernels and 1x1 data
                operator[sequence] = op.CONV2D
                kernel_size[sequence] = FC_KERNEL
                padding[sequence] = [0, 0]
            else:
                error_exit(f'Unknown value "{ll[key]}" for `{key}`', sequence)
        else:
            wprint('Defaulting to `op: Conv2d` for '
                   f'layer sequence {sequence} in YAML configuration.')

        if 'pad' in ll:
            val = ll['pad']
            if val < 0:
                error_exit(f'Unsupported value {val} for `pad`', sequence)
            padding[sequence] = [val, val]

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
            if sequence:
                error_exit('`data_format` can only be configured for the first layer', sequence)

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
                if device == 84 and val not in ['3x3'] \
                        or device != 84 and val not in ['1x1', '3x3']:
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
                if device == 84 and val != 9 or val < 1 or val > 9:
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
                if operator[sequence] == op.CONV2D and val != 1 \
                   or (device == 84 and val != 3 or val != 1):
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
            if isinstance(ll['in_sequences'], list):
                if any([(i >= sequence) for i in ll['in_sequences']]):
                    error_exit('`in_sequences` cannot be greater than layer sequence', sequence)
            elif ll['in_sequences'] >= sequence:
                error_exit('`in_sequences` cannot be greater than layer sequence', sequence)
            in_sequences[sequence] = ll['in_sequences']

        if 'conv_groups' in ll or 'groups' in ll:
            key = 'conv_groups' if 'conv_groups' in ll else 'groups'
            conv_groups[sequence] = ll[key]

        if 'write_gap' in ll:
            write_gap[sequence] = ll['write_gap']

        # Fix up values for 1D convolution or no convolution
        if operator[sequence] == op.CONV1D:
            padding[sequence][1] = 0
            pool[sequence][1] = 1
            pool_stride[sequence][1] = 1
            stride[sequence][1] = 1
        elif operator[sequence] == op.NONE:
            kernel_size[sequence] = [1, 1]
        elif operator[sequence] == op.CONVTRANSPOSE2D:
            stride[sequence] = [2, 2]

        # Check for early exit
        if max_conv is not None:
            if max_conv == 0:
                if output_map[sequence] is None and (len(cfg['layers']) > sequence + 1):
                    if 'processors' in cfg['layers'][sequence+1]:
                        output_map[sequence] = cfg['layers'][sequence+1]['processors']
                break
            max_conv -= 1

        sequence += 1

    # Sequence specification may have holes. Contract to the used layers.
    for ll in range(tc.dev.MAX_LAYERS-1, -1, -1):
        if processor_map[ll] is None:
            del processor_map[ll]
            del padding[ll]
            del pool[ll]
            del pool_stride[ll]
            del input_chan[ll]
            del input_skip[ll]
            del input_chan_skip[ll]
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
            del pooling_enabled[ll]
            del streaming[ll]
            del flatten[ll]
            del operands[ll]
            del eltwise[ll]
            del conv_groups[ll]
            del write_gap[ll]

    # Check all but last layer
    for ll in range(len(output_map) - 1):
        if output_width[ll] != 8:
            error_exit('`output_width` is not 8 for intermediate layer', ll)
        # Fix up default output maps
        if output_map[ll] is None:
            output_map[ll] = processor_map[ll+1]

    # Check all but first layer
    for ll in range(1, len(input_offset)):
        # Fix up default input maps
        if input_offset[ll] is None:
            input_offset[ll] = output_offset[ll-1]
        # Check we don't turn on streaming too late
        if streaming[ll] and not streaming[ll-1]:
            error_exit('Enable streaming from the first layer on', ll)
    # Check first layer
    if input_offset[0] is None:
        input_offset[0] = 0
    if device != devices.CMSISNN:
        # Check last layer
        if output_map[-1] is None and 'output_map' in cfg:
            output_map[-1] = cfg['output_map']
        if output_width[-1] != 8 and activation[-1] is not None:
            error_exit('`output_width` must be 8 when activation is used', len(activation))

    # Check all layers
    for ll, e in enumerate(operator):
        # Warn when using default pool stride of 1, 1
        if pool_stride[ll][0] is None:
            if pooling_enabled[ll]:
                wprint(f'Using default pool stride of 1 in layer {ll}.')
            pool_stride[ll] = [1, 1]

        # Check that pass-through does not use activation
        if e == op.NONE:
            if activation[ll] is not None:
                error_exit('Pass-through layers must not use activation', ll)
            if padding[ll][0] != 0 or padding[ll][1] != 0:
                error_exit('Padding must be zero for passthrough layers', ll)
            if output_shift[ll] != 0 and output_shift[ll] is not None:
                error_exit('`output_shift` must be zero for passthrough layers', ll)
        # Check that pooling isn't set for ConvTranspose2d:
        elif e == op.CONVTRANSPOSE2D:
            if pooling_enabled[ll]:
                error_exit('ConvTranspose2d cannot be used with pooling', ll)
        # Check that element-wise does not use Conv1d
        if e == op.CONV1D and operands[ll] > 1:
            error_exit('Element-wise operations cannot be combined with Conv1d', ll)
        if not pool_first[ll] and (operands[ll] == 1 or pool[ll][0] == 1 and pool[ll][1] == 1):
            error_exit('`pool_first: False` requires both pooling and element-wise operations', ll)

    if device == 84:
        # Fix up defaults for Conv1D:
        for ll, e in enumerate(operator):
            if e == op.CONV1D:
                kernel_size[ll] = [9, 1]

    settings = {}
    settings['padding'] = padding
    settings['pool'] = pool
    settings['pooling_enabled'] = pooling_enabled
    settings['pool_stride'] = pool_stride
    settings['input_chan'] = input_chan
    settings['input_chan_skip'] = input_chan_skip
    settings['input_skip'] = input_skip
    settings['input_dim'] = input_dim
    settings['input_offset'] = input_offset
    settings['output_chan'] = output_chan
    settings['output_offset'] = output_offset
    settings['processor_map'] = processor_map
    settings['average'] = average
    settings['activation'] = activation
    settings['big_data'] = big_data
    settings['quantization'] = quantization
    settings['bias_quantization'] = bias_quantization
    settings['output_shift'] = output_shift
    settings['output_processor_map'] = output_map
    settings['output_width'] = output_width
    settings['operator'] = operator
    settings['dilation'] = dilation
    settings['kernel_size'] = kernel_size
    settings['stride'] = stride
    settings['streaming'] = streaming
    settings['flatten'] = flatten
    settings['operands'] = operands
    settings['eltwise'] = eltwise
    settings['pool_first'] = pool_first
    settings['in_sequences'] = in_sequences
    settings['conv_groups'] = conv_groups
    settings['write_gap'] = write_gap

    return cfg, len(processor_map), settings
