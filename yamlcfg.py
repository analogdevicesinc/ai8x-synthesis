###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
YAML Configuration Routines
"""
import sys
import yaml
import tornadocnn as tc


SUPPORTED_DATASETS = ['mnist', 'fashionmnist', 'cifar-10', 'speechcom', 'test_conv1d',
                      'test_conv1x1', 'test_nonsquare']
DEFAULT_2D_KERNEL = [3, 3]
DEFAULT_1D_KERNEL = [9, 1]


def parse(config_file, device=84):  # pylint: disable=unused-argument
    """
    Configure network parameters from the YAML configuration file `config_file`.
    The function returns both the YAML dictionary as well as a settings dictionary.
    `device` is `84`, `85`, etc.
    """

    def error_exit(message, sequence):
        """
        Print error message `message` for layer sequence `sequence` and exit.
        """
        print(f'{message} (found in layer sequence {sequence} in YAML configuration).')
        sys.exit(1)

    # Load configuration file
    with open(config_file) as cfg_file:
        print(f'Reading {config_file} to configure network...')
        cfg = yaml.load(cfg_file, Loader=yaml.SafeLoader)

    if bool(set(cfg) - set(['dataset', 'layers', 'output_map', 'arch'])):
        print(f'Configuration file {config_file} contains unknown key(s).')
        sys.exit(1)

    if 'layers' not in cfg or 'arch' not in cfg or 'dataset' not in cfg:
        print(f'Configuration file {config_file} does not contain '
              f'`layers`, `arch`, or `dataset`.')
        sys.exit(1)

    if bool(set([cfg['dataset'].lower()]) - set(SUPPORTED_DATASETS)):
        print(f'Configuration file {config_file} contains unknown `dataset`.')
        sys.exit(1)

    # These are initializaed with 'None'. Use this to see whether a layer was configured,
    # will be auto-initialized to previous layer's value or a default.
    processor_map = [None] * tc.dev.MAX_LAYERS
    output_map = [None] * tc.dev.MAX_LAYERS
    input_offset = [None] * tc.dev.MAX_LAYERS
    input_dim = [None] * tc.dev.MAX_LAYERS
    # All other variables are initialized with the default values
    padding = [[1, 1]] * tc.dev.MAX_LAYERS
    pool = [[0, 0]] * tc.dev.MAX_LAYERS
    pooling_enabled = [False] * tc.dev.MAX_LAYERS
    average = [0] * tc.dev.MAX_LAYERS
    pool_stride = [[1, 1]] * tc.dev.MAX_LAYERS
    quantization = [8] * tc.dev.MAX_LAYERS
    output_offset = [0] * tc.dev.MAX_LAYERS
    relu = [0] * tc.dev.MAX_LAYERS
    big_data = [False] * tc.dev.MAX_LAYERS
    output_width = [8] * tc.dev.MAX_LAYERS
    convolution = [2] * tc.dev.MAX_LAYERS
    # We don't support changing the following (yet), but leave as parameters:
    dilation = [[1, 1]] * tc.dev.MAX_LAYERS
    kernel_size = [DEFAULT_2D_KERNEL] * tc.dev.MAX_LAYERS
    stride = [[1, 1]] * tc.dev.MAX_LAYERS

    sequence = 0
    for ll in cfg['layers']:
        if bool(set(ll) - set(['max_pool', 'avg_pool', 'convolution', 'in_dim',
                               'in_offset', 'kernel_size', 'pool_stride', 'out_offset',
                               'activate', 'data_format', 'output_processors', 'output_width',
                               'processors', 'pad', 'quantization', 'sequence', 'stride'])):
            print(f'Configuration file {config_file} contains unknown key(s) for `layers`.')
            sys.exit(1)

        if 'sequence' in ll:
            sequence = ll['sequence']  # Override sequence information

        if processor_map[sequence]:
            error_exit('Layer was already specified', sequence)
        if 'processors' in ll:
            processor_map[sequence] = ll['processors']
        if not processor_map[sequence]:
            error_exit('`processors` must not be zero or missing', sequence)

        if 'output_processors' in ll:
            output_map[sequence] = ll['output_processors']
            if not output_map[sequence]:
                error_exit('output_processors` cannot be zero', sequence)

        if 'pad' in ll:
            val = ll['pad']
            if val < 0:
                error_exit(f'Unsupported value {val} for `pad`', sequence)
            padding[sequence] = [val, val]
        if 'max_pool' in ll:
            val = ll['max_pool']
            pool[sequence] = [val, val]
            pooling_enabled[sequence] = True
        elif 'avg_pool' in ll:
            val = ll['avg_pool']
            pool[sequence] = [val, val]
            pooling_enabled[sequence] = True
            average[sequence] = 1

        if 'pool_stride' in ll:
            val = ll['pool_stride']
            pool_stride[sequence] = [val, val]

        if 'quantization' in ll:
            val = ll['quantization']
            if val not in [1, 2, 4, 8]:
                error_exit('`quantization` must be 1, 2, 4, or 8', sequence)
            quantization[sequence] = val

        if 'in_dim' in ll:
            if ll['in_dim'].lower() == 'flatten':
                input_dim[sequence] = [1, 1]
            else:
                input_dim[sequence] = ll['in_dim']
        if 'in_offset' in ll:
            input_offset[sequence] = ll['in_offset']
        if 'out_offset' in ll:
            output_offset[sequence] = ll['out_offset']

        if 'activate' in ll:
            if ll['activate'].lower() == 'relu':
                relu[sequence] = 1
            else:
                error_exit(f'Unknown value "{ll["activate"]}" for `activate`', sequence)
                sys.exit(1)

        if 'convolution' in ll:
            conv = ll['convolution'].lower()
            if conv == 'conv1d':
                convolution[sequence] = 1
            elif conv == 'conv2d':
                convolution[sequence] = 2
            else:
                error_exit(f'Unknown value "{ll["convolution"]}" for `convolution`', sequence)
                sys.exit(1)

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
            val = str(ll['kernel_size']).lower()
            if convolution[sequence] == 2:
                if device == 84 and val not in ['3x3'] \
                        or device >= 85 and val not in ['1x1', '3x3']:
                    error_exit('Unsupported value for `kernel_size`', sequence)
                kernel_size[sequence] = [int(val[0]), int(val[2])]
            else:
                if val not in ['9']:
                    error_exit('Unsupported value for `kernel_size`', sequence)
                kernel_size[sequence] = [int(val), 1]
        elif convolution[sequence] == 1:  # Set default for 1D convolution
            kernel_size[sequence] = DEFAULT_1D_KERNEL

        if 'stride' in ll:
            val = ll['stride']
            if pooling_enabled[sequence]:
                # Must use the default stride when pooling, otherwise stride can be set
                if convolution == 2 and val != 1 or val != 3:
                    error_exit('Cannot set `stride` to non-default value when pooling', sequence)
                if convolution != 2:
                    stride[sequence] = [3, 1]  # Fix default for 1D
            else:
                # Stride can be set
                stride[sequence] = [val, val]

        # Fix up values for 1D Convolution
        if convolution[sequence] == 1:
            padding[sequence][1] = 0
            pool[sequence][1] = 0
            pool_stride[sequence][1] = 1
            stride[sequence][1] = 1

        sequence += 1

    # Sequence specification may have holes. Contract to the used layers.
    for ll in range(tc.dev.MAX_LAYERS-1, -1, -1):
        if processor_map[ll] is None:
            del processor_map[ll]
            del padding[ll]
            del pool[ll]
            del pool_stride[ll]
            del input_dim[ll]
            del input_offset[ll]
            del output_offset[ll]
            del average[ll]
            del relu[ll]
            del big_data[ll]
            del quantization[ll]
            del output_map[ll]
            del output_width[ll]
            del convolution[ll]
            del dilation[ll]
            del kernel_size[ll]
            del stride[ll]
            del pooling_enabled[ll]

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
    # Check first layer
    if input_offset[0] is None:
        input_offset[0] = 0
    # Check last layer
    if output_map[-1] is None and 'output_map' in cfg:
        output_map[-1] = cfg['output_map']
    if output_width[-1] != 8 and relu[-1]:
        error_exit('`output_width` must be 8 when activation is used', len(relu))

    # Fix up defaults for Conv1D:
    for ll, e in enumerate(convolution):
        if e == 1:
            kernel_size[ll] = [9, 1]

    settings = {}
    settings['padding'] = padding
    settings['pool'] = pool
    settings['pooling_enabled'] = pooling_enabled
    settings['pool_stride'] = pool_stride
    settings['input_dim'] = input_dim
    settings['input_offset'] = input_offset
    settings['output_offset'] = output_offset
    settings['processor_map'] = processor_map
    settings['average'] = average
    settings['relu'] = relu
    settings['big_data'] = big_data
    settings['quantization'] = quantization
    settings['output_processor_map'] = output_map
    settings['output_width'] = output_width
    settings['convolution'] = convolution
    settings['dilation'] = dilation
    settings['kernel_size'] = kernel_size
    settings['stride'] = stride

    return cfg, settings
