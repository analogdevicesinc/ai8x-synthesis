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


SUPPORTED_DATASETS = ['mnist', 'fashionmnist', 'cifar-10']


def parse(config_file, ai85=False):  # pylint: disable=unused-argument
    """
    Configure network parameters from the YAML configuration file `config_file`.
    The function returns both the YAML dictionary as well as a settings dictionary.
    `ai85` is `True` for AI85 and AI86, `False` for AI84.
    """

    def error_exit(message, sequence):
        """
        Print error message `message` for layer sequence `sequence` and exit.
        """
        print(f'{message} for layer sequence {sequence} in YAML configuration.')
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

    processor_map = [None] * tc.MAX_LAYERS  # Use this to see whether a layer was configured
    output_map = [None] * tc.MAX_LAYERS
    padding = [1] * tc.MAX_LAYERS
    pool = [0] * tc.MAX_LAYERS
    average = [0] * tc.MAX_LAYERS
    pool_stride = [0] * tc.MAX_LAYERS
    quantization = [8] * tc.MAX_LAYERS
    output_offset = [0] * tc.MAX_LAYERS
    relu = [0] * tc.MAX_LAYERS
    big_data = [False] * tc.MAX_LAYERS

    sequence = 0
    for ll in cfg['layers']:
        if bool(set(ll) - set(['max_pool', 'avg_pool', 'pool_stride', 'out_offset',
                               'activate', 'data_format', 'output_processors', 'processors', 'pad',
                               'quantization', 'sequence'])):
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
            p = ll['pad']
            if p < 0 or p > 2:
                error_exit('Unsupported value {p} for `pad`', sequence)
            padding[sequence] = p
        if 'max_pool' in ll:
            pool[sequence] = ll['max_pool']
        elif 'avg_pool' in ll:
            pool[sequence] = ll['avg_pool']
            average[sequence] = 1
        if 'pool_stride' in ll:
            pool_stride[sequence] = ll['pool_stride']

        if 'quantization' in ll:
            q = ll['quantization']
            if q not in [1, 2, 4, 8]:
                error_exit('`quantization` must be 1, 2, 4, or 8', sequence)
            quantization[sequence] = q

        if 'out_offset' in ll:
            output_offset[sequence] = ll['out_offset']

        if 'activate' in ll:
            if ll['activate'].lower() == 'relu':
                relu[sequence] = 1
            else:
                error_exit(f'Unknown value "{ll["activate"]}" for `activate`', sequence)
                sys.exit(1)

        if 'data_format' in ll:
            if sequence:
                error_exit('`data_format` can only be configured for the first layer', sequence)

            df = ll['data_format'].lower()
            if df in ['chw', 'big']:
                big_data[sequence] = True
            elif df in ['hwc', 'little']:
                pass
            else:
                error_exit('Unknown value for `data_format`', sequence)

        sequence += 1

    # Sequence specification may have holes. Contract to the used layers.
    for ll in range(tc.MAX_LAYERS-1, -1, -1):
        if processor_map[ll] is None:
            del processor_map[ll]
            del padding[ll]
            del pool[ll]
            del pool_stride[ll]
            del output_offset[ll]
            del average[ll]
            del relu[ll]
            del big_data[ll]
            del quantization[ll]
            del output_map[ll]

    # Fix up default output maps
    for ll in range(len(output_map) - 1):  # No output map for very last layer
        if output_map[ll] is None:
            output_map[ll] = processor_map[ll+1]
    if output_map[-1] is None and 'output_map' in cfg:
        output_map[-1] = cfg['output_map']

    settings = {}
    settings['padding'] = padding
    settings['pool'] = pool
    settings['pool_stride'] = pool_stride
    settings['output_offset'] = output_offset
    settings['processor_map'] = processor_map
    settings['average'] = average
    settings['relu'] = relu
    settings['big_data'] = big_data
    settings['quantization'] = quantization
    settings['output_processor_map'] = output_map

    # We don't support changing the following, but leave as parameters
    settings['dilation'] = [[1, 1]] * len(cfg['layers'])
    settings['kernel_size'] = [[3, 3]] * len(cfg['layers'])
    settings['stride'] = [1] * len(cfg['layers'])

    return cfg, settings
