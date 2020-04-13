###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Checkpoint File Routines
"""
import sys

import numpy as np
import torch

import tornadocnn


def load(
        checkpoint_file,
        arch,
        fc_layer,
        quantization,
        bias_quantization,
        verbose=False,
        no_bias=None,
):
    """
    Load weights and biases from `checkpoint_file`. If `arch` is not None and does not match
    the architecuture in the checkpoint file, abort with an error message. If `fc_layer` is
    `True`, configure a single fully connected classification layer for software rather than
    hardware.
    `quantization` is a list of expected bit widths for the layer weights (always 8 for AI84).
    This value is checked against the weight inputs.
    `bias_quantization` is a list of the expected bit widths for the layer weights (always
    8 for AI84/AI85).
    In addition to returning weights anf biases, this function configures the network output
    channels and the number of layers.
    When `verbose` is set, display the shapes of the weights.
    """
    no_bias = no_bias or []
    weights = []
    bias = []
    fc_weights = []
    fc_bias = []
    weight_keys = []
    bias_keys = []
    quant = []
    bias_quant = []
    weight_min = []
    weight_max = []
    weight_size = []
    bias_min = []
    bias_max = []
    bias_size = []

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    print(f'Reading {checkpoint_file} to configure network weights...')

    if 'state_dict' not in checkpoint or 'arch' not in checkpoint:
        raise RuntimeError("\nNo `state_dict` or `arch` in checkpoint file.")

    if arch and checkpoint['arch'].lower() != arch.lower():
        print(f"Network architecture of configuration file ({arch}) does not match "
              f"network architecture of checkpoint file ({checkpoint['arch']}).")
        sys.exit(1)

    checkpoint_state = checkpoint['state_dict']
    layers = 0
    num_conv_layers = len(quantization)
    have_fc_layer = False
    output_channels = []
    input_channels = []
    param_count = 0
    param_size = 0

    for _, k in enumerate(checkpoint_state.keys()):
        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['weight']:
            module, op = k.split(sep='.', maxsplit=1)
            op = op.rsplit(sep='.', maxsplit=1)[0]
            if module != 'fc' or module == 'fc' and not fc_layer:
                if layers >= num_conv_layers:
                    continue
                quant.append(quantization[layers])

                w = checkpoint_state[k].numpy().astype(np.int64)
                w_min, w_max = w.min(), w.max()

                assert w_min >= -(2**(quantization[layers]-1))
                assert w_max < 2**(quantization[layers]-1)

                weight_min.append(w_min)
                weight_max.append(w_max)

                input_channels.append(w.shape[1])  # Input channels
                output_channels.append(w.shape[0])  # Output channels

                w_count = np.prod(w.shape)
                param_count += w_count
                w_size = (w_count * 8 + (quantization[layers]-1)) // quantization[layers]
                weight_size.append(w_size)
                param_size += w_size

                if op == 'convtranspose2d':
                    # For ConvTranspose2d, flip the weights as follows:
                    w = np.flip(w, axis=(2, 3)).swapaxes(0, 1)

                if len(w.shape) == 2:  # linear - add dummy 'channel'
                    w = np.expand_dims(w, axis=0)
                else:  # conv1d, conv2d, ... - combine input and output channels
                    w = np.reshape(w, (-1, ) + w.shape[2:])

                weights.append(w)
                weight_keys.append(k)
                # Is there a bias for this layer?
                bias_name = operation + '.bias'

                if bias_name in checkpoint_state and layers not in no_bias:
                    w = checkpoint_state[bias_name].numpy(). \
                        astype(np.int64) // tornadocnn.dev.BIAS_DIV

                    w_min, w_max = w.min(), w.max()
                    assert w_min >= -(2**(bias_quantization[layers]-1))
                    assert w_max < 2**(bias_quantization[layers]-1)

                    bias_min.append(w_min)
                    bias_max.append(w_max)

                    bias.append(w)
                    bias_keys.append(bias_name)
                    bias_quant.append(bias_quantization[layers])
                    w_count = np.prod(w.shape)
                    param_count += w_count
                    w_size = (
                        w_count * 8 + (bias_quantization[layers]-1)
                    ) // bias_quantization[layers]
                    bias_size.append(w_size)
                    param_size += w_size
                else:
                    bias.append(None)
                    bias_min.append(0)
                    bias_max.append(0)
                    bias_keys.append('N/A')
                    bias_quant.append(0)
                    bias_size.append(0)
                layers += 1
            elif have_fc_layer:
                print('The network cannot have more than one fully connected software layer, '
                      'and it must be the output layer.')
                sys.exit(1)
            elif fc_layer:
                w = checkpoint_state[k].numpy().astype(np.int64)
                assert w.min() >= -128 and w.max() <= 127
                fc_weights.append(w)
                # Is there a bias for this layer?
                bias_name = operation + '.bias'
                if bias_name in checkpoint_state:
                    # Do not divide bias for FC
                    w = checkpoint_state[bias_name].numpy().astype(np.int64)
                    assert w.min() >= -128 and w.max() <= 127
                    fc_bias.append(w)
                else:
                    fc_bias.append(None)
                have_fc_layer = True

    if verbose:
        print(f'Checkpoint for epoch {checkpoint["epoch"]}, model {checkpoint["arch"]} - '
              'weight and bias data:')
        print('Layer  InCh OutCh  Weights         Quant  Min Max   Size '
              'Key                       Bias       Quant  Min Max Size Key')
        for ll in range(layers):
            if ll < len(weights) and weights[ll] is not None:
                weight_shape = str(weights[ll].shape)
                if bias[ll] is not None:
                    bias_shape = str(bias[ll].shape)
                else:
                    bias_shape = 'N/A'
                print(f'{ll:4}: '
                      f'{input_channels[ll]:5} {output_channels[ll]:5}  '
                      f'{weight_shape:15} '
                      f'{quant[ll]:5} {weight_min[ll]:4} {weight_max[ll]:3} {weight_size[ll]:6} '
                      f'{weight_keys[ll]:25} '
                      f'{bias_shape:10} '
                      f'{bias_quant[ll]:5} {bias_min[ll]:4} {bias_max[ll]:3} {bias_size[ll]:4} '
                      f'{bias_keys[ll]:25}')
        print(f'TOTAL: {layers} layers, {param_count:,} parameters, {param_size:,} bytes')

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
