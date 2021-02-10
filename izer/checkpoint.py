###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Checkpoint File Routines
"""
import sys

import numpy as np
import torch

from . import op as opn
from . import tornadocnn as tc
from .eprint import eprint
from .utils import fls


def load(
        checkpoint_file,
        arch,
        quantization,
        bias_quantization,
        output_shift,
        kernel_size,
        operator,
        verbose=False,
        no_bias=None,
        conv_groups=None,
):
    """
    Load weights and biases from `checkpoint_file`. If `arch` is not None and does not match
    the architecture in the checkpoint file, abort with an error message.
    `quantization` is a list of expected bit widths for the layer weights. `-1` is a special value
    denoting -1/+1. This value is checked against the weight inputs.
    `bias_quantization` is a list of the expected bit widths for the layer weights (always 8).
    In addition to returning weights and biases, this function configures the network output
    channels and the number of layers.
    When `verbose` is set, display the shapes of the weights.
    """
    no_bias = no_bias or []
    weights = []
    bias = []
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
        eprint(f"Network architecture of configuration file ({arch}) does not match "
               f"network architecture of checkpoint file ({checkpoint['arch']}).")

    checkpoint_state = checkpoint['state_dict']
    layers = 0
    num_conv_layers = len(quantization)
    output_channels = []
    input_channels = []
    param_count = 0
    param_size = 0
    error_exit = False
    seq = 0

    for _, k in enumerate(checkpoint_state.keys()):
        # Skip over non-weight layers
        while seq < len(operator) and operator[seq] == opn.NONE:
            seq += 1

        operation, parameter = k.rsplit(sep='.', maxsplit=1)
        if parameter in ['weight']:
            _, op = k.split(sep='.', maxsplit=1)
            op = op.rsplit(sep='.', maxsplit=1)[0]
            if layers >= num_conv_layers or seq >= num_conv_layers:
                continue

            w = checkpoint_state[k].numpy().astype(np.int64)
            w_min, w_max, w_abs = w.min(), w.max(), np.abs(w)

            # Determine quantization or make sure that what was given fits
            if quantization[seq] is not None:
                if quantization[seq] == -1:
                    assert w_abs.min() == w_abs.max() == 1
                else:
                    assert w_min >= -(2**(quantization[seq]-1))
                    assert w_max < 2**(quantization[seq]-1)
            else:
                if tc.dev.SUPPORT_BINARY_WEIGHTS and w_abs.min() == w_abs.max() == 1:
                    quantization[seq] = -1
                else:
                    if w_max > 0:
                        w_max_m = int(w_max)
                    else:
                        w_max_m = int(abs(w_max)) - 1
                    if w_min > 0:
                        w_min_m = int(w_min)
                    else:
                        w_min_m = int(abs(w_min)) - 1
                    quantization[seq] = 1 << (fls(max(fls(w_max_m), fls(w_min_m)) + 1) + 1)
                assert quantization[seq] <= 8
            quant.append(quantization[seq])

            weight_min.append(w_min)
            weight_max.append(w_max)

            if operator[seq] == opn.CONVTRANSPOSE2D:
                # For ConvTranspose2d, flip the weights as follows:
                w = np.flip(w, axis=(2, 3)).swapaxes(0, 1)

            mult = conv_groups[seq] if operator[seq] != opn.CONVTRANSPOSE2D else 1
            input_channels.append(w.shape[1] * mult)  # Input channels
            mult = conv_groups[seq] if operator[seq] == opn.CONVTRANSPOSE2D else 1
            output_channels.append(w.shape[0] * mult)  # Output channels

            if len(w.shape) == 2:  # MLP
                if kernel_size[seq][0] != 1 or kernel_size[seq][1] != 1:
                    eprint(f'The `kernel_size` for the MLP layer {seq} should '
                           f'be set to 1x1 instead of '
                           f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.', exit_code=None)
                    error_exit = True
            elif len(w.shape) == 3:  # 1D
                if kernel_size[seq][0] != w.shape[2] or kernel_size[seq][1] != 1:
                    eprint(f'The `kernel_size` for the 1D layer {seq} should '
                           f'be set to {w.shape[2]}x1 instead of '
                           f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.', exit_code=None)
                    error_exit = True
            elif len(w.shape) == 4:  # 2D
                if kernel_size[seq][0] != w.shape[2] \
                   or kernel_size[seq][1] != w.shape[3]:
                    eprint(f'The `kernel_size` for the 2D layer {seq} should '
                           f'be set to {w.shape[2]}x{w.shape[3]} instead of '
                           f'{kernel_size[seq][0]}x{kernel_size[seq][1]}.', exit_code=None)
                    error_exit = True

            w_count = np.prod(w.shape)
            param_count += w_count
            w_size = (w_count * abs(quantization[seq]) + 7) // 8
            weight_size.append(w_size)
            param_size += w_size

            if len(w.shape) == 2:  # linear - add dummy 'channel'
                w = np.expand_dims(w, axis=0)
            else:  # conv1d, conv2d, ... - combine input and output channels
                w = np.reshape(w, (-1, ) + w.shape[2:])

            weights.append(w)
            weight_keys.append(k)

            # Is there a bias for this layer?
            bias_name = operation + '.bias'

            if bias_name in checkpoint_state and seq not in no_bias:
                w = checkpoint_state[bias_name].numpy(). \
                    astype(np.int64) // tc.dev.BIAS_DIV

                w_min, w_max = w.min(), w.max()
                assert w_min >= -(2**(bias_quantization[seq]-1))
                assert w_max < 2**(bias_quantization[seq]-1)

                bias_min.append(w_min)
                bias_max.append(w_max)

                bias.append(w)
                bias_keys.append(bias_name)
                bias_quant.append(bias_quantization[seq])
                w_count = np.prod(w.shape)
                param_count += w_count
                w_size = (
                    w_count * 8 + (bias_quantization[seq]-1)
                ) // bias_quantization[seq]
                bias_size.append(w_size)
                param_size += w_size
            else:
                bias.append(None)
                bias_min.append(0)
                bias_max.append(0)
                bias_keys.append('N/A')
                bias_quant.append(0)
                bias_size.append(0)

            # Not overriding output_shift?
            if output_shift[seq] is None:
                output_shift_name = operation.rsplit(sep='.', maxsplit=1)[0] + '.output_shift'
                # Is there an output_shift for this layer?
                if output_shift_name in checkpoint_state:
                    w = checkpoint_state[output_shift_name].numpy().astype(np.int64)

                    assert len(w) == 1
                    output_shift[seq] = w[0]
                else:
                    output_shift[seq] = 0

            # Add implicit shift based on quantization
            output_shift[seq] += 8 - abs(quantization[seq])

            layers += 1
            seq += 1

    if verbose:
        print(f'Checkpoint for epoch {checkpoint["epoch"]}, model {checkpoint["arch"]} - '
              'weight and bias data:')
        print(' InCh OutCh  Weights         Quant Shift  Min Max   Size '
              'Key                                 Bias       Quant  Min Max Size Key')
        for ll in range(layers):
            if ll < len(weights) and weights[ll] is not None:
                weight_shape = str(weights[ll].shape)
                if bias[ll] is not None:
                    bias_shape = str(bias[ll].shape)
                else:
                    bias_shape = 'N/A'
                if output_shift[ll] is not None:
                    output_shift_shape = output_shift[ll]
                else:
                    output_shift_shape = 'N/A'
                print(f'{input_channels[ll]:5} {output_channels[ll]:5}  '
                      f'{weight_shape:15} '
                      f'{quant[ll]:5} {output_shift_shape:5} '
                      f'{weight_min[ll]:4} {weight_max[ll]:3} {weight_size[ll]:6} '
                      f'{weight_keys[ll]:35} '
                      f'{bias_shape:10} '
                      f'{bias_quant[ll]:5} {bias_min[ll]:4} {bias_max[ll]:3} {bias_size[ll]:4} '
                      f'{bias_keys[ll]:25}')
        print(f'TOTAL: {layers} layers, {param_count:,} parameters, {param_size:,} bytes')

    if error_exit:
        sys.exit(1)

    return layers, weights, bias, output_shift, \
        input_channels, output_channels
