###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Load hard-coded sample weights from .npy files.
"""
import os

import numpy as np

import op
from utils import fls


def load(
        dataset,
        quantization,
        bias_quantization,  # pylint: disable=unused-argument
        output_shift,
        cfg_layers,
        cfg_weights=None,
        cfg_bias=None,
        no_bias=None,
        conv_groups=None,
        operator=None,
):
    """
    Return sample weights.
    """
    no_bias = no_bias or []
    weights = []
    fc_weights = []
    fc_bias = []
    output_channels = []
    input_channels = []
    layers = 0

    dataset = dataset.lower()

    # Load weights saved using:
    #    w = np.random.randint(-128, 127, (2, 64, 64, 3, 3), dtype=np.int8)
    #    np.save(f'tests/{dataset}', w)

    w = []
    layers = 0
    if cfg_weights is None:
        fname = os.path.join('tests', f'weights_{dataset}.npy')
    else:
        fname = os.path.join('tests', f'{cfg_weights}.npy')
    with open(fname, mode='rb') as file:
        print(f'Reading weights from {fname}...')
        try:
            while True:
                w.append(np.load(file))
                layers += 1
        except ValueError:
            pass

    if layers == 1:  # If the weights file wasn't a list
        w = w[0]
        layers = w.shape[0]

    layers = min(layers, cfg_layers)

    bias = [None] * layers

    if cfg_bias is not None:
        ll = 0
        fname = os.path.join('tests', f'bias_{cfg_bias}.npy')
        with open(fname, mode='rb') as file:
            print(f'Reading bias from {fname}...')
            try:
                while ll < layers:
                    if ll not in no_bias:
                        bias[ll] = np.load(file)
                    ll += 1
            except ValueError:
                pass

    for ll in range(layers):
        # Set to default?
        if quantization[ll] is None:
            quantization[ll] = 8

        # Re-quantize if needed (these random sample weights, so no need to round etc.)
        max_w = int(w[ll].max())
        if max_w < 0:
            max_w += 1
        min_w = int(w[ll].min())
        if min_w < 0:
            min_w += 1
        current_quant = max(fls(abs(min_w)), fls(abs(max_w))) + 2
        if current_quant > 8:  # Either way, more than 8 bits is an error
            raise ValueError('ERROR: Weight file includes values larger than 8 bit!')
        if current_quant > quantization[ll]:
            w[ll] >>= current_quant - quantization[ll]

        # Specified output_shift?
        if output_shift[ll] is None:
            output_shift[ll] = 0
        # Add based on quantization
        output_shift[ll] += 8 - quantization[ll]

        mult = conv_groups[ll] if operator[ll] == op.CONVTRANSPOSE2D else 1
        output_channels.append(w[ll].shape[0] * mult)  # Output channels
        mult = conv_groups[ll] if operator[ll] != op.CONVTRANSPOSE2D else 1
        input_channels.append(w[ll].shape[1] * mult)  # Input channels
        if len(w[ll].shape) == 4:
            weights.append(w[ll].reshape(-1, w[ll].shape[-2], w[ll].shape[-1]))
        else:
            weights.append(w[ll].reshape(-1, w[ll].shape[-1]))

    return layers, weights, bias, output_shift, \
        fc_weights, fc_bias, input_channels, output_channels
