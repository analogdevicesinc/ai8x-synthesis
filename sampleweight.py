###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Load hard-coded sample weights from .npy files.
"""
import os

import numpy as np

from utils import fls


def load(
        dataset,
        quantization,
        bias_quantization,  # pylint: disable=unused-argument
        cfg_layers,
        cfg_bias,
):
    """
    Return sample weights.
    """
    weights = []
    fc_weights = []
    fc_bias = []
    output_channels = []
    input_channels = []
    layers = 0

    dataset = dataset.lower()

    # Load weights saved using:
    #    w = np.random.randint(-128, 127, (2, 64, 64, 3, 3), dtype=np.int8)
    #    np.save(f'tests/{dataset}', w, allow_pickle=False, fix_imports=False)

    w = []
    layers = 0
    fname = os.path.join('tests', f'weights_{dataset}.npy')
    with open(fname, mode='rb') as file:
        print(f'Reading weights from {fname}...')
        try:
            while True:
                w.append(np.load(file, allow_pickle=False, fix_imports=False))
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
                    bias[ll] = np.load(file, allow_pickle=False, fix_imports=False)
                    ll += 1
            except ValueError:
                pass

    for ll in range(layers):
        # Re-quantize if needed (these random sample weights, so no need to round etc.)
        current_quant = max(fls(int(w[ll].max())), fls(int(w[ll].min() + 1))) + 2
        if current_quant > quantization[ll]:
            w[ll] >>= current_quant - quantization[ll]

        output_channels.append(w[ll].shape[0])  # Output channels
        input_channels.append(w[ll].shape[1])  # Input channels
        if len(w[ll].shape) == 4:
            weights.append(w[ll].reshape(-1, w[ll].shape[-2], w[ll].shape[-1]))
        else:
            weights.append(w[ll].reshape(-1, w[ll].shape[-1]))

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
