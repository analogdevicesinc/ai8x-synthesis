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


def load(dataset, _quantization):
    """
    Return sample weights.
    """
    weights = []
    bias = []
    fc_weights = []
    fc_bias = []
    output_channels = []
    input_channels = []
    layers = 0

    dataset = dataset.lower()

    # Load weights saved using:
    #    w = np.random.randint(-128, 127, (2, 64, 64, 3, 3), dtype=np.int8)
    #    np.save(f'tests/{dataset}', w, allow_pickle=False, fix_imports=False)

    w = np.load(os.path.join('tests', f'weights_{dataset}.npy'),
                allow_pickle=False, fix_imports=False)
    layers = w.shape[0]
    for ll in range(layers):
        output_channels.append(w[ll].shape[0])  # Output channels
        input_channels.append(w[ll].shape[1])  # Input channels
        if len(w[ll].shape) == 4:
            weights.append(w[ll].reshape(-1, w[ll].shape[-2], w[ll].shape[-1]))
        else:
            weights.append(w[ll].reshape(-1, w[ll].shape[-1]))
        bias.append(None)

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
