###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
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

    # Load weights saved using
    # np.save(f'tests/{dataset}', w.reshape(w.shape[0], w.shape[1], w.shape[-2], w.shape[-1]),
    #         allow_pickle=False, fix_imports=False)

    layers = 1
    w = np.load(os.path.join('tests', f'weights_{dataset}.npy'),
                allow_pickle=False, fix_imports=False)
    output_channels.append(w.shape[0])  # Output channels
    input_channels.append(w.shape[1])  # Input channels
    if len(w.shape) == 4:
        weights.append(w.reshape(-1, w.shape[-2], w.shape[-1]))
    else:
        weights.append(w.reshape(-1, w.shape[-1]))
    bias.append(None)

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
