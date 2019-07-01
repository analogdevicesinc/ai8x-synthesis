###################################################################################################
#
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
###################################################################################################
"""
Contains hard coded sample weights.
"""
import sys
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

    if dataset == 'test_conv1d':
        layers = 1
        w = np.array([[[-1, -56, -24, 125, 90, 127, -55, 37, -33],
                       [-119, 127, -33, -128, 97, 101, -128, -128, -12],
                       [-60, 116, -128, -62, -56, 85, 108, -11, 42]],
                      [[60, 127, -128, -117, -128, 87, 127, -128, 127],
                       [7, 120, -92, -128, -51, -44, -128, -128, 97],
                       [-70, 127, 127, 36, 124, -80, 44, 127, -82]],
                      [[-82, 42, 48, 127, -92, 63, 127, 127, -128],
                       [91, -5, 4, -33, 83, -28, -128, 127, -119],
                       [-22, 97, 118, -49, -128, -128, 60, -128, 69]],
                      [[-84, 127, -128, -70, 19, -58, -128, 127, -2],
                       [127, 81, -60, 33, -128, -55, 10, -46, 127],
                       [-71, -114, 98, 105, 64, -2, -67, 64, 82]],
                      [[-71, 78, -128, 127, 1, -128, -81, 127, -64],
                       [-9, 127, -83, -128, -61, -65, 127, 118, -67],
                       [-56, 127, 127, 127, -119, 77, 95, 4, 99]]],
                     dtype=np.int64)
        input_channels.append(w.shape[1])  # Input channels
        output_channels.append(w.shape[0])  # Output channels
        weights.append(w.reshape(-1, w.shape[-1]))
        bias.append(None)
    else:
        print(f"No sample weights for dataset `{dataset}`.")
        sys.exit(1)

    return layers, weights, bias, fc_weights, fc_bias, input_channels, output_channels
