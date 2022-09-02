###################################################################################################
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Load hard-coded sample weights from .npy files.
"""
import os

import numpy as np

from . import op
from .eprint import eprint
from .utils import fls


def load(
        dataset,
        quantization,
        output_shift,
        cfg_layers,
        cfg_weights=None,
        conv_groups=None,
        operator=None,
        bypass=None,
        filename=None,
):
    """
    Return sample weights.
    """
    weights = []
    output_channels = []
    input_channels = []

    dataset = dataset.lower()

    # Load weights saved using:
    #    w = np.random.randint(-128, 127, (2, 64, 64, 3, 3), dtype=np.int8)
    #    np.save(f'tests/{dataset}', w)

    w = []
    if filename is not None:
        fname = filename
    elif cfg_weights is None:
        fname = os.path.join('tests', f'weights_{dataset}.npy')
    else:
        fname = os.path.join('tests', f'{cfg_weights}.npy')
    if not os.path.exists(fname):
        eprint(f'Sample weights file {fname} does not exist!')
    with open(fname, mode='rb') as file:
        print(f'Reading weights from {fname}...')
        try:
            while True:
                w.append(np.load(file))
        except ValueError:
            pass

    if len(w) == 1:  # If the weights file wasn't a list
        w = w[0]

    ll = 0
    seq = 0
    while seq < cfg_layers:
        while seq < cfg_layers and (operator[seq] == op.NONE or bypass[ll]):
            seq += 1
        if seq >= cfg_layers:
            break

        # Set to default?
        quant = 8 if quantization[seq] is None else quantization[seq]

        # Re-quantize if needed (these are random sample weights, so no need to round etc.)
        max_w = int(w[ll].max())
        if max_w < 0:
            max_w += 1
        min_w = int(w[ll].min())
        if min_w < 0:
            min_w += 1
        current_quant = max(fls(abs(min_w)), fls(abs(max_w))) + 2
        if current_quant > 8:  # Either way, more than 8 bits is an error
            raise ValueError('ERROR: Weight file includes values larger than 8 bit!')
        if quant == -1:
            w[ll][np.where(w[ll] >= 0)] = 1
            w[ll][np.where(w[ll] < 0)] = -1
        elif current_quant > abs(quant):
            w[ll] >>= current_quant - abs(quant)

        # Specified output_shift?
        if output_shift[seq] is None:
            output_shift[seq] = 0
        # Add based on quantization
        output_shift[seq] += 8 - abs(quant)

        mult = conv_groups[seq] if operator[seq] == op.CONVTRANSPOSE2D else 1
        output_channels.append(w[ll].shape[0] * mult)  # Output channels
        mult = conv_groups[seq] if operator[seq] != op.CONVTRANSPOSE2D else 1
        input_channels.append(w[ll].shape[1] * mult)  # Input channels
        if len(w[ll].shape) == 4:
            weights.append(w[ll].reshape(-1, w[ll].shape[-2], w[ll].shape[-1]))
        else:
            weights.append(w[ll].reshape(-1, w[ll].shape[-1]))

        ll += 1
        seq += 1

    return len(weights), weights, output_shift, \
        input_channels, output_channels


def load_bias(
        layers,
        cfg_bias=None,
        no_bias=None,
        operator=None,
        bypass=None,
        filename=None,
):
    """
    Return sample bias weights.
    """
    no_bias = no_bias or []
    bias = [None] * layers

    if cfg_bias is not None or filename is not None:
        ll = 0
        seq = 0
        if filename is not None:
            fname = filename
        else:
            fname = os.path.join('tests', f'bias_{cfg_bias}.npy')
        if not os.path.exists(fname):
            eprint(f'Sample bias file {fname} does not exist!')
        with open(fname, mode='rb') as file:
            print(f'Reading bias from {fname}...')
            try:
                while ll < layers:
                    if operator[ll] != op.NONE and not bypass[ll]:
                        if ll not in no_bias:
                            bias[seq] = np.load(file)
                        else:
                            _ = np.load(file)
                        seq += 1
                    ll += 1
            except ValueError:
                pass

    return bias
