###################################################################################################
# Copyright (C) 2020-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Contains hard coded sample inputs.
"""
import operator
import os
from functools import reduce

import numpy as np

from . import stats
from .eprint import eprint
from .utils import s2u, u2s


def get(
        filename,
        synthesize_input=None,
        synthesize_words=8,
):
    """
    Return a sample input image from the file name `filename` in channel-first format
    (i.e., CL, CHW)
    """
    if not os.path.exists(filename):
        eprint(f'Sample data file {filename} does not exist!')

    # Load data saved using
    # np.save(os.path.join('tests', f'sample_{dataset}'), data,
    #         allow_pickle=False, fix_imports=False)

    data = np.load(filename)
    if data.dtype.type is not np.dtype('int64').type:
        eprint(f'The sample data array in {filename} is of type {data.dtype}, rather than '
               'int64!')

    shape = data.shape
    stats.resourcedict['input_size'] = reduce(operator.mul, shape)

    if synthesize_input is not None:
        # Every 8 (or synthesize_words) words, add data to the
        # combined 32-bit word for up to 4 channels
        if shape[0] < 1 or shape[0] > 4:
            eprint('`--synthesize-input` requires 1 to 4 input channels.')
        data = data.reshape(shape[0], -1)
        if data.shape[1] % synthesize_words != 0:
            eprint('`--synthesize-words` must be a divisor of the number of pixels per channel '
                   f'({data.shape[1]}).')
        if shape[0] == 3:
            mask = 0xffffff
        elif shape[0] == 2:
            mask = 0xffff
        elif shape[0] == 1:
            mask = 0xff
        else:
            mask = 0xffffffff
        for i in range(synthesize_words, data.shape[1], synthesize_words):
            for j in range(synthesize_words):
                val = 0
                for c in range(shape[0]-1, -1, -1):
                    val = val << 8 | s2u(data[c, i+j-synthesize_words])
                val += synthesize_input
                val &= mask
                for c in range(shape[0]-1, -1, -1):
                    data[c, i+j] = u2s((val >> c * 8) & 0xff)
        data = data.reshape(shape)

    return data
