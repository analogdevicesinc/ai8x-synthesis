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
Contains hard coded sample inputs.
"""
import os

import numpy as np


def get(dataset):
    """
    Return a sample input image for the dataset `dataset` (MNIST/FashionMNIST/CIFAR-10/etc.)
    in CHW format.
    """

    # Load data saved using
    # np.save(os.path.join('tests', f'sample_{dataset}'), data,
    #         allow_pickle=False, fix_imports=False)

    return np.load(os.path.join('tests', f'sample_{dataset.lower()}.npy'),
                   allow_pickle=False, fix_imports=False)
