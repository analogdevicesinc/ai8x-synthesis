#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Generate sample bias data
"""

import numpy as np

# The generated file is a sequence of one-dimensional NP arrays with ChOut elements.
# Use "with open() as fhandle:".

d = np.full((1), 8, dtype=np.int64)

with open('bias_eight.npy', mode='wb') as file:
    np.save(file, d, allow_pickle=False, fix_imports=False)
