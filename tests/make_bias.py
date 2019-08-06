#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
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
