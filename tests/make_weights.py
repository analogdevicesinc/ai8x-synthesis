#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Generate sample weight data
"""

import numpy as np

# The generated file is one of the following:
# 1) Five-dimensional NP array (Layers, ChOut, ChIn, KernX, KernY) for Conv2D

d = np.random.randint(-128, 127, (1, 2, 5, 1, 1), dtype=np.int64)
np.save('weights_test_conv2Dk1x1', d, allow_pickle=False, fix_imports=False)

# 2) Four-dimensional NP array (Layers, ChOut, ChIn, KernL) for Conv1D
# 3) Sequence of four-dimensional NP arrays (ChOut, ChIn, KernX, KernY) for Conv2D cases where the
#    input channels and output channels aren't the same. In that case, use "with open() as fhandle:"
#    and multiple np.save(handle, ...)

#with open('weights_test_wide3to512to3in.npy', mode='wb') as fhandle:
#  d = np.random.randint(-128, 127, (512, 3, 3, 3), dtype=np.int64)
#  np.save(fhandle, d, allow_pickle=False, fix_imports=False)
#  d = np.random.randint(-128, 127, (96, 512, 3, 3), dtype=np.int64)
#  np.save(fhandle, d, allow_pickle=False, fix_imports=False)
#  d = np.random.randint(-128, 127, (3, 96, 3, 3), dtype=np.int64)
#  np.save(fhandle, d, allow_pickle=False, fix_imports=False)
