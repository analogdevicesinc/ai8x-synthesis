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
# 2) Four-dimensional NP array (Layers, ChOut, ChIn, KernL) for Conv1D
# 3) Sequence of four-dimensional NP arrays (ChOut, ChIn, KernX, KernY) for Conv2D cases where the
#    input channels and output channels aren't the same. In that case, use "with open() as fhandle:"
#    and multiple np.save(handle, ...)

d = np.random.randint(-128, 127, (1, 5, 5, 3, 3), dtype=np.int64)
np.save('weights_test_5x17x23', d, allow_pickle=False, fix_imports=False)
