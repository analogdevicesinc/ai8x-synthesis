#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Confidential
#
# Written by RM
###################################################################################################
"""
Generate sample input data.
"""

import numpy as np

# CHW, in this example 3x11x11
d = np.random.randint(-128, 127, (3, 11, 11), dtype=np.int64)
np.save('sample_test_layers', d, allow_pickle=False, fix_imports=False)

# Sample with known data:
# MLP shape is Nx1x1
# d = np.array([[[-53]], [[-11]], [[44]], [[-116]], [[-42]], [[-72]], [[-50]], [[-46]], [[-22]], [[-56]], [[34]], [[16]]], dtype=np.int64)
# np.save('sample_test_mlp12to2', d, allow_pickle=False, fix_imports=False)


