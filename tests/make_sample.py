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

