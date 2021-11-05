#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Generate sample input data.
"""

import numpy as np

# CHW, in this example 3x11x11
d = np.random.randint(-128, 127, (3, 11, 11), dtype=np.int64)
np.save('sample_test_layers', d, allow_pickle=False, fix_imports=False)

# Use fixed alternating values
# d = np.full((64, 8, 8), -86, dtype=np.int64)
# d[:,:,::2] = 0x55
# np.save('sample_test_burnin_64x8x8', d, allow_pickle=False, fix_imports=False)

# Sample for 7x12x12 element-wise operations (double the channels for two operators)
# d = np.random.randint(-128, 127, (14, 12, 12), dtype=np.int64)
# np.save('sample_test_eltwise7', d, allow_pickle=False, fix_imports=False)

# Sample with known data:
# MLP shape is Nx1x1
# d = np.array([[[-53]], [[-11]], [[44]], [[-116]], [[-42]], [[-72]], [[-50]], [[-46]],
#               [[-22]], [[-56]], [[34]], [[16]]], dtype=np.int64)
# np.save('sample_test_mlp12to2', d, allow_pickle=False, fix_imports=False)

# For --synthesize-input
# with open('sample_test_fifostream-vga.npy', mode='wb') as fhandle:
#     d = np.random.randint(-128, 127, (3, 480, 640), dtype=np.int64)
#     l = 8
#     while l < 480:
#         d[:,l:l+8,:] = ((d[:,l-8:l,:] + 128 + 0x07) % 256) - 128
#         l += 8
#     np.save(fhandle, d, allow_pickle=False, fix_imports=False)
