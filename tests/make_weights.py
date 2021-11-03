#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Generate sample weight data
"""

import numpy as np

# The generated file is one of the following:
# 1) Five-dimensional NP array (Layers, ChOut, ChIn, KernX, KernY) for Conv2D

d = np.random.randint(-128, 127, (1, 2, 5, 1, 1), dtype=np.int64)
np.save('weights_test_conv2Dk1x1', d, allow_pickle=False, fix_imports=False)

# Fixed alternating values (single layer)
# d = np.full((1, 64, 64, 3, 3), -86, dtype=np.int64)
# d[:, ::2, :, ::2, ::2] = 0x55
# d[:, ::2, :, 1::2, 1::2] = 0x55
# d[:, 1::2, :, ::2, 1::2] = 0x55
# d[:, 1::2, :, 1::2, ::2] = 0x55
# np.save('weights_test_burnin_64-64', d, allow_pickle=False, fix_imports=False)

# Fixed alternating values on x4 boundary (single layer)
# d = np.full((1, 60, 3, 3, 3), -86, dtype=np.int64)
# d[:, ::8, :, ::2, ::2] = 0x55
# d[:, ::8, :, 1::2, 1::2] = 0x55
# d[:, 1::8, :, ::2, ::2] = 0x55
# d[:, 1::8, :, 1::2, 1::2] = 0x55
# d[:, 2::8, :, ::2, ::2] = 0x55
# d[:, 2::8, :, 1::2, 1::2] = 0x55
# d[:, 3::8, :, ::2, ::2] = 0x55
# d[:, 3::8, :, 1::2, 1::2] = 0x55
# d[:, 4::8, :, ::2, 1::2] = 0x55
# d[:, 4::8, :, 1::2, ::2] = 0x55
# d[:, 5::8, :, ::2, 1::2] = 0x55
# d[:, 5::8, :, 1::2, ::2] = 0x55
# d[:, 6::8, :, ::2, 1::2] = 0x55
# d[:, 6::8, :, 1::2, ::2] = 0x55
# d[:, 7::8, :, ::2, 1::2] = 0x55
# d[:, 7::8, :, 1::2, ::2] = 0x55
# np.save('weights_test_alternating', d, allow_pickle=False, fix_imports=False)

# Same for element-wise (ChIn == ChOut)
# d = np.zeros((1, 7, 7, 1, 1), dtype=np.int64)
# np.save('weights_test_eltwise7', d, allow_pickle=False, fix_imports=False)

# Sample with known data for MLP:
# 1, 2, 12, 1, 1
# d = np.array([[[[[-29]], [[-101]], [[67]], [[-37]], [[19]], [[-89]], [[121]], [[63]],
#                 [[116]], [[44]], [[51]], [[84]]],
#                [[[-99]], [[-2]], [[24]], [[41]], [[-31]], [[71]], [[71]], [[82]],
#                 [[41]], [[2]], [[-128]], [[68]]]]], dtype=np.int64)
# np.save('weights_test_mlp12to2', d, allow_pickle=False, fix_imports=False)

# 2) Four-dimensional NP array (Layers, ChOut, ChIn, KernL) for Conv1D
# 3) Sequence of four-dimensional NP arrays (ChOut, ChIn, KernX, KernY) for Conv2D cases where the
#    input channels and output channels aren't the same. In that case, use
#    "with open() as fhandle:" and multiple np.save(handle, ...)

# with open('weights_test_wide3to512to3in.npy', mode='wb') as fhandle:
#   d = np.random.randint(-128, 127, (512, 3, 3, 3), dtype=np.int64)
#   np.save(fhandle, d, allow_pickle=False, fix_imports=False)
#   d = np.random.randint(-128, 127, (96, 512, 3, 3), dtype=np.int64)
#   np.save(fhandle, d, allow_pickle=False, fix_imports=False)
#   d = np.random.randint(-128, 127, (3, 96, 3, 3), dtype=np.int64)
#   np.save(fhandle, d, allow_pickle=False, fix_imports=False)
