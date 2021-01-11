###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Contains hard coded sample inputs.
"""
import numpy as np


def get(
        filename,
):
    """
    Return a sample input image from the file name `filename` in channel-first format
    (i.e., CL, CHW)
    """

    # Load data saved using
    # np.save(os.path.join('tests', f'sample_{dataset}'), data,
    #         allow_pickle=False, fix_imports=False)

    return np.load(filename)
