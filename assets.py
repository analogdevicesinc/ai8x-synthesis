###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Copy assets
"""
import os
import shutil


def copy(
        base,
        source,
        target,
        test_name,
):
    """
    Copy all files from `base`/`source` to `target`/`test_name`.
    """
    dst = os.path.join(target, test_name)
    for _, _, files in sorted(os.walk(os.path.join(base, source))):
        for name in sorted(files):
            shutil.copy(os.path.join(base, source, name), dst)
