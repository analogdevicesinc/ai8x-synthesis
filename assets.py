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


def eclipse_template(
        base,
        source,
        target,
        test_name,
):
    """
    Copy all files `base`/`source` to `target`/`test_name`, with file name and
    content substitution.
    """
    template = 'template'

    for _, _, files in sorted(os.walk(os.path.join(base, source))):
        for name in sorted(files):
            if name.startswith(template):
                dst = os.path.join(
                    target,
                    test_name,
                    name[len(template):].replace('##__PROJ_NAME__##', test_name),
                )
                with open(os.path.join(base, source, name)) as infile, open(dst, 'w+') as outfile:
                    for line in infile:
                        outfile.write(line.replace('##__PROJ_NAME__##', test_name))
            else:
                shutil.copy(os.path.join(base, source, name), os.path.join(target, test_name))
