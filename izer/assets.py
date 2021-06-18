###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Copy assets
"""
import os
import shutil

from . import state
from . import tornadocnn as tc


def copy(
        base: str,
        source: str,
        target: str,
        test_name: str,
) -> None:
    """
    Copy all files from `base`/`source` to `target`/`test_name`.
    """
    dst = os.path.join(target, test_name)
    for _, _, files in sorted(os.walk(os.path.join(base, source))):
        for name in sorted(files):
            shutil.copy(os.path.join(base, source, name), dst)


def from_template(
        base: str,
        source: str,
        target: str,
        test_name: str,
        board_name: str,
        insert: str = '',
) -> None:
    """
    Copy all files `base`/`source` to `target`/`test_name`, with file name and
    content substitution.
    """
    template = 'template'

    assert tc.dev is not None
    if state.riscv:
        elf_file = f'{tc.dev.partnum.lower()}-combined.elf'
    else:
        elf_file = f'{tc.dev.partnum.lower()}.elf'

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
                        outfile.write(
                            line.replace('##__PROJ_NAME__##', test_name).
                            replace('##__ELF_FILE__##', elf_file).
                            replace('##__BOARD__##', board_name).
                            replace('##__FILE_INSERT__##', insert)
                        )
            else:
                shutil.copy(os.path.join(base, source, name), os.path.join(target, test_name))
