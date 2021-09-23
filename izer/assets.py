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
    assert tc.dev is not None

    template = 'template'
    part = tc.dev.partnum.lower()
    openocd = state.eclipse_openocd_args.replace('##__TARGET_LC__##', part)
    prefix = 'riscv-none-embed-' if state.riscv else 'arm-none-eabi-'
    elf_file = f'{test_name}-combined.elf' if state.riscv else f'{test_name}.elf'

    basepath = os.path.join(base, source)
    for folderpath, _, files in sorted(os.walk(basepath)):
        folder = os.path.relpath(folderpath, basepath)
        if folder != '.':
            test_path = os.path.join(test_name, folder)
            os.makedirs(os.path.join(target, test_path), exist_ok=True)
        else:
            test_path = test_name

        for name in sorted(files):
            if folder != '.':
                source_path = os.path.join(folder, name)
            else:
                source_path = name
            if name.startswith(template):
                dst = os.path.join(
                    target,
                    test_path,
                    name[len(template):].replace('##__PROJ_NAME__##', test_name),
                )
                with open(
                    os.path.join(base, source, source_path)
                ) as infile, open(dst, 'w+') as outfile:
                    for line in infile:
                        outfile.write(
                            line.replace('##__PROJ_NAME__##', test_name).
                            replace('##__ELF_FILE__##', elf_file).
                            replace('##__BOARD__##', board_name).
                            replace('##__FILE_INSERT__##', insert).
                            replace('##__OPENOCD_PARAMS__##', openocd).
                            replace('##__TARGET_UC__##', part.upper()).
                            replace('##__TARGET_LC__##', part).
                            replace('##__ADDITIONAL_INCLUDES__##', state.eclipse_includes).
                            replace('##__GCC_PREFIX__##', prefix).
                            replace('##__GCC_SUFFIX__##', state.defines).
                            replace('##__DEFINES__##', state.defines).
                            replace('##__DEFINES_ARM__##', state.defines_arm).
                            replace('##__ARM_DEFINES__##', state.defines_arm).
                            replace('##__DEFINES_RISCV__##', state.defines_riscv).
                            replace('##__RISC_DEFINES__##', state.defines_riscv).
                            replace('##__ADDITIONAL_VARS__##', state.eclipse_variables)
                        )
            else:
                shutil.copy(os.path.join(base, source, source_path),
                            os.path.join(target, test_path))
