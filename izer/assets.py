###################################################################################################
# Copyright (C) 2019-2021 Maxim Integrated Products, Inc. All Rights Reserved.
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
    processor_defines = state.defines_riscv if state.riscv else state.defines_arm
    if state.measure_energy and tc.dev.REQUIRE_PMON_GPIO:
        pmon_gpio = 'extern mxc_gpio_cfg_t gpio_trig1, gpio_trig2;'
        cnn_start = 'MXC_GPIO_OutSet(gpio_trig2.port, gpio_trig2.mask)'
        cnn_complete = 'MXC_GPIO_OutClr(gpio_trig2.port, gpio_trig2.mask)'
        sys_start = 'MXC_GPIO_OutSet(gpio_trig1.port, gpio_trig1.mask)'
        sys_complete = 'MXC_GPIO_OutClr(gpio_trig1.port, gpio_trig1.mask)'
    else:
        pmon_gpio = ''
        cnn_start = 'LED_On(1)'
        cnn_complete = 'LED_Off(1)'
        sys_start = 'LED_On(0)'
        sys_complete = 'LED_Off(0)'

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
                    os.path.join(base, source, source_path),
                    mode='r',
                    encoding='utf-8',
                ) as infile, open(dst, mode='w+', encoding='utf-8') as outfile:
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
                            replace('##__PROCESSOR_DEFINES__##', processor_defines).
                            replace('##__ADDITIONAL_VARS__##', state.eclipse_variables).
                            replace('##__PMON_GPIO_PINS__##', pmon_gpio).
                            replace('##__CNN_START__##', cnn_start).
                            replace('##__CNN_COMPLETE__##', cnn_complete).
                            replace('##__SYS_START__##', sys_start).
                            replace('##__SYS_COMPLETE__##', sys_complete)
                        )
            else:
                shutil.copy(os.path.join(base, source, source_path),
                            os.path.join(target, test_path))


def vscode(
    out_stem: str,
    out_branch: str,
    part_num: str = "",
    board: str = "",
    elf_file: str = "",
    defines: str = "",
    i_paths: str = "",
    v_paths: str = "",
):
    """
    Generates vscode project files from a template, and places
    the contents of the template folder `assets/vscode` in `out_stem/out_branch`.
    Optional arguments will load from the global state unless overridden.

    Parameters:
        out_stem:  Output root directory.
        out_branch:  Output sub-directory.
        (optional) part_num:  Target part number.  Ex: MAX78000
        (optional) board:  Target board, case sensitive.  Ex:  EvKit_V1, FTHR_RevA, etc.
        (optional) elf_file:  Sets the name of the output file.  Ex:  hello_world.elf
        (optional) i_paths:  Space-separated include paths the C/C++ parser should use.
        (optional) defines:  Space-separated compiler definitions the C/C++ parser should use.
        (optional) v_paths:  Space-separated additional browse paths the C/C++ parser should use.

    Returns:
        Nothing
    """

    template_dir = os.path.join("assets", "vscode")  # Where to find the VS Code template directory
    template_prefix = "template"  # Filenames beginning with this will have substitution

    # Load defaults from global state...
    # ---
    if part_num == "":
        assert tc.dev is not None
        part_num = tc.dev.partnum

    if board == "":
        board = state.board_name

    if elf_file == "":
        if state.riscv:
            # RISC-V projects will look for ...-combined.elf
            elf_file = "${config:proj_name}-combined.elf"
        else:
            # Default is project name (which defaults to folder name in template)
            elf_file = "${config:proj_name}.elf"

    if defines == "":
        defines = state.defines

    if i_paths == "":
        i_paths = state.eclipse_includes  # TODO: rename state var to 'includes'?

    if v_paths == "":
        pass  # TODO: Support adding browse paths on command line
    # ---

    tmp = []  # Work-horse list, linter be nice
    # Parse defines...
    # ---
    tmp = defines.split(" ")

    if state.defines_arm != "":
        # Split & append Arm defines
        tmp += state.defines_arm.split(" ")

    if state.riscv and state.defines_riscv != "":
        # Split & append risc-v defines
        tmp += state.defines_arm.split(" ")

    tmp = list(map(lambda s: s.strip("-D"), tmp))  # VS Code doesn't want -D
    tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
    defines_parsed = ",\n\t\t\t\t".join(tmp)  # csv, newline, and tab alignment
    # ---

    # Parse include paths...
    tmp = i_paths.split(" ")  # Space-separated
    tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
    i_paths_parsed = ",\n\t\t\t\t".join(tmp)  # csv, newline, and tab alignment

    # Parse browse paths...
    tmp = v_paths.split(" ")  # Space-separated
    tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
    v_paths_parsed = ",\n\t\t\t\t\t".join(tmp)  # csv, newline, and tab alignment

    # Create template...
    for directory, _, files in sorted(os.walk(template_dir)):
        # ^ For each directory in the directory tree rooted at top (including top itself,
        # but excluding '.' and '..'), yields a 3-tuple (dirpath, dirnames, filenames)

        # Get current directory relative to root
        rel_dir = os.path.relpath(directory, template_dir)

        # Figure out whether we're in a subfolder of the template directory,
        # and form output path accordingly.
        out_path = ""
        if rel_dir != '.':
            # We're in a sub-folder.  Replicate this folder in the output directory
            out_path = os.path.join(os.path.join(out_stem, out_branch), rel_dir)
            os.makedirs(out_path, exist_ok=True)
        else:
            # We're in the root template folder.
            out_path = os.path.join(out_stem, out_branch)

        # Any files to copy?
        for file in sorted(files):

            if file.startswith(template_prefix):

                # There is a template file to copy.  Perform string substitution in output file.
                out_loc = os.path.join(out_path, file[len(template_prefix):])
                with open(os.path.join(directory, file), mode='r', encoding='utf-8') as in_file, \
                        open(out_loc, 'w+', encoding='utf-8') as out_file:
                    for line in in_file.readlines():
                        out_file.write(
                            line.replace("##__TARGET_UC__##", part_num.upper()).
                            replace("##__TARGET_LC__##", f"{part_num.lower()}.cfg").
                            replace("##__BOARD__##", board).
                            replace("##__ELF_FILE__##", elf_file).
                            replace("##__OCD_INTERFACE__##", "cmsis-dap.cfg").
                            replace("\"##__ADDITIONAL_INCLUDES__##\"", i_paths_parsed).
                            replace("\"##__DEFINES__##\"", defines_parsed).
                            replace("\"##__ADDITIONAL_SOURCES__##\"", v_paths_parsed)
                        )

                        # Template notes:
                        # The template replacements should only have to touch
                        # 'settings.json'.  The other .vscode files should load from
                        # settings.json unless there's an extreme circumstance requiring a
                        # hard over-write.

                        # - ##__TARGET_UC__## Sets the target micro.  This needs to be uppercase,
                        # since the SDK file-paths use uppercase
                        # - ##__TARGET_LC__## Sets the target OCD config file.  This needs to be
                        # lowercase, since the .cfg files are lowercase in the SDK
                        # - ##__BOARD__## Sets the target board.  This needs to exist in the BSP
                        # directory of the SDK and is case-sensitive.
                        # Ex:  EvKit_V1, FTHR_RevA, etc.
                        # - ##__ELF_FILE__## sets the output filename.  Include the .elf extension.
                        # - ##__OCD_INTERFACE__## Sets the OCD interface file to use.
                        # Defaults to cmsis-dap.cfg
                        # - ##__ADDITIONAL_INCLUDES__## sets additional include paths for C/C++
                        # parser.  This should be comma + new-line separated.
                        # - ##__DEFINES__## sets compiler definitions used by the C/C++ parser.
                        # This should be comma + new-line separated.  This has no effect on the
                        # build system, it's just for intellisense.
                        # - ##__ADDITIONAL_SOURCES__## sets additional browse paths for C/C++
                        # parser (ie. where to find .c files).  Again, no effect on build system,
                        # just for intellisense.  This should be comma + new-line separated.

            else:
                # There is a non-template file to copy
                shutil.copy(os.path.join(directory, file), out_path)
