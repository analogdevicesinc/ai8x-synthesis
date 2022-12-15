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
import json
import os
import shutil
import stat
from collections.abc import MutableMapping
from pathlib import Path

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


def parse_json(filename):
    """
    Utility function for parsing values from a json file into an all-caps keyed dictionary.
    This is used by the VS Code project generator to load default function arguments.
    """
    f = open(filename, "r", encoding="UTF-8")
    d = json.load(f)

    # Convert key values to uppercase for easier template parsing
    keys = list(d.keys())  # Keys are changing on the fly, so can't use a view object
    for k in keys:
        d[k.upper()] = d.pop(k)

    return d


# Load default argument values from the default settings.json file
# These settings are suitable for most use-cases.
defaults = parse_json(Path("assets").joinpath("vscode", "defaults", "settings.json"))


def vscode(
    out_root: str,
    out_stem: str,
    target: str,
    board: str,
    overwrite=False,
    program_file: str = defaults["PROGRAM_FILE"],
    symbol_file: str = defaults["SYMBOL_FILE"],
    m4_ocd_interface_file: str = defaults["M4_OCD_INTERFACE_FILE"],
    m4_ocd_target_file: str = defaults["M4_OCD_TARGET_FILE"],
    rv_ocd_interface_file: str = defaults["RV_OCD_INTERFACE_FILE"],
    rv_ocd_target_file: str = defaults["RV_OCD_TARGET_FILE"],
    defines: list = defaults["C_CPP.DEFAULT.DEFINES"],
    i_paths: list = defaults["C_CPP.DEFAULT.INCLUDEPATH"],
    v_paths: list = defaults["C_CPP.DEFAULT.BROWSE.PATH"],
    v_arm_gcc: str = defaults["V_ARM_GCC"],
    v_xpack_gcc: str = defaults["V_XPACK_GCC"],
    ocd_path: str = defaults["OCD_PATH"],
    arm_gcc_path: str = defaults["ARM_GCC_PATH"],
    xpack_gcc_path: str = defaults["XPACK_GCC_PATH"],
    make_path: str = defaults["MAKE_PATH"],
):
    """
    Generates Visual Studio Code project files from the VSCode-Maxim project.
    """

    out_path = Path(out_root).joinpath(out_stem)

    template_dir = Path("assets").joinpath("vscode", "template").resolve()
    # Where to find the VS Code template directory relative to this script

    template_prefix = "template"
    # Filenames beginning with this will have substitution

    if not template_dir.exists():
        raise Exception(f"Failed to find project template folder '{template_dir}'.")

    tmp = []  # Work-horse list, linter be nice
    # Parse compiler definitions...
    if defines != []:
        tmp = defines
        tmp = list(map(lambda s: s.strip("-D"), tmp))  # VS Code doesn't want -D
        tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
        defines_parsed = ",\n        ".join(tmp)  # csv, newline, and tab (w/ spaces) alignment
    else:
        defines_parsed = ""

    # Parse include paths...
    tmp = i_paths
    tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
    i_paths_parsed = ",\n        ".join(tmp).replace(target, "${config:target}").replace("\\", "/")

    # Parse browse paths...
    tmp = v_paths
    tmp = list(map(lambda s: f"\"{s}\"", tmp))  # Surround with quotes
    v_paths_parsed = ",\n        ".join(tmp).replace(target, "${config:target}").replace("\\", "/")

    updated = set()
    # Create template...
    for directory, _, files in sorted(os.walk(template_dir)):
        # ^ For each directory in the directory tree rooted at top (including top itself,
        # but excluding '.' and '..'), yields a 3-tuple (dirpath, dirnames, filenames)

        # Get current directory relative to root
        rel_dir = Path(directory).relative_to(Path(template_dir))

        # Figure out whether we're in a subfolder of the template directory,
        # and form output path accordingly.
        if rel_dir != Path('.'):
            # We're in a sub-folder.  Replicate this folder in the output directory
            out_path = Path(out_path).joinpath(rel_dir)
            os.makedirs(out_path, exist_ok=True)
        else:
            # We're in the root template folder, no need to create a directory.
            pass

        # Any files to copy?
        for file in sorted(files):

            if file.startswith(template_prefix):

                # There is a template file to copy.  Perform string substitution in output file.
                out_file = Path(out_path).joinpath(file[len(template_prefix):])  # Remove prefix
                template = Path(directory).joinpath(file)

                content = None
                with open(template, 'r', encoding="UTF-8") as f:
                    content = f.read()
                    content = content.replace("##__TARGET__##", target.upper()). \
                        replace("##__BOARD__##", board). \
                        replace("##__PROGRAM_FILE__##", program_file). \
                        replace("##__SYMBOL_FILE__##", symbol_file). \
                        replace("##__M4_OCD_INTERFACE_FILE__##", m4_ocd_interface_file). \
                        replace("##__M4_OCD_TARGET_FILE__##", m4_ocd_target_file). \
                        replace("##__RV_OCD_INTERFACE_FILE__##", rv_ocd_interface_file). \
                        replace("##__RV_OCD_TARGET_FILE__##", rv_ocd_target_file). \
                        replace("\"##__I_PATHS__##\"", i_paths_parsed). \
                        replace("\"##__DEFINES__##\"", defines_parsed). \
                        replace("\"##__V_PATHS__##\"", v_paths_parsed). \
                        replace("##__V_ARM_GCC__##", v_arm_gcc). \
                        replace("##__V_XPACK_GCC__##", v_xpack_gcc). \
                        replace("##__OCD_PATH__##", ocd_path). \
                        replace("##__ARM_GCC_PATH__##", arm_gcc_path). \
                        replace("##__XPACK_GCC_PATH__##", xpack_gcc_path). \
                        replace("##__MAKE_PATH__##", make_path)

                if (not out_file.exists()) or overwrite:
                    with open(out_file, "w+", encoding="UTF-8") as f:
                        f.write(content)
                    os.chmod(out_file, stat.S_IRWXU | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
                    if out_file not in updated:
                        updated.add(out_file)

                    # print(f"Wrote {os.path.basename(out_loc)}")  # Uncomment to debug

            else:
                # There is a non-template file to copy
                in_file = Path(directory).joinpath(file)
                out_file = Path(out_path).joinpath(file)

                if (not out_file.exists()) or overwrite:
                    shutil.copy(in_file, out_path)
                    os.chmod(out_file, stat.S_IRWXU | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH)
                    if out_file not in updated:
                        updated.add(out_file)
                    # print(f"Wrote {os.path.basename(file)}") # Uncomment to debug

    return updated


class MakefileMapping(MutableMapping):
    """
    This class is a modified dictionary that maps template strings (keys) to
    values while enforcing template pattern matching.  It also handles special
    cases where a value is not a 1:1 match to what should be written to
    the template Makefile.  For example, pre-pending source files with
    'SRCS +=', etc. is done "on the fly" through this mapping object.

    The key:value rules are as follows:
        * The key is lowercase, and any '.' is replaced is '_'. (revelant for JSON parsing)

        * The value is a tuple with 2 items:
            * index 0: The template string
            * index 1: The value to use when replacing the template string

    Ex:  The setter `MakefileMapping["mykey"]="myvalue"` results in a getter
    for `MakefileMapping["mykey"]` that would return `("##__MYKEY__##", "myvalue")`
    """

    def __init__(self, *args, **kwargs):
        self.d = {}
        self.update(dict(*args, **kwargs))

    def _form_key(self, key):
        # Dictionary keys should be lower-case
        # Ex: templatestring
        return str(key).lower().replace('.', '_')

    def _form_tstring(self, key):
        # Template strings should be uppercase
        # Ex: ##__TEMPLATESTRING__##
        return f"##__{self._form_key(key).upper()}__##"

    # Override the dictionary setter to handle special cases.
    def __setitem__(self, key, value) -> None:

        _value = str(value)
        _key = self._form_key(key)

        # Handle special cases/parsing here.
        if _key == "srcs":
            if isinstance(value, (list, tuple)):
                tmp = tuple(map(lambda s: f"SRCS += {s}", value))
                _value = "\n".join(tmp)
            elif isinstance(value, str):
                _value = f"SRCS += {value}\n"

        elif _key == "vpaths":
            if isinstance(value, (list, tuple)):
                tmp = tuple(map(lambda s: f"VPATH += {s}", value))
                _value = "\n".join(tmp)
            elif isinstance(value, str):
                _value = f"VPATH += {value}\n"

        elif _key == "ipaths":
            if isinstance(value, (list, tuple)):
                tmp = tuple(map(lambda s: f"IPATH += {s}", value))
                _value = "\n".join(tmp)
            elif isinstance(value, str):
                _value = f"IPATH += {value}\n"

        elif _key == "defines":
            if isinstance(value, (list, tuple)):
                tmp = tuple(map(lambda s: f"PROJ_CFLAGS += -D{s}", value))
                _value = "\n".join(tmp)
            elif isinstance(value, str):
                _value = f"PROJ_CFLAGS += -D{value}\n"

        elif _key == "depth":
            if isinstance(value, int):
                _value = "../" * value

        self.d[_key] = (self._form_tstring(key), _value)

    def __getitem__(self, key):
        return self.d[self._form_key(key)]

    def __delitem__(self, key) -> None:
        del self.d[self._form_key(key)]

    def __iter__(self):
        return iter(self.d)

    def __len__(self):
        return len(self.d)


def write_mapping(template_file: Path, out_path: Path, mapping: MakefileMapping,
                  overwrite=False, backup=False):
    """
    Apply a MakefileMapping to 'template_file', then write the converted template to 'out_path'.

    Returns True if the file has been written/modified, otherwise returns False.
    """
    template = None
    with open(template_file, "r", encoding="utf-8") as f:
        template = f.read()
        for _k, v in mapping.items():
            template = template.replace(v[0], v[1])
            # ^ Mapping values are a tuple.
            # v[0] = template string, v[1] = value to replace the template string with

    out_file = Path(out_path).joinpath(template_file.name)
    if not out_file.parent.exists():
        out_file.parent.mkdir()

    write = False
    if (not out_file.exists()) or overwrite:
        if backup:
            shutil.copy(out_file, out_file.parent.joinpath(f"{out_file.name}-backup.mk"))
        with open(out_file, "w+", encoding="utf-8") as f:
            f.write(template)
        write = True

    return write


# Default values that will be used in the template if these keyword
# arguments are not passed to 'create_makefile'
defaults = dict(
    vpaths=[".", "src"],
    ipaths=[".", "include"],
    autosearch=1,
    defines=["MXC_ASSERT_ENABLE", "ARM_MATH_CM4"],
    float_abi="softfp",
    depth=3,
    olevel_debug="g",
    olevel_release=2,
    olevel_default=2,
    compiler="GCC",
    linkerfile="$(TARGET_LC).ld",
    sbt=0,
    default_goal="all"
)


def makefile(
    out_root: str,
    out_stem: str,
    target: str,
    board: str,
    overwrite=False,
    backup=False,
    overwrite_projectmk=False,
    **kwargs
):
    """
    Inject a project.mk-based Makefile system into the directory specified by out_root/out_stem.
    """
    out_path = Path(out_root).joinpath(out_stem)

    # Calculate the depth of the project to use for relative
    # search if no MAXIM_PATH is set.  This assumes that
    # out_path is a relative location to the ai8x-synthesis
    # directory (ie. sdk/Examples/MAX78000/CNN)
    depth = len(out_path.parts)
    defaults["depth"] = depth - 1

    for k, v in defaults.items():
        kwargs.setdefault(k, v)

    mapping = MakefileMapping(kwargs)

    mapping["target"] = target
    mapping["target_uc"] = target.upper()
    mapping["target_lc"] = target.lower()
    mapping["board"] = board

    template_makefile = Path("assets").joinpath("makefile", "Makefile").resolve()
    template_projectmk = Path("assets").joinpath("makefile", "project.mk").resolve()

    wrote_makefile = write_mapping(
            template_makefile,
            out_path,
            mapping,
            overwrite=overwrite,
            backup=backup
        )

    wrote_projectmk = write_mapping(
            template_projectmk,
            out_path,
            mapping,
            overwrite=overwrite_projectmk,
            backup=backup
        )

    # Return value indicates whether or not any files have been changed.
    return wrote_makefile or wrote_projectmk
