###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
RTL simulation support routines
"""
import os


GLOBAL_TIME_OFFSET = 3


def create_runtest_sv(
        block_mode,
        base_directory,
        test_name,
        runtest_filename,
        input_filename,
        c_filename,
        timeout,
        riscv=False,
):
    """
    For for test `test_name`, create the runtest.sv file named `runtest_filename`, in the
    directory `base_directory`. The file contains the timeout value `timeout`.
    If in `block_mode`, it will refer to the `input_filename`.
    """

    with open(os.path.join(base_directory, test_name, runtest_filename), mode='w') as runfile:
        if block_mode:
            runfile.write('// Check default register values.\n')
            runfile.write('// Write all registers.\n')
            runfile.write('// Make sure only writable bits will change.\n')
            runfile.write('int     inp1;\n')
            runfile.write('string  fn;\n\n')
            if timeout:
                runfile.write(f'defparam REPEAT_TIMEOUT = {timeout};\n\n')
            runfile.write('initial begin\n')
            runfile.write('  //----------------------------------------------------------------\n')
            runfile.write('  // Initialize the CNN\n')
            runfile.write('  //----------------------------------------------------------------\n')
            runfile.write('  #200000;\n')
            runfile.write(f'  fn = {{`TARGET_DIR,"/{input_filename}.mem"}};\n')
            runfile.write('  inp1=$fopen(fn, "r");\n')
            runfile.write('  if (inp1 == 0) begin\n')
            runfile.write('    $display("ERROR : CAN NOT OPEN THE FILE");\n')
            runfile.write('  end\n')
            runfile.write('  else begin\n')
            runfile.write('    write_cnn(inp1);\n')
            runfile.write('    $fclose(inp1);\n')
            runfile.write('  end\n')
            runfile.write('end\n\n')
            runfile.write('initial begin\n')
            runfile.write('  #1;\n')
            runfile.write('  error_count = 0;\n')
            runfile.write('  @(posedge rstn);\n')
            runfile.write('  #5000;     // for invalidate done\n')
            runfile.write('  -> StartTest;\n')
            runfile.write('end\n')
        else:
            runfile.write(f'// {runtest_filename}\n')
            runfile.write(f'`define ARM_PROG_SOURCE {c_filename}.c\n')
            if riscv:
                runfile.write(f'`define RISCV_PROG_SOURCE {c_filename}_riscv.c\n')
                runfile.write('`define MULTI_CPU_SETUP\n')
            if timeout:
                runfile.write(f'defparam REPEAT_TIMEOUT = {timeout};\n\n')
            if riscv:
                runfile.write(
                    'event ev_load_riscv_flash_image;\n'
                    'initial begin\n'
                    '    @(por_done);\n'
                    '    $display("Loading RISC-V FLASH main array image %s at %0t", '
                    'FLASH_IMAGE, $time);\n'
                    '    $readmemh({`TARGET_DIR,"/RISCV_PROG_flash.prog"}, '
                    '`FLASH.main_mem, 32\'h0000, 32\'h83FF);\n'
                    '    ->ev_load_riscv_flash_image;\n'
                    '    #1;\n'
                    '    multi_cpu_en = 1\'b0;\n'
                    'end\n'
                )


def append_regression(
        top_level,
        test_name,
        queue_name,
        autogen_dir,
):
    """
    Append test `test_name` to the regression list in directory `autogen_dir` with
    queue `queue_name`.
    `top_level` indicates whether to insert another directory level into the output
    path..
    """

    # Append to regression list?
    if not top_level:
        testname = f'tests/{test_name}/run_test:{queue_name}'
    else:
        testname = f'tests/{top_level}/{test_name}/run_test:{queue_name}'
    found = False
    try:
        with open(os.path.join(autogen_dir, 'autogen_list'), mode='r') as listfile:
            for line in listfile:
                if testname in line:
                    found = True
                    break
    except FileNotFoundError:
        pass
    if not found:
        with open(os.path.join(autogen_dir, 'autogen_list'), mode='a') as listfile:
            listfile.write(f'{testname}\n')
