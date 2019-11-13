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
        input_csv=None,
        input_period=80,
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
            if input_csv is not None:
                runfile.write(f'\n`define CSV_FILE {{`TARGET_DIR,"/{input_csv}"}}\n')
                runfile.write('`include "pcif_defines_af2.sv"\n')
                runfile.write('`define NO_FLASH_MODEL\n\n')
                runfile.write('integer input_file;\n')
                runfile.write('string  null_string;\n')
                runfile.write('logic [7:0] data;\n\n')
                runfile.write('int count;\n\n')
                runfile.write('logic old_pixclk_val;\n')
                runfile.write('logic pixclk_val;\n')
                runfile.write('logic hsync_val;\n')
                runfile.write('logic vsync_val;\n')
                runfile.write('logic [11:0] data_val;\n\n')
                runfile.write('assign `PCIF_PIXCLK  = pixclk_val;\n')
                runfile.write('assign `PCIF_HSYNC   = hsync_val;\n')
                runfile.write('assign `PCIF_VSYNC   = vsync_val;\n')
                runfile.write('assign `PCIF_DATA_11 = data_val[11];\n')
                runfile.write('assign `PCIF_DATA_10 = data_val[10];\n')
                runfile.write('assign `PCIF_DATA_9  = data_val[9];\n')
                runfile.write('assign `PCIF_DATA_8  = data_val[8];\n')
                runfile.write('assign `PCIF_DATA_7  = data_val[7];\n')
                runfile.write('assign `PCIF_DATA_6  = data_val[6];\n')
                runfile.write('assign `PCIF_DATA_5  = data_val[5];\n')
                runfile.write('assign `PCIF_DATA_4  = data_val[4];\n')
                runfile.write('assign `PCIF_DATA_3  = data_val[3];\n')
                runfile.write('assign `PCIF_DATA_2  = data_val[2];\n')
                runfile.write('assign `PCIF_DATA_1  = data_val[1];\n')
                runfile.write('assign `PCIF_DATA_0  = data_val[0];\n\n')
                runfile.write('initial begin\n')
                runfile.write('  old_pixclk_val = 0;\n')
                runfile.write('  pixclk_val = 0;\n')
                runfile.write('  hsync_val = 0;\n')
                runfile.write('  vsync_val = 0;\n')
                runfile.write('  data_val = \'0;\n')
                runfile.write('  input_file = $fopen(`CSV_FILE, "r");\n\n')
                runfile.write('  if (!input_file)\n')
                runfile.write('    begin\n')
                runfile.write('    $display("Error opening %s", `CSV_FILE);\n')
                runfile.write('    $finish;\n')
                runfile.write('  end\n\n')
                runfile.write('  @(posedge sim.trig[0]);\n\n')
                runfile.write('  $fgets(null_string, input_file);\n')
                runfile.write('  $display("Reading camera image from %s", `CSV_FILE);\n\n')
                runfile.write('  while (!$feof(input_file))\n')
                runfile.write('    begin\n')
                runfile.write('      count++;\n')
                runfile.write('      $fscanf(input_file, "%H,%H,%H,%H", '
                              'vsync_val, hsync_val, pixclk_val, data_val);\n')
                runfile.write(f'      #{input_period}ns;\n')
                runfile.write('    end\n\n')
                runfile.write('  $fclose(input_file);\n')
                runfile.write('  $display("Camera image data read");\n\n')
                runfile.write('end\n')


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
