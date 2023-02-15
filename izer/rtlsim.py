###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
RTL simulation support routines
"""
import os
from typing import List, Optional, Tuple

from . import apbaccess, state, stats
from . import tornadocnn as tc

MIN_SIMULATION_TIMEOUT = 10  # ms

NS_PER_CNN_CYCLE = 25  # 20 ns * 1.25

GLOBAL_TIME_OFFSET = 3
ARM_APB_ACCESS = 500  # ns
RISCV_APB_ACCESS = 1000  # ns
RISCV_FASTFIFO_ACCESS = 100  # ns
GENERAL_OFFSET = 3000000  # ns
ZERO_SRAM_OFFSET = 16000000  # ns
NS_TO_MS = 1000000


def calculate_timeout(
        total: int,
        apb: apbaccess.APB,
        input_dim: Optional[List[Tuple[int, int]]],
        in_expand:  Optional[List[int]],
) -> int:
    """
    Estimate timeout based on CNN cycles, and memory/register read/write operations
    """
    assert input_dim is not None
    assert in_expand is not None
    # Times in ns
    apb_access_time = RISCV_APB_ACCESS if state.riscv else ARM_APB_ACCESS
    if total > 0:
        # Use cycle count if we have it
        timeout = total * NS_PER_CNN_CYCLE
        reads, writes, fifo_reads, fifo_writes, fastfifo_reads, fastfifo_writes = \
            apb.get_access_count()
        timeout += (reads + writes) * apb_access_time
        write_ratio = 1.
        if fifo_writes > 0 or fastfifo_writes > 0:
            for ll, delta1 in enumerate(state.delta1):
                if ll > state.start_layer:
                    # Discount FIFO reads by delta ratios -- some of the FIFO writes occur in
                    # parallel to processing that's already accounted for (does not apply to
                    # first layer, since everything needs to be popped)
                    write_count = input_dim[ll][0] * input_dim[ll][1] * in_expand[ll]
                    if delta1 != 0:
                        ratio = 1 / delta1
                    if state.delta2[ll] != 0:
                        ratio *= state.delta2[ll] / input_dim[ll][0]
                    if state.stream_start[ll] != 0:
                        start = state.stream_start[ll]
                        ratio = (start + ratio * (write_count - start)) / write_count
                    write_ratio *= ratio

            timeout += (fifo_reads + fifo_writes) * int(apb_access_time * write_ratio)
            timeout += (fastfifo_reads + fastfifo_writes) \
                * int(RISCV_FASTFIFO_ACCESS * write_ratio)
        # General overhead
        timeout += GENERAL_OFFSET
    else:
        # If no timeout specified, and no cycle count available,
        # calculate timeout based on reads/writes
        timeout = 10 * NS_TO_MS * (apb.get_time() + GLOBAL_TIME_OFFSET)

    if state.zero_sram:
        timeout += ZERO_SRAM_OFFSET
    # Add weight and bias write times
    if not state.rtl_preload_weights:
        timeout += stats.resourcedict['kmem_used'] * apb_access_time // 4
    if state.verify_kernels:
        access_factor = 6 if state.mexpress else 4  # Reading back more than we wrote?
        timeout += stats.resourcedict['kmem_used'] * access_factor * apb_access_time // 4

    # Convert to ms
    timeout //= NS_TO_MS
    timeout = max(MIN_SIMULATION_TIMEOUT, timeout)

    state.timeout = timeout
    return timeout


def create_runtest_sv(
        test_name: str,
        timeout: int,
        groups_used: Optional[List[int]] = None,
        cnn_cycles: int = 0,
        apb: Optional[apbaccess.APB] = None,
        input_dim: Optional[List[Tuple[int, int]]] = None,
        in_expand: Optional[List[int]] = None,
):
    """
    For for test `test_name`, create the runtest.sv file named `runtest_filename`, in the
    directory `base_directory`. The file contains the timeout value `timeout`.
    If in `block_mode`, it will refer to the `input_filename`.
    """
    assert tc.dev is not None
    assert apb is not None

    if not timeout:
        timeout = calculate_timeout(cnn_cycles, apb, input_dim, in_expand)

    # Cache for faster access
    result_output = state.result_output
    riscv = state.riscv

    with open(
        os.path.join(state.base_directory, test_name, state.runtest_filename),
        mode='w',
        encoding='utf-8',
    ) as runfile:
        if state.block_mode:
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
            runfile.write(f'  fn = {{`TARGET_DIR,"/{state.input_filename}.mem"}};\n')
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
            runfile.write(f'// {state.runtest_filename}\n')
            runfile.write(f'`define ARM_PROG_SOURCE {state.c_filename}.c\n')
            if riscv:
                runfile.write(f'`define RISCV_PROG_SOURCE {state.c_filename}_riscv.c\n')
                runfile.write('`define MULTI_CPU_SETUP\n')
            if timeout:
                runfile.write(f'// Timeout: {timeout} ms\n')
                runfile.write(f'defparam REPEAT_TIMEOUT = {timeout/10.0:0.1f};\n')
            if riscv:
                runfile.write(
                    '\nevent ev_load_riscv_flash_image;\n'
                    'initial begin\n'
                    '    @(por_done);\n'
                    '    $display("Loading RISC-V FLASH main array image %s at %0t", '
                    'FLASH_IMAGE, $time);\n'
                    '    $readmemh({`TARGET_DIR,"/RISCV_PROG_flash.prog"}, '
                    f'`FLASH.main_mem, 32\'h0000, 32\'h{tc.dev.FLASH_SIZE + 0x3FF:04X});\n'
                    '    ->ev_load_riscv_flash_image;\n'
                    '    #1;\n'
                    'end\n'
                )
            if tc.dev.MODERN_SIM:
                runfile.write(
                    '\n`ifdef gate_sims\n'
                    '  `define CNN_ENA  `DIGITAL_TOP.xuut1.x16proc_0__xproc_xuut.xcnn_fsm2.cnnena'
                    '\n  `define CNN_CLK  `DIGITAL_TOP.xuut1.x16proc_0__xproc_xuut.clk\n'
                    '`else\n'
                    '  `define CNN_ENA  `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.cnnena\n'
                    '  `define CNN_CLK  `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.clk\n')
                if state.log_intermediate:
                    runfile.write(
                        '  `define CNN_SWR  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.sramwr\n'
                        '  `define CNN_WAD  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.cnnadr0\n'
                        '  `define CNN_WDT  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.cnnout0\n'
                        '  `define CNN_SRD  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.req_sram_rd\n'
                        '  `define CNN_RAD  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.rptr\n'
                        '  `define CNN_WRT  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.sram_wrt\n'
                        '  `define CNN_ADR  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.ram_adr\n'
                        '  `define CNN_DAT  `DIGITAL_TOP.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.ram_data\n'
                        '  `define CNN_BYP  `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.bypass\n'
                        '  `define CNN_STRM `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.strmact\n'
                        '  `define CNN_LYR  `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.lyrcnt_e\n'
                        '  `define CNN_SLYR `DIGITAL_TOP.xuut1.x16proc[0].xproc.xuut.lyrsel\n'
                        '  `define DBG_FILE {`TARGET_DIR,"/debug/cnn_debug_data.csv"}\n'
                    )
                runfile.write(
                    '`endif\n\n'
                )
            else:
                runfile.write(
                    '\n`define CNN_ENA  tb.xchip.xuut1.x16proc[0].xproc.xuut.cnnena\n'
                    '`define CNN_CLK  tb.xchip.xuut1.x16proc[0].xproc.xuut.clk\n'
                )
                if state.log_intermediate:
                    runfile.write(
                        '  `define CNN_SWR  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.sramwr\n'
                        '  `define CNN_WAD  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.cnnadr0\n'
                        '  `define CNN_WDT  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.cnnout0\n'
                        '  `define CNN_SRD  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.req_sram_rd\n'
                        '  `define CNN_RAD  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.rptr\n'
                        '  `define CNN_WRT  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.sram_wrt\n'
                        '  `define CNN_ADR  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.ram_adr\n'
                        '  `define CNN_DAT  tb.xchip.xuut1.x16proc[0].xproc.'
                        'xuut.xsram[0].xcnn_ram.ram_data\n'
                        '  `define CNN_BYP  tb.xchip.xuut1.x16proc[0].xproc.xuut.bypass\n'
                        '  `define CNN_STRM tb.xchip.xuut1.x16proc[0].xproc.xuut.strmact\n'
                        '  `define CNN_LYR  tb.xchip.xuut1.x16proc[0].xproc.xuut.lyrcnt_e\n'
                        '  `define CNN_SLYR tb.xchip.xuut1.x16proc[0].xproc.xuut.lyrsel\n'
                        '  `define DBG_FILE {`TARGET_DIR,"/debug/cnn_debug_output.csv"}\n'
                    )
                runfile.write(
                    '\n'
                )
            runfile.write(
                'real  start_time;\n'
                'real  end_time;\n'
                'real  clk1_time;\n'
                'real  clk2_time;\n'
                'logic start_ena;\n'
                'logic clkena1;\n'
                'logic clkena2;\n'
                'logic clkena3;\n'
            )
            if result_output:
                runfile.write('int   chk_stat;\n')
            runfile.write(
                'logic chk_clk;\n'
            )
            if state.log_intermediate:
                runfile.write(
                    'int   i;\n\n'
                    'logic [3:0] lcnt;\n'
                    'logic [3:0] lcnt_wsel;\n'
                    'logic [3:0] lcnt_rsel;\n'
                    'logic [3:0] lcnt_dly [9:0];\n\n'
                    'int   cycle_cnt;\n'
                    'int   fd;\n'
                )
            runfile.write(
                '\ninitial begin\n'
            )
            if result_output:
                runfile.write(
                    '   open_files;\n'
                    '   chk_stat   = 0;\n'
                )
            runfile.write(
                '   start_time = 0;\n'
                '   end_time   = 0;\n'
                '   clk1_time  = 0;\n'
                '   clk2_time  = 0;\n'
                '   start_ena  = 0;\n'
                '   clkena1    = 0;\n'
                '   clkena2    = 0;\n'
                '   clkena3    = 0;\n'
            )
            if state.log_intermediate:
                runfile.write(
                    '\n   for(i=0;i<10;i++)\n'
                    '     lcnt_dly[i] = 0;\n\n'
                    '   lcnt       = 0;\n'
                    '   lcnt_wsel  = 0;\n'
                    '   lcnt_rsel  = 0;\n'
                    '   cycle_cnt  = 0;\n'
                    '   fd = $fopen(`DBG_FILE,"w");\n'
                )
            runfile.write(
                'end\n\n'
                'always @(posedge `CNN_ENA) begin\n'
                '  if (!start_ena) begin\n'
            )
            if state.rtl_preload or state.rtl_preload_weights:
                for i in groups_used:
                    runfile.write(f'    load_cnn_mems_{i};\n')
            runfile.write(
                '    start_time  = $realtime;\n'
                '    start_ena   = 1;\n'
                '    $display("CNN enabled");\n'
                '  end\n'
                'end\n\n'
                'assign #10 chk_clk = `CNN_ENA;\n\n'
                'always @(negedge chk_clk) begin\n'
                '  if (start_ena) begin\n'
                '    end_time  = $realtime;\n'
                '    clkena1   = 1;\n'
            )
            if result_output:
                for i in groups_used:
                    runfile.write(f'    dump_cnn_mems_{i};\n')
                runfile.write(
                    '    close_files;\n'
                )
            runfile.write(
                '  end\n'
                'end\n\n'
                'always @(posedge `CNN_CLK) begin\n'
                '  if (clkena1) begin\n'
                '    clk1_time = $realtime;\n'
                '    clkena1   = 0;\n'
                '    clkena2   = 1;\n'
                '  end else if (clkena2) begin\n'
                '    clk2_time = $realtime;\n'
                '    clkena2   = 0;\n'
                '    $display("CNN Cycles = %.0f", '
                '$ceil((end_time - start_time)/(clk2_time - clk1_time)) - 1);\n'
                '    clkena3   = 1;\n'
                '  end\n'
                'end\n'
            )
            if result_output:
                runfile.write(
                    '\nalways @(posedge clkena3) begin\n'
                    '    chk_stat = $system({`TARGET_DIR,"/verify-output.py ",`TARGET_DIR});\n'
                    '    if (chk_stat != 0)\n'
                    '      error_count++;\n'
                    'end\n'
                )
            if state.log_intermediate:
                runfile.write(
                    '\nalways @* begin\n'
                    '  assign lcnt      = (`CNN_STRM == 1\'b1)? `CNN_SLYR : `CNN_LYR;\n'
                    '  assign lcnt_wsel = (`CNN_BYP  == 1\'b1)?  lcnt     :  lcnt_dly[9];\n'
                    '  assign lcnt_rsel = (`CNN_BYP  == 1\'b1)?  lcnt     :  lcnt_dly[1];\n'
                    'end\n\n'
                    'always @(posedge `CNN_CLK) begin\n'
                    '  if (`CNN_WRT) begin\n'
                    '    $fdisplay(fd,"w,%0h,%0h,%0d,%0d",`CNN_ADR,`CNN_DAT,-1,cycle_cnt);\n'
                    '  end\n'
                    '  if (start_ena) begin\n'
                    '    lcnt_dly <= {lcnt_dly[8:0],lcnt};\n\n'
                    '    if (`CNN_SWR) begin\n'
                    '      $fdisplay(fd,"w,%0h,%0h,%0d,%0d",'
                    '`CNN_WAD,`CNN_WDT,lcnt_wsel,cycle_cnt);\n'
                    '    end\n'
                    '    if (`CNN_SRD) begin\n'
                    '      $fdisplay(fd,"r,%0h,0,%0d,%0d",`CNN_RAD,lcnt_rsel,cycle_cnt);\n'
                    '    end\n'
                    '    cycle_cnt++;\n'
                    '  end\n'
                    'end\n'
                )

            if state.input_csv is not None:
                runfile.write(f'\n`define CSV_FILE {{`TARGET_DIR,"/{state.input_csv}"}}\n')
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
                runfile.write('assign `PCIF_DATA_0  = data_val[0];\n')

                if state.input_sync:
                    runfile.write(f'\nparameter pclk_ai_per_pix = {state.input_pix_clk};\n\n')
                    if tc.dev.MODERN_SIM:
                        runfile.write('`define PCLK_AI  `DIGITAL_TOP.pclk_ai\n')
                    else:
                        runfile.write('logic        pclk_ai;\n')
                    runfile.write('logic        clk_pix;\n')
                    runfile.write('logic        start_io;\n')
                    runfile.write('logic        end_of_file;\n')
                    runfile.write('integer      clk_pix_i;\n\n')
                    if not tc.dev.MODERN_SIM:
                        runfile.write('assign pclk_ai = tb.xchip.pclk_ai;\n\n')
                        runfile.write('always @(posedge pclk_ai) begin\n')
                    else:
                        runfile.write('always @(posedge `PCLK_AI) begin\n')
                    runfile.write('  if (clk_pix_i == pclk_ai_per_pix)\n')
                    runfile.write('    clk_pix_i = 0;\n')
                    runfile.write('  else\n')
                    runfile.write('    clk_pix_i = clk_pix_i + 1;\n')
                    runfile.write('end\n\n')
                    runfile.write('assign clk_pix = clk_pix_i == pclk_ai_per_pix;\n\n')
                    runfile.write('always @(posedge clk_pix) begin\n')
                    runfile.write('  if (start_io) begin\n')
                    runfile.write('    if (!$feof(input_file)) begin\n')
                    runfile.write('      $fscanf(input_file, "%H,%H,%H,%H", '
                                  'vsync_val, hsync_val, pixclk_val, data_val);\n')
                    runfile.write('    end else begin\n')
                    runfile.write('      end_of_file = 1;\n')
                    runfile.write('      start_io    = 0;\n')
                    runfile.write('    end\n')
                    runfile.write('  end\n')
                    runfile.write('end\n')

                runfile.write('\ninitial begin\n')
                runfile.write('  old_pixclk_val = 0;\n')
                runfile.write('  pixclk_val = 0;\n')
                runfile.write('  hsync_val = 0;\n')
                runfile.write('  vsync_val = 0;\n')
                runfile.write('  data_val = \'0;\n')
                runfile.write('  input_file = $fopen(`CSV_FILE, "r");\n')

                if state.input_sync:
                    runfile.write('\n  start_io    = 0;\n')
                    runfile.write('  end_of_file = 0;\n')
                    runfile.write('  clk_pix_i   = 0;\n')

                runfile.write('\n  if (!input_file)\n')
                runfile.write('    begin\n')
                runfile.write('    $display("Error opening %s", `CSV_FILE);\n')
                runfile.write('    $finish;\n')
                runfile.write('  end\n\n')
                runfile.write('  @(posedge sim.trig[0]);\n\n')
                runfile.write('  $fgets(null_string, input_file);\n')
                runfile.write('  $display("Reading camera image from %s", `CSV_FILE);\n')

                if state.input_sync:
                    runfile.write('\n  start_io = 1;\n')
                    runfile.write('  while (!end_of_file) begin\n')
                    runfile.write('    count++;\n')
                    runfile.write('    #200ns;\n')
                    runfile.write('  end\n')
                else:
                    runfile.write('\n  while (!$feof(input_file))\n')
                    runfile.write('    begin\n')
                    runfile.write('      count++;\n')
                    runfile.write('      $fscanf(input_file, "%H,%H,%H,%H", '
                                  'vsync_val, hsync_val, pixclk_val, data_val);\n')
                    runfile.write(f'      #{state.input_csv_period}ns;\n')
                    runfile.write('    end\n')

                runfile.write('\n  $fclose(input_file);\n')
                runfile.write('  $display("Camera image data read");\n\n')
                runfile.write('end\n')

    if state.log_intermediate:
        os.makedirs(os.path.join(state.base_directory, test_name, 'debug'), exist_ok=True)


def append_regression(
        top_level,
        test_name,
        queue_name,
        autogen_dir,
        autogen_list='autogen_list',
):
    """
    Append test `test_name` to the regression list in directory `autogen_dir` with
    queue `queue_name`.
    `top_level` indicates whether to insert another directory level into the output
    path..
    """
    if queue_name is None:
        queue_name = 'long' if state.timeout > 50 else 'short'

    # Append to regression list?
    if not top_level:
        testname = f'tests/{test_name}/run_test:{queue_name}'
    else:
        testname = f'tests/{top_level}/{test_name}/run_test:{queue_name}'
    found = False
    try:
        with open(os.path.join(autogen_dir, autogen_list), mode='r', encoding='utf-8') as listfile:
            for line in listfile:
                if testname in line:
                    found = True
                    break
    except FileNotFoundError:
        pass
    if not found:
        with open(os.path.join(autogen_dir, autogen_list), mode='a', encoding='utf-8') as listfile:
            listfile.write(f'{testname}\n')


def write_latency(test_name: str, total: int, per_layer: List[int]) -> None:
    """
    Create a file called latency.txt in the sim directory.
    """
    with open(
        os.path.join(state.base_directory, test_name, 'latency.txt'),
        mode='w',
        encoding='utf-8',
    ) as f:
        f.write(f'{total}\n')
        for i in per_layer:
            f.write(f'{i}\n')
