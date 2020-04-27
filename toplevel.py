###################################################################################################
# Copyright (C) 2018-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Toplevel C file structure generation
"""
import rv
import tornadocnn as tc
from armx4weights import convert_to_x4_q7_weights
from eprint import eprint

COPYRIGHT = \
    '// ---------------------------------------------------------------------------\n' \
    '// Copyright (C) 2019-2020 Maxim Integrated Products, Inc.\n' \
    '// All rights reserved. Product of the U.S.A.\n' \
    '// ---------------------------------------------------------------------------\n\n'


def copyright_header(
        memfile,
):
    """
    Write the copyright header to .c file handle `memfile`.
    """
    memfile.write(COPYRIGHT)


def header(
        memfile,
        apb_base,
        embedded_code=False,
        cmsis_nn=False,
        compact_weights=False,
        compact_data=False,
        weight_filename='weights.h',
        sample_filename='sampledata.h',
        master=False,
        verify_kernels=False,
        riscv=False,
        camera=False,
):
    """
    Write include files and forward definitions to .c file handle `memfile`.
    The APB base address is passed in `apb_base`.
    """
    memfile.write('#include <stdlib.h>\n')
    memfile.write('#include <stdint.h>\n')
    if embedded_code or verify_kernels:
        memfile.write('#include <string.h>\n')
    if embedded_code:
        memfile.write('#include <stdio.h>\n')
    if not cmsis_nn:
        memfile.write('#include "global_functions.h" // For RTL Simulation\n')
        if embedded_code:
            memfile.write('#include "tmr_utils.h"\n')
    if camera:
        memfile.write('#include "pcif_defines_af2.h"\n')
        memfile.write('#define NUM_DATA_WORDS 4\n')
        memfile.write('#include "pcif.c"\n')
    if embedded_code:
        memfile.write('#include "tornadocnn.h"\n')
    if embedded_code or compact_weights:
        memfile.write(f'#include "{weight_filename}"\n')
    if embedded_code or compact_data:
        memfile.write(f'#include "{sample_filename}"\n')
    memfile.write('\n')

    if not cmsis_nn and (riscv is None or riscv):
        if embedded_code:
            memfile.write('uint32_t cnn_time; // Stopwatch\n\n')

            memfile.write('void fail(void)\n{\n')
            memfile.write('  printf("\\n*** FAIL ***\\n\\n");\n')
            memfile.write('  while (1);\n}\n\n')

        memfile.write('void cnn_wait(void)\n{\n')
        memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + tc.dev.C_CNN_BASE:08x}) '
                      '& (1<<12)) != 1<<12) ;\n')
        if embedded_code:
            memfile.write('  CNN_COMPLETE; // Signal that processing is complete\n')
            memfile.write('  cnn_time = TMR_SW_Stop(MXC_TMR0);\n')
        memfile.write('}\n\n')

    if master is not False:
        addr = apb_base + tc.dev.C_CNN_BASE + tc.dev.C_GROUP_OFFS * master

        memfile.write('void cnn_restart(void)\n{\n')
        memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) |= 1; '
                      f'// Re-enable group {master}\n')
        memfile.write('}\n\n')

        memfile.write('void cnn_stop(void)\n{\n')
        memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) &= ~1; '
                      f'// Disable group {master}\n')
        memfile.write('}\n\n')


def load_header(
        memfile,
        riscv_flash=False,
):
    """
    Write the header for the CNN configuration loader function to `memfile`.
    """
    if riscv_flash:
        memfile.write(rv.RISCV_FLASH)
    memfile.write('int cnn_load(void)\n{\n')


def load_footer(
        memfile,
        embedded_code=False,  # pylint: disable=unused-argument
):
    """
    Write the footer for the CNN configuration loader function to `memfile`.
    """
    memfile.write('\n  return 1;\n}\n\n')


def main(
        memfile,
        classification_layer=False,
        unload=False,
        softmax=False,
        embedded_code=False,
        oneshot=0,
        stopstart=False,
        riscv=None,
        riscv_exclusive=False,
        riscv_flash=False,  # pylint: disable=unused-argument
        riscv_cache=False,
        camera=False,
        camera_format=None,
        device=84,
        channels=None,
        sleep=False,
        output_width=8,
        num_classes=None,
        clock_trim=None,
        embedded_arm=False,
):
    """
    Write the main function (including an optional call to the fully connected layer if
    `classification_layer` is `True`) to `memfile`.
    """
    if softmax and output_width == 8:
        eprint('--softmax should only be used with `output_width: 32`', error=False)

    if unload:
        memfile.write(f'#define NUM_OUTPUTS {num_classes}\n')
        memfile.write(f'static uint{output_width}_t ml_data[NUM_OUTPUTS];\n\n')

    memfile.write('int main(void)\n{\n')
    if clock_trim is not None and not riscv:
        memfile.write('  uint32_t trim;\n')
    if embedded_code and (classification_layer or softmax) or oneshot > 0:
        memfile.write('  int i;\n\n')

    if riscv is None or not riscv:
        memfile.write('  icache_enable();\n\n')
        if embedded_code or embedded_arm:
            if device == 84:
                memfile.write('  SYS_ClockEnable(SYS_PERIPH_CLOCK_AI);\n')
            else:
                if clock_trim is not None:
                    memfile.write('  // Manual clock trim override:\n')
                    memfile.write('  *((volatile uint32_t *) 0x40000c00) = 1; '
                                  '// Set TME\n')
                    if clock_trim[0] or clock_trim[1]:
                        memfile.write('  trim = *((volatile uint32_t *) 0x40005420);\n')
                        if clock_trim[0]:
                            memfile.write('  trim &= ~0x1ff;\n'
                                          f'  trim |= 0x{clock_trim[0]:x}; '
                                          '// HIRC8M (7.3728 MHz) trim\n')
                        if clock_trim[1]:
                            memfile.write('  trim &= ~(0x1ff << 22);\n'
                                          f'  trim |= 0x{clock_trim[1]:x} << 22; '
                                          '// HIRC (60 MHz) trim\n')
                        memfile.write('  *((volatile uint32_t *) 0x40005420) = trim;\n')
                    if clock_trim[2]:
                        memfile.write('  trim = *((volatile uint32_t *) 0x40005440) & '
                                      '~(0x1ff << 15);\n')
                        memfile.write('  *((volatile uint32_t *) 0x40005440) = '
                                      'trim | (0xff << 15); // HILIM\n')
                        memfile.write('  *((volatile uint32_t *) 0x40006c04) = '
                                      f'0x{clock_trim[2]:x}; // HIRC96M (96 MHz) trim\n')
                    memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0; '
                                  '// Clear TME\n\n')

                memfile.write('  MXC_GCR->clkcn |= MXC_F_GCR_CLKCN_HIRC96M_EN; '
                              '// Enable 96 MHz\n')
                memfile.write('  while ((MXC_GCR->clkcn & MXC_F_GCR_CLKCN_HIRC96M_RDY) == 0) ; '
                              '// Wait for 96 MHz\n')
                memfile.write('  MXC_GCR->clkcn |= MXC_S_GCR_CLKCN_CLKSEL_HIRC96; '
                              '// Select 96 MHz\n')

                memfile.write('\n  // Reset all domains, restore power to CNN\n')
                memfile.write('  MXC_BBFC->reg3 = 0xf; // Reset\n')
                memfile.write('  MXC_BBFC->reg1 = 0xf; // Mask\n')
                memfile.write('  MXC_BBFC->reg0 = 0xf; // Power\n')
                memfile.write('  MXC_BBFC->reg2 = 0x0; // Iso\n')
                memfile.write('  MXC_BBFC->reg3 = 0x0; // Reset\n\n')

                memfile.write('  MXC_GCR->pckdiv = 0x00010000; // AI clock: 96 MHz div 2\n')
                memfile.write('  MXC_GCR->perckcn &= ~0x2000000; // Enable AI clock\n')

                memfile.write('\n  printf("\\n*** CNN Test ***\\n");\n')
        else:
            if device == 84:
                memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable AI clock\n')
            else:
                memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000001; // Set TME\n')
                memfile.write('  *((volatile uint32_t *) 0x40006c04) = 0x000001a0; // 96M trim\n')
                memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000000; '
                              '// Clear TME\n\n')
                memfile.write('  MXC_GCR->clkcn |= MXC_F_GCR_CLKCN_HIRC96M_EN; // Enable 96M\n')
                memfile.write('  while ((MXC_GCR->clkcn & MXC_F_GCR_CLKCN_HIRC96M_RDY) == 0) ; '
                              '// Wait for 96M\n')
                memfile.write('  MXC_GCR->clkcn |= MXC_S_GCR_CLKCN_CLKSEL_HIRC96; // Select 96M\n')

                memfile.write('\n  // Reset all domains, restore power to CNN\n')
                memfile.write('  MXC_BBFC->reg3 = 0xf; // Reset\n')
                memfile.write('  MXC_BBFC->reg1 = 0xf; // Mask\n')
                memfile.write('  MXC_BBFC->reg0 = 0xf; // Power\n')
                memfile.write('  MXC_BBFC->reg2 = 0x0; // Iso\n')
                memfile.write('  MXC_BBFC->reg3 = 0x0; // Reset\n\n')

                memfile.write('  MXC_GCR->pckdiv = 0x00010000; // AI clock 96M div 2\n')
                memfile.write('  MXC_GCR->perckcn &= ~0x2000000; // Enable AI clock\n')
        if riscv is not None:
            if riscv_cache:
                if embedded_code:
                    memfile.write('  MXC_NBBFC->reg4 = (uint32_t) &_rvflash; '
                                  '// Set RISC-V boot address\n')
                else:
                    memfile.write(f'  MXC_NBBFC->reg4 = 0x{rv.RISCV_CODE_ORIGIN:08x}; '
                                  '// Set RISC-V boot address\n')
            if riscv_exclusive:
                memfile.write('  *((volatile uint32_t *) 0x40000814) |= 0x00000001; '
                              '// Exclusive SRAM access for RISC-V (MXC_NBBFC->reg5)\n')
            memfile.write('  MXC_GCR->perckcn1 &= ~MXC_F_GCR_PERCKCN1_CPU1; '
                          '// Enable RISC-V clock\n')
        memfile.write('\n')
    elif riscv and riscv_cache:
        memfile.write('  icache1_enable();\n')
        memfile.write('  invalidate_icache1();\n\n')

    if camera:
        memfile.write('  enable_pcif_clock(); // Enable camera clock\n')
        memfile.write('  set_pcif_gpio_altf();\n\n')
        if camera_format == 555:
            mode = '10'
            comment = '555'
        elif camera_format == 565:
            mode = '12'
            comment = '565'
        else:
            mode = '8'  # Default
            comment = '888'
        memfile.write(f'  // Enable {comment} format single image in external timing mode\n')
        memfile.write('  MXC_CAMERAIF0->ctrl = MXC_S_CAMERAIF_CTRL_READ_MODE_SINGLE_IMG +\n'
                      f'                        MXC_S_CAMERAIF_CTRL_DATA_WIDTH_{mode}BIT +\n'
                      '                        MXC_S_CAMERAIF_CTRL_DS_TIMING_EN_DIS +\n'
                      '                        MXC_S_CAMERAIF_CTRL_PCIF_SYS_EN_EN')
        if channels == 3:
            memfile.write(' +\n                        (1<<30);\n\n')
        else:
            memfile.write(';\n\n')

    if riscv is None or riscv:
        memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n')
        if embedded_code:
            memfile.write('  TMR_SW_Start(MXC_TMR0, NULL);\n')
        if stopstart:
            memfile.write('\n  cnn_stop();\n')
            memfile.write('  cnn_restart();\n\n')

        memfile.write('  cnn_wait();\n\n')
        if oneshot > 0:
            memfile.write(f'  for (i = 0; i < {oneshot}; i++) {{\n')
            memfile.write('    cnn_restart();\n')
            memfile.write('    cnn_wait();\n')
            memfile.write('  }\n\n')
        memfile.write('  if (!cnn_check()) fail();\n')
        if classification_layer or softmax:
            memfile.write(f'  if (!{"softmax" if softmax else "fc"}_layer()) fail();\n')
        elif unload:
            memfile.write('  cnn_unload(ml_data);\n')
        if classification_layer:
            memfile.write('  if (!fc_verify()) fail();\n')

        if embedded_code:
            memfile.write('\n  printf("\\n*** PASS ***\\n\\n");\n')
            memfile.write('  printf("Time for CNN: %d us\\n\\n", cnn_time);\n\n')
            if classification_layer or softmax:
                memfile.write('  printf("Classification results:\\n");\n'
                              '  for (i = 0; i < NUM_OUTPUTS; i++) {\n'
                              '    printf("[%6d] -> Class %d: %0.1f%%\\n", '
                              f'{"fc_output" if classification_layer else "ml_data"}[i], '
                              'i, (double) (100.0 * fc_softmax[i] / 32768.0));\n'
                              '  }\n\n')

    if riscv is not None and not riscv:
        if sleep:
            memfile.write('  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=1\n')
        memfile.write('  asm volatile("wfi"); // Let RISC-V run\n')

    memfile.write('  pass();\n')
    memfile.write('  return 0;\n}\n\n')


def verify_header(
        memfile,
        riscv_flash=False,
):
    """
    Write the header for the CNN verification function to `memfile`.
    """
    if riscv_flash:
        memfile.write(rv.RISCV_FLASH)
    memfile.write('int cnn_check(void)\n{\n  int rv = 1;\n')


def verify_footer(
        memfile,
):
    """
    Write the footer for the CNN verification function to `memfile`.
    """
    memfile.write('  return rv;\n}\n\n')


def fc_layer(
        memfile,
        weights_fh,
        weights,
        bias,
        cmsis_nn=False,
        softmax_only=False,
        output_width=8,
        num_classes=None,
):
    """
    Write the call to the fully connected layer with the given `weights` and
    `bias` to `memfile`. The `bias` argument can be `None`.
    """
    memfile.write('// Classification layer:\n')
    if not softmax_only:
        memfile.write(f'#define FC_IN {weights.shape[1]}\n')
        memfile.write(f'#define NUM_OUTPUTS {weights.shape[0]}\n')
    else:
        memfile.write(f'#define NUM_OUTPUTS {num_classes}\n')

    if not softmax_only:
        weights = convert_to_x4_q7_weights(weights)

        c_define(weights_fh, weights, 'FC_WEIGHTS', '%d', 16)
        memfile.write('static const q7_t fc_weights[] = FC_WEIGHTS;\n\n')

    if not cmsis_nn:
        memfile.write(f'static uint{output_width}_t ml_data[NUM_OUTPUTS];\n')
    if not softmax_only:
        memfile.write('static q15_t fc_buffer[FC_IN];\n')
        memfile.write('static q15_t fc_output[NUM_OUTPUTS];\n')
    memfile.write('static q15_t fc_softmax[NUM_OUTPUTS];\n\n')

    if bias is not None and not softmax_only:
        c_define(weights_fh, bias, 'FC_BIAS', '%d', 16)
        memfile.write('static const q7_t fc_bias[] = FC_BIAS;\n\n')

    if not cmsis_nn:
        memfile.write(f'int {"softmax" if softmax_only else "fc"}_layer(void)\n'
                      '{\n  cnn_unload(ml_data);\n')
    else:
        memfile.write('int fc_layer(q7_t *ml_data)\n'
                      '{\n')

    if not softmax_only:
        memfile.write('  arm_fully_connected_q7_q8p7_opt((q7_t *) ml_data, fc_weights, '
                      'FC_IN, NUM_OUTPUTS, 0, 7, '
                      f'{"fc_bias" if bias is not None else "NULL"}, '
                      'fc_output, fc_buffer);\n')
        memfile.write('  arm_softmax_q8p7_q15(fc_output, NUM_OUTPUTS, fc_softmax);\n\n')
    elif output_width == 32:
        memfile.write('  arm_softmax_q17p14_q15((const q31_t *) ml_data, '
                      'NUM_OUTPUTS, fc_softmax);\n\n')
    else:
        memfile.write('  arm_softmax_q7_q15((const q7_t *) ml_data, NUM_OUTPUTS, fc_softmax);\n\n')

    memfile.write('  return 1;\n}\n\n')


def fc_verify(
        memfile,
        sampledata,
        data,
):
    """
    Write the code to verify the fully connected layer to `memfile` against `data`.
    """
    memfile.write('// Expected output of classification layer:\n')
    c_define(sampledata, data, 'FC_EXPECTED', '%d', 16)
    memfile.write('static q15_t fc_expected[NUM_OUTPUTS] = FC_EXPECTED;\n\n')
    memfile.write('int fc_verify(void)\n'
                  '{\n')
    memfile.write('  return memcmp(fc_output, fc_expected, '
                  'NUM_OUTPUTS * sizeof(q15_t)) == 0;\n}\n\n')


def c_define(
        memfile,
        array,
        define_name,
        fmt,
        columns=8,
):
    """
    Write a #define to `memfile` for array `array` to `define_name`, using format `fmt` and
    creating a line break after `columns` items each.
    `fmt` can have two parts, separated by '%'. The part before the '%' sign is an optional
    prefix and can be empty, the part after the '%' is a formatting directive, e.g. '%08x'.
    """
    prefix, formatting = fmt.split('%')
    memfile.write(f'#define {define_name} {{ \\\n  ')
    for i, e in enumerate(array):
        memfile.write('{prefix}{item:{format}}'.format(prefix=prefix, item=e, format=formatting))
        if i + 1 < len(array):
            memfile.write(', ')
            if (i + 1) % columns == 0:
                memfile.write('\\\n  ')
    memfile.write(' \\\n}\n')
