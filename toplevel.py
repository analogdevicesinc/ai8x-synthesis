###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Toplevel C file structure generation
"""
import rv
import tornadocnn as tc
from armx4weights import convert_to_x4_q7_weights
from eprint import eprint

COPYRIGHT = \
    '/*******************************************************************************\n' \
    '* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.\n' \
    '*\n' \
    '* This software is protected by copyright laws of the United States and\n' \
    '* of foreign countries. This material may also be protected by patent laws\n' \
    '* and technology transfer regulations of the United States and of foreign\n' \
    '* countries. This software is furnished under a license agreement and/or a\n' \
    '* nondisclosure agreement and may only be used or reproduced in accordance\n' \
    '* with the terms of those agreements. Dissemination of this information to\n' \
    '* any party or parties not specified in the license agreement and/or\n' \
    '* nondisclosure agreement is expressly prohibited.\n' \
    '*\n' \
    '* The above copyright notice and this permission notice shall be included\n' \
    '* in all copies or substantial portions of the Software.\n' \
    '*\n' \
    '* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS\n' \
    '* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF\n' \
    '* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.\n' \
    '* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES\n' \
    '* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,\n' \
    '* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR\n' \
    '* OTHER DEALINGS IN THE SOFTWARE.\n' \
    '*\n' \
    '* Except as contained in this notice, the name of Maxim Integrated\n' \
    '* Products, Inc. shall not be used except as stated in the Maxim Integrated\n' \
    '* Products, Inc. Branding Policy.\n' \
    '*\n' \
    '* The mere transfer of this software does not imply any licenses\n' \
    '* of trade secrets, proprietary technology, copyrights, patents,\n' \
    '* trademarks, maskwork rights, or any other form of intellectual\n' \
    '* property whatsoever. Maxim Integrated Products, Inc. retains all\n' \
    '* ownership rights.\n' \
    '*******************************************************************************/\n\n'


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
        embedded_arm=False,
        fail_indicator=False,
        measure_energy=False,
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
        if embedded_code or embedded_arm:
            memfile.write('#include "mxc_sys.h"\n')
            if tc.dev.SUPPORT_GCFR:
                memfile.write('#include "gcfr_regs.h"\n')
            else:
                memfile.write('#include "bbfc_regs.h"\n')
            memfile.write('#include "fcr_regs.h"\n')
            memfile.write('#include "icc.h"\n')
            memfile.write('#include "led.h"\n')
            memfile.write('#include "tmr.h"\n')
            if measure_energy:
                memfile.write('#include "mxc_delay.h"\n')
        else:
            memfile.write('#include "global_functions.h" // For RTL Simulation\n')
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

    if embedded_arm:
        memfile.write('extern volatile void const *__FlashStart_; // Defined in linker file\n\n')

    if not cmsis_nn and (riscv is None or riscv):
        if embedded_code:
            if not measure_energy:
                memfile.write('uint32_t cnn_time; // Stopwatch\n\n')

            memfile.write('void fail(void)\n{\n')

            if fail_indicator:
                memfile.write('  mxc_gpio_cfg_t gpio_out;\n')
                memfile.write('  gpio_out.port = MXC_GPIO2;\n')
                memfile.write('  gpio_out.mask = MXC_GPIO_PIN_4;\n')
                memfile.write('  gpio_out.pad = MXC_GPIO_PAD_NONE;\n')
                memfile.write('  gpio_out.func = MXC_GPIO_FUNC_OUT;\n')
                memfile.write('  MXC_GPIO_Config(&gpio_out);\n')
                memfile.write('  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);\n\n')

            memfile.write('  printf("\\n*** FAIL ***\\n\\n");\n')
            memfile.write('  while (1);\n}\n\n')

        memfile.write('void cnn_wait(void)\n{\n')
        memfile.write(f'  while ((*((volatile uint32_t *) 0x{apb_base + tc.dev.C_CNN_BASE:08x}) '
                      '& (1<<12)) != 1<<12) ;\n')
        if embedded_code:
            memfile.write('  CNN_COMPLETE; // Signal that processing is complete\n')
            if not measure_energy:
                memfile.write('  cnn_time = MXC_TMR_SW_Stop(MXC_TMR0);\n')
        memfile.write('}\n\n')

    if master is not False:
        addr = apb_base + tc.ctl_addr(master, tc.dev.REG_CTL)

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
        riscv_debug=False,
        riscv_debugwait=True,
        camera=False,
        camera_format=None,
        device=84,
        channels=None,
        sleep=False,
        output_width=8,
        num_classes=None,
        clock_trim=None,
        embedded_arm=False,
        groups=None,
        boost=None,
        forever=False,
        fifo=False,
        mexpress=False,
        measure_energy=False,
):
    """
    Write the main function (including an optional call to the fully connected layer if
    `classification_layer` is `True`) to `memfile`.
    """
    assert groups is not None
    mask = 0
    for _, group in enumerate(groups):
        mask |= 1 << group
    unmask = ~mask & ((1 << tc.dev.P_NUMGROUPS) - 1)

    if softmax and output_width == 8:
        eprint('--softmax should only be used with `output_width: 32`', error=False)

    if unload:
        memfile.write(f'#define NUM_OUTPUTS {num_classes}\n')
        memfile.write(f'static int{output_width}_t ml_data[NUM_OUTPUTS];\n\n')

    if riscv is not None and not riscv and (embedded_arm or tc.dev.MODERN_SIM):
        memfile.write('void WakeISR(void)\n'
                      '{\n'
                      '  MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN & ~MXC_F_SEMA_IRQ0_CM4_IRQ;\n'
                      '}\n\n')

    memfile.write('int main(void)\n{\n')
    if clock_trim is not None and not riscv:
        memfile.write('  uint32_t trim;\n')
    if embedded_code and (classification_layer or softmax) or oneshot > 0:
        memfile.write('  int i;\n')
    if embedded_arm and riscv_debugwait:
        memfile.write('  int i;\n')
    if embedded_code and (classification_layer or softmax):
        memfile.write('  int digs, tens;\n')

    bbfc = 'BBFC' if not tc.dev.SUPPORT_GCFR else 'GCFR'

    if riscv is None or not riscv:
        if embedded_code or embedded_arm:
            if device == 84:
                memfile.write('  icache_enable();\n\n')
                memfile.write('  SYS_ClockEnable(SYS_PERIPH_CLOCK_AI);\n')
            else:
                memfile.write('\n  MXC_ICC_Enable(MXC_ICC0); // Enable cache\n\n')
                if clock_trim is not None:
                    memfile.write('  // Manual clock trim override:\n')
                    memfile.write('  *((volatile uint32_t *) 0x40000c00) = 1; '
                                  '// Set TME\n')
                    if clock_trim[0] or clock_trim[1]:
                        memfile.write('  trim = *((volatile uint32_t *) 0x40005420);\n')
                        if clock_trim[0]:
                            memfile.write('  trim &= ~0xffff;\n'
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
                                      f'0x{clock_trim[2]:x}; // HIRC96M (100 MHz) trim\n')
                    memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0; '
                                  '// Clear TME\n\n')

                if not measure_energy:
                    memfile.write('  // Switch to 100 MHz clock\n'
                                  '  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);\n')
                else:
                    memfile.write('  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IBRO);\n'
                                  '  // Switch to 7.37 MHz clock\n'
                                  '  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IBRO);\n')
                memfile.write('  SystemCoreClockUpdate();\n')

                if measure_energy:
                    memfile.write('\n  // Disable CNN clock\n'
                                  '  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);\n'
                                  '  // Disable power to CNN\n'
                                  '  MXC_BBFC->reg3 = 0xf; // Reset\n'
                                  '  MXC_BBFC->reg1 = 0x0; // Mask memory\n'
                                  '  MXC_BBFC->reg0 = 0x0; // Power\n'
                                  '  MXC_BBFC->reg2 = 0xf; // Iso\n'
                                  '  MXC_BBFC->reg3 = 0x0; // Reset\n'
                                  '  // Enable primary clock\n'
                                  '  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);\n\n'
                                  '  printf("Measuring system base power...\\n");\n'
                                  '  SYS_START;\n'
                                  '  MXC_Delay(SEC(1));\n'
                                  '  SYS_COMPLETE;\n')

                memfile.write('\n  // Reset all domains, restore power to CNN\n')
                memfile.write(f'  MXC_{bbfc}->reg3 = 0xf; // Reset\n')
                memfile.write(f'  MXC_{bbfc}->reg1 = 0x{mask:01x}; // Mask memory\n')
                memfile.write(f'  MXC_{bbfc}->reg0 = 0x{mask:01x}; // Power\n')
                memfile.write(f'  MXC_{bbfc}->reg2 = 0x{unmask:01x}; // Iso\n')
                memfile.write(f'  MXC_{bbfc}->reg3 = 0x0; // Reset\n\n')

                if not measure_energy:
                    memfile.write('  MXC_GCR->pclkdiv &= ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | '
                                  'MXC_F_GCR_PCLKDIV_CNNCLKSEL);\n'
                                  '  MXC_GCR->pclkdiv |= MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1; '
                                  ' // CNN clock: 0.5*100 MHz div 1\n'
                                  '  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); '
                                  '// Enable CNN clock\n')
                else:
                    memfile.write('  MXC_GCR->pclkdiv &= ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | '
                                  'MXC_F_GCR_PCLKDIV_CNNCLKSEL);\n'
                                  '  MXC_GCR->pclkdiv |= MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV8; '
                                  '// CNN clock: 0.5*7.37 MHz div 8\n'
                                  '  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); '
                                  '// Enable CNN clock\n')

                if boost is not None:
                    memfile.write(f'\n  // Configure P{boost[0]}.{boost[1]}, '
                                  'turn on the CNN Boost\n')
                    memfile.write('  mxc_gpio_cfg_t gpio_out;\n')
                    memfile.write(f'  gpio_out.port = MXC_GPIO{boost[0]};\n')
                    memfile.write(f'  gpio_out.mask = MXC_GPIO_PIN_{boost[1]};\n')
                    memfile.write('  gpio_out.pad = MXC_GPIO_PAD_NONE;\n')
                    memfile.write('  gpio_out.func = MXC_GPIO_FUNC_OUT;\n')
                    memfile.write('  MXC_GPIO_Config(&gpio_out);\n')
                    memfile.write('  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);\n')
        else:
            memfile.write('  icache_enable();\n\n')
            if device == 84:
                memfile.write('  MXC_GCR->perckcn1 &= ~0x20; // Enable CNN clock\n')
            else:
                memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000001; // Set TME\n')
                memfile.write('  *((volatile uint32_t *) 0x40006c04) = 0x000001a0; // 96M trim\n')
                memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000000; '
                              '// Clear TME\n\n')
                if tc.dev.SUPPORT_GCFR:
                    memfile.write('  MXC_GCR->clkctrl |= MXC_F_GCR_CLKCTRL_IPO_EN;'
                                  ' // Enable internal primary osc (IPO)\n')
                    memfile.write('  while ((MXC_GCR->clkctrl & MXC_F_GCR_CLKCTRL_IPO_RDY) == 0) ;'
                                  ' // Wait for osc\n')
                    memfile.write('  MXC_GCR->clkctrl |= MXC_S_GCR_CLKCTRL_SYSCLK_SEL_IPO;'
                                  ' // Select osc\n')

                    memfile.write('\n  // Reset all domains, restore power to CNN\n')
                    memfile.write('  MXC_GCFR->reg3 = 0xf; // Reset\n')
                    memfile.write(f'  MXC_GCFR->reg1 = 0x{mask:01x}; // Mask memory\n')
                    memfile.write(f'  MXC_GCFR->reg0 = 0x{mask:01x}; // Power\n')
                    memfile.write(f'  MXC_GCFR->reg2 = 0x{unmask:01x}; // Iso\n')
                    memfile.write('  MXC_GCFR->reg3 = 0x0; // Reset\n\n')

                    memfile.write('  MXC_GCR->pclkdiv &= ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | '
                                  'MXC_F_GCR_PCLKDIV_CNNCLKSEL);\n'
                                  '  MXC_GCR->pclkdiv |= MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1; '
                                  '// CNN clock: APB div 1\n')
                    memfile.write('  MXC_GCR->pclkdis0 &= ~MXC_F_GCR_PCLKDIS0_CNN;'
                                  ' // Enable CNN clock\n')
                else:
                    memfile.write('  MXC_GCR->clkcn |= MXC_F_GCR_CLKCN_HIRC96M_EN;'
                                  ' // Enable 96M\n')
                    memfile.write('  while ((MXC_GCR->clkcn & MXC_F_GCR_CLKCN_HIRC96M_RDY) == 0) ;'
                                  ' // Wait for 96M\n')
                    memfile.write('  MXC_GCR->clkcn |= MXC_S_GCR_CLKCN_CLKSEL_HIRC96;'
                                  ' // Select 96M\n')

                    memfile.write('\n  // Reset all domains, restore power to CNN\n')
                    memfile.write('  MXC_BBFC->reg3 = 0xf; // Reset\n')
                    memfile.write(f'  MXC_BBFC->reg1 = 0x{mask:01x}; // Mask memory\n')
                    memfile.write(f'  MXC_BBFC->reg0 = 0x{mask:01x}; // Power\n')
                    memfile.write(f'  MXC_BBFC->reg2 = 0x{unmask:01x}; // Iso\n')
                    memfile.write('  MXC_BBFC->reg3 = 0x0; // Reset\n\n')

                    memfile.write('  MXC_GCR->pckdiv = 0x00010000; // CNN clock 96M div 2\n')
                    memfile.write('  MXC_GCR->perckcn &= ~0x2000000; // Enable CNN clock\n')

        if riscv is not None:
            if riscv_cache:
                if embedded_code or embedded_arm:
                    memfile.write('\n  MXC_FCR->urvbootaddr = (uint32_t) &__FlashStart_; '
                                  '// Set RISC-V boot address\n')
                elif tc.dev.MODERN_SIM:
                    memfile.write(f'  MXC_FCR->urvbootaddr = 0x{rv.RISCV_CODE_ORIGIN:08x}; '
                                  '// Set RISC-V boot address\n')
                else:
                    memfile.write(f'  MXC_NBBFC->reg4 = 0x{rv.RISCV_CODE_ORIGIN:08x}; '
                                  '// Set RISC-V boot address\n')
            elif tc.dev.MODERN_SIM:
                memfile.write(f'  MXC_FCR->urvbootaddr = 0x{tc.dev.RISCV_SRAM_ORIGIN:08x}; '
                              '// Set RISC-V boot address\n')
            if riscv_exclusive:
                if embedded_code or embedded_arm or tc.dev.MODERN_SIM:
                    memfile.write('  MXC_FCR->urvctrl |= 0x00000001; '
                                  '// Exclusive SRAM access for RISC-V\n')
                else:
                    memfile.write('  *((volatile uint32_t *) 0x40000814) |= 0x00000001; '
                                  '// Exclusive SRAM access for RISC-V (MXC_NBBFC->reg5)\n')
            if embedded_code or embedded_arm or tc.dev.MODERN_SIM:
                memfile.write('  MXC_GCR->pclkdis1 &= ~MXC_F_GCR_PCLKDIS1_SMPHR; '
                              '// Enable Sempahore clock\n'
                              '  NVIC_SetVector(RISCV_IRQn, WakeISR); // Wakeup ISR\n'
                              '  MXC_GCR->pclkdis1 &= ~MXC_F_GCR_PCLKDIS1_CPU1; '
                              '// Enable RISC-V clock\n')
            else:
                memfile.write('  MXC_GCR->perckcn1 &= ~MXC_F_GCR_PERCKCN1_CPU1; '
                              '// Enable RISC-V clock\n')
        memfile.write('\n')
    elif riscv:
        if riscv_debug and embedded_code:
            memfile.write('  Debug_Init(); // Set up RISCV JTAG\n')
        if riscv_cache:
            if not embedded_code:
                memfile.write('  icache1_enable();\n')
                memfile.write('  invalidate_icache1();\n\n')
            else:
                memfile.write('  MXC_ICC_Enable(MXC_ICC1); // Enable cache\n\n')

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
        if embedded_code:
            memfile.write('  printf("\\n*** CNN Inference Test ***\\n");\n\n')

        if embedded_code:
            memfile.write('  if (!cnn_load()) fail();\n')
            memfile.write('  MXC_TMR_SW_Start(MXC_TMR0);\n')
        else:
            memfile.write('  if (!cnn_load()) { fail(); pass(); return 0; }\n')

        if stopstart:
            memfile.write('\n  cnn_stop();\n')
            memfile.write('  cnn_restart();\n\n')

        memfile.write('  cnn_wait();\n\n')
        if oneshot > 0:
            memfile.write(f'  for (i = 0; i < {oneshot}; i++) {{\n')
            memfile.write('    cnn_restart();\n')
            memfile.write('    cnn_wait();\n')
            memfile.write('  }\n\n')

        if not forever and boost is not None:
            memfile.write('  // Turn off the CNN Boost\n')
            memfile.write('  MXC_GPIO_OutClr(gpio_out.port, gpio_out.mask);\n\n')

        memfile.write('  if (!cnn_check()) fail();\n')
        if classification_layer or softmax:
            memfile.write(f'  if (!{"softmax" if softmax else "fc"}_layer()) fail();\n')
        elif unload:
            memfile.write(f'  cnn_unload((uint{output_width}_t *) ml_data);\n')
        if classification_layer:
            memfile.write('  if (!fc_verify()) fail();\n')

        if embedded_code:
            memfile.write('\n  printf("\\n*** PASS ***\\n\\n");\n\n')
            if not measure_energy:
                memfile.write('  printf("Inference time: %d us\\n\\n", cnn_time);\n\n')
            else:
                memfile.write('  printf("See monitor display for inference energy.\\n");\n\n')

        if not forever:
            if embedded_code:
                memfile.write('\n  // Disable CNN clock\n'
                              '  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);\n')
            elif tc.dev.MODERN_SIM:
                memfile.write('\n  // Disable CNN clock\n'
                              '  MXC_GCR->pclkdis0 |= MXC_F_GCR_PCLKDIS0_CNN;\n')
            memfile.write('  // Disable power to CNN\n')
            memfile.write(f'  MXC_{bbfc}->reg3 = 0xf; // Reset\n')
            memfile.write(f'  MXC_{bbfc}->reg1 = 0x0; // Mask memory\n')
            memfile.write(f'  MXC_{bbfc}->reg0 = 0x0; // Power\n')
            memfile.write(f'  MXC_{bbfc}->reg2 = 0xf; // Iso\n')
            memfile.write(f'  MXC_{bbfc}->reg3 = 0x0; // Reset\n\n')

        if not forever:
            if classification_layer or softmax:
                memfile.write('  printf("Classification results:\\n");\n'
                              '  for (i = 0; i < NUM_OUTPUTS; i++) {\n'
                              '    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;\n'
                              '    tens = digs % 10;\n'
                              '    digs = digs / 10;\n'
                              '    printf("[%7d] -> Class %d: %d.%d%%\\n", '
                              f'{"fc_output" if classification_layer else "ml_data"}[i], '
                              'i, digs, tens);\n'
                              '  }\n\n')
        else:
            memfile.write('  printf("Starting endless loop...\\n");\n\n  LED_On(1);\n\n')

            memfile.write('  while(1) {\n')

            gval = tc.dev.READY_SEL << 1
            if fifo:
                gval |= 1 << 15
            if device != 84:
                gval |= 1 << 3  # Enable clocks
            if mexpress:
                gval |= 1 << 20

            for _, group in enumerate(groups):
                addr = tc.dev.APB_BASE + tc.ctl_addr(group, tc.dev.REG_CTL)
                memfile.write(f'    *((volatile uint32_t *) 0x{addr:08x}) = 0x{gval:08x}; '
                              '// Stop SM\n')
            for _, group in enumerate(groups):
                val = gval | 0x800
                if group > 0:
                    val |= 0x01
                addr = tc.dev.APB_BASE + tc.ctl_addr(group, tc.dev.REG_CTL)
                memfile.write(f'    *((volatile uint32_t *) 0x{addr:08x}) = 0x{val:08x}; '
                              f'// Enable group {group}\n')

            addr = tc.dev.APB_BASE + tc.ctl_addr(0, tc.dev.REG_CTL)
            memfile.write(f'    *((volatile uint32_t *) 0x{addr:08x}) = 0x{gval | 0x01:08x}; '
                          '// Master enable group 0\n')

            memfile.write(f'    while ((*((volatile uint32_t *) '
                          f'0x{tc.dev.APB_BASE + tc.dev.C_CNN_BASE:08x}) '
                          '& (1<<12)) != 1<<12) ;\n')

            memfile.write('  }\n')

    if riscv is not None and not riscv:
        if sleep:
            memfile.write('  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=1\n')
        if embedded_arm or tc.dev.MODERN_SIM:
            if embedded_arm and riscv_debugwait:
                memfile.write('  for (i = 0; i < (1 << 27); i++); '
                              '// Let debugger interrupt if needed\n')
            memfile.write('  __WFI(); // Let RISC-V run\n')
        else:
            memfile.write('  asm volatile("wfi"); // Let RISC-V run\n')
    elif embedded_arm or tc.dev.MODERN_SIM:
        memfile.write('  // Signal the Cortex-M4\n'
                      '  MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN | MXC_F_SEMA_IRQ0_CM4_IRQ;\n\n')

    if not embedded_code and not embedded_arm:
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
        memfile.write(f'static int{output_width}_t ml_data[NUM_OUTPUTS];\n')
    if not softmax_only:
        memfile.write('static q15_t fc_buffer[FC_IN];\n')
        memfile.write('static q15_t fc_output[NUM_OUTPUTS];\n')
    memfile.write('static q15_t ml_softmax[NUM_OUTPUTS];\n\n')

    if bias is not None and not softmax_only:
        c_define(weights_fh, bias, 'FC_BIAS', '%d', 16)
        memfile.write('static const q7_t fc_bias[] = FC_BIAS;\n\n')

    if not cmsis_nn:
        memfile.write(f'int {"softmax" if softmax_only else "fc"}_layer(void)\n'
                      f'{{\n  cnn_unload((uint{output_width}_t *) ml_data);\n')
    else:
        memfile.write('int fc_layer(q7_t *ml_data)\n'
                      '{\n')

    if not softmax_only:
        memfile.write('  arm_fully_connected_q7_q8p7_opt((q7_t *) ml_data, fc_weights, '
                      'FC_IN, NUM_OUTPUTS, 0, 7, '
                      f'{"fc_bias" if bias is not None else "NULL"}, '
                      'fc_output, fc_buffer);\n')
        memfile.write('  arm_softmax_q8p7_q15(fc_output, NUM_OUTPUTS, ml_softmax);\n\n')
    elif output_width == 32:
        memfile.write('  softmax_q17p14_q15((const q31_t *) ml_data, '
                      'NUM_OUTPUTS, ml_softmax);\n\n')
    else:
        memfile.write('  arm_softmax_q7_q15((const q7_t *) ml_data, NUM_OUTPUTS, ml_softmax);\n\n')

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
