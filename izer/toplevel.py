###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Toplevel C file structure generation
"""
from typing import List, Optional, TextIO

from . import devices, rv, state
from . import tornadocnn as tc

COPYRIGHT = \
    '/*******************************************************************************\n' \
    '* Copyright (C) 2019-2023 Maxim Integrated Products, Inc., All rights Reserved.\n' \
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
        memfile: TextIO,
) -> None:
    """
    Write the copyright header to .c file handle `memfile`.
    """
    memfile.write(COPYRIGHT)


def header(
        memfile: TextIO,
        embedded_code: bool = False,
        master: bool = False,
        riscv: Optional[bool] = False,  # Tri-state: None/False/True
        embedded_arm: bool = False,
        groups: Optional[List[int]] = None,
        lib: Optional[bool] = False,  # Tri-state: None/False/True
        oneshot: int = 0,
) -> None:
    """
    Write include files and forward definitions to .c file handle `memfile`.
    The APB base address is passed in `apb_base`.
    """
    assert tc.dev is not None

    cmsis_nn = tc.dev.device == devices.CMSISNN
    main_code = riscv is None or riscv or cmsis_nn
    input_csv = state.input_csv is not None and main_code

    if tc.dev.SUPPORT_PLL:
        memfile.write('#include <assert.h>\n')
    memfile.write('#include <stdlib.h>\n'
                  '#include <stdint.h>\n')
    if embedded_code or state.verify_kernels:
        memfile.write('#include <string.h>\n')
    if embedded_code or state.verify_kernels and state.new_kernel_loader:
        memfile.write('#include <stdio.h>\n')
    if not cmsis_nn:
        if embedded_code or embedded_arm:
            memfile.write('#include "mxc.h"\n')
            if lib is None or lib:
                if tc.dev.SUPPORT_GCFR:
                    memfile.write('#include "gcfr_regs.h"\n')
                else:
                    memfile.write('#include "bbfc_regs.h"\n')
            if riscv is not None and not lib:
                memfile.write('#include "fcr_regs.h"\n'
                              '#include "sema_regs.h"\n')
        else:
            if tc.dev.MODERN_SIM:
                if input_csv:
                    memfile.write('#include "cameraif_regs.h"\n')
                memfile.write('#include "mxc_device.h"\n'
                              '#include "mxc_delay.h"\n'
                              '#include "mxc_assert.h"\n'
                              '#include "mxc_errors.h"\n'
                              '#include "mxc_lock.h"\n'
                              '#include "mxc_pins.h"\n'
                              '#include "mxc_sys.h"\n'
                              '#include "nvic_table.h"\n')
            memfile.write('#include "global_functions.h" // For RTL Simulation\n')
    if input_csv:
        memfile.write('#include "pcif_defines_af2.h"\n'
                      '#define NUM_DATA_WORDS 4\n'
                      '#include "pcif.c"\n')
    if embedded_code:
        memfile.write('#include "cnn.h"\n')
    elif not embedded_arm and tc.dev.SUPPORT_PLL:
        memfile.write('#define ipll_ctrl ito_ctrl\n'
                      '#define MXC_F_GCR_IPLL_CTRL_EN MXC_F_GCR_ITO_CTRL_EN\n'
                      '#define MXC_F_GCR_IPLL_CTRL_RDY MXC_F_GCR_ITO_CTRL_RDY\n'
                      '#define MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL MXC_S_GCR_PCLKDIV_CNNCLKSEL_ITO\n')

    if main_code:
        if (lib is True or lib is None) and not state.rtl_preload_weights:
            memfile.write(f'#include "{state.weight_filename}"\n')
        if not lib and (embedded_code or state.compact_data):
            memfile.write(f'#include "{state.sample_filename}"\n')
            if not cmsis_nn and state.generate_kat \
               and state.result_filename is not None:
                memfile.write(f'#include "{state.result_filename}"\n')
    memfile.write('\n')

    if not (embedded_code or embedded_arm):
        memfile.write('#define CNN_FAIL 0\n'
                      '#define CNN_OK 1\n\n')

    if not lib and embedded_arm:
        memfile.write('extern volatile void const *__FlashStart_; // Defined in linker file\n\n')

    if not lib and not cmsis_nn and main_code:
        if embedded_code or tc.dev.MODERN_SIM:
            if state.measure_energy and tc.dev.REQUIRE_PMON_GPIO:
                memfile.write('mxc_gpio_cfg_t gpio_trig1, gpio_trig2; // Port pins for PMON\n')
            memfile.write('volatile uint32_t cnn_time; // Stopwatch\n\n')

        if embedded_code:
            function_header(memfile, prefix='', function='fail', return_type='void')

            if state.forever or state.snoop_loop:
                memfile.write('  mxc_gpio_cfg_t gpio_out;\n')
                memfile.write('  gpio_out.port = MXC_GPIO2;\n')
                memfile.write('  gpio_out.mask = MXC_GPIO_PIN_4;\n')
                memfile.write('  gpio_out.pad = MXC_GPIO_PAD_NONE;\n')
                memfile.write('  gpio_out.func = MXC_GPIO_FUNC_OUT;\n')
                memfile.write('  MXC_GPIO_Config(&gpio_out);\n')
                memfile.write('  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);\n\n')

            memfile.write('  printf("\\n*** FAIL ***\\n\\n");\n')
            memfile.write('  while (1);\n')
            function_footer(memfile, return_value='void')  # fail()

    if (lib is None or lib) and not cmsis_nn and main_code:
        if embedded_code or tc.dev.MODERN_SIM:
            if not riscv:
                function_header(memfile, prefix='', function='CNN_ISR',
                                return_type='void')
            else:
                function_header(memfile, prefix='', function='CNN_IRQHandler',
                                return_type='void __attribute__((interrupt("machine")))')
            memfile.write('  // Acknowledge interrupt to all quadrants\n')
            for group in groups:
                addr = tc.dev.APB_BASE + tc.ctl_addr(group, tc.dev.REG_CTL)
                memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) &= ')
                if oneshot > 0 and not tc.dev.REQUIRE_ONESHOT_CLEAR:
                    memfile.write('~(1<<12);\n')
                elif any(state.streaming):
                    memfile.write('~((1<<12) | (1<<14) | 1);\n')
                else:
                    memfile.write('~((1<<12) | 1);\n')
            memfile.write('\n')
            if embedded_code and not state.measure_energy:
                memfile.write('  CNN_COMPLETE; // Signal that processing is complete\n')
            memfile.write('#ifdef CNN_INFERENCE_TIMER\n'
                          '  cnn_time = MXC_TMR_SW_Stop(CNN_INFERENCE_TIMER);\n'
                          '#else\n'
                          '  cnn_time = 1;\n'
                          '#endif\n')
            if riscv:
                memfile.write('\n  NVIC_ClearPendingIRQ(CNN_IRQn);\n'
                              '  NVIC_ClearPendingEVENT(CNN_IRQn);\n')
            function_footer(memfile, return_value='void')  # ISR()
        else:
            function_header(memfile, function='wait', return_type='void')
            memfile.write('  while ((*((volatile uint32_t *) '
                          f'0x{state.apb_base + tc.dev.C_CNN_BASE:08x}) & (1<<12)) != 1<<12) ;\n')
            function_footer(memfile, return_value='void')  # wait()

    if master is not False and (lib is None or lib):
        addr = state.apb_base + tc.ctl_addr(master, tc.dev.REG_CTL)

        function_header(memfile, function='continue')
        if embedded_code or tc.dev.MODERN_SIM:
            memfile.write('  cnn_time = 0;\n\n')
        memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) |= 1; '
                      f'// Re-enable quadrant {master}\n')
        function_footer(memfile)  # continue()

        function_header(memfile, function='stop')
        memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) &= ~1; '
                      f'// Disable quadrant {master}\n')
        function_footer(memfile)  # stop()


def function_header(
        memfile: Optional[TextIO],
        function: str = 'configure',
        arguments: str = 'void',
        return_type: str = 'int',
        prefix: str = 'cnn_',
) -> None:
    """
    Write the header for the a function to `memfile`. Optionally add the RV32 Flash attribute
    when `riscv_flash`. The function name is composed from `prefix` (default: 'cnn') and
    `function_name`, the return type is `return_type` (default: 'int').
    """
    if memfile is None:
        return
    if state.riscv_flash and not state.riscv_cache:
        memfile.write(rv.RISCV_FLASH)
    memfile.write(f'{return_type} {prefix}{function}({arguments})\n{{\n')


def function_footer(
        memfile: Optional[TextIO],
        return_value: str = 'CNN_OK',
) -> None:
    """
    Write the footer for a function to `memfile`, either returning `return_value` or nothing
    when `return_value` is 'void'.
    """
    if memfile is None:
        return
    if return_value != 'void':
        memfile.write(f'\n  return {return_value};\n')
    memfile.write('}\n\n')


def write_ml_data(
        memfile: TextIO,
        output_width: int,
) -> None:
    """
    Write the ml_data variable with `output_width` to `memfile`.
    """
    if output_width != 32:
        memfile.write('static int32_t ml_data32[(CNN_NUM_OUTPUTS + '
                      f'{32 // output_width - 1}) / {32 // output_width}];\n')
    else:
        memfile.write('static int32_t ml_data[CNN_NUM_OUTPUTS];\n')


def main(
        memfile: TextIO,
        apifile: Optional[TextIO],
        unload: bool = False,
        embedded_code: bool = False,
        riscv: Optional[bool] = None,
        channels: Optional[int] = None,
        output_width: int = 8,
        embedded_arm: bool = False,
        groups: Optional[List[int]] = None,
        forever: bool = False,
        fifo: bool = False,
        bias: bool = False,
        load_kernels: bool = True,
        oneshot: int = 0,
        name: str = '',
) -> None:
    """
    Write the main function to `memfile`.
    """
    assert tc.dev is not None

    main_code = riscv is None or riscv
    arm_code = riscv is None or not riscv
    arm_code_wrapper = riscv is not None and not riscv
    unload = unload and state.unload

    # Cache local variables
    boost = state.boost
    clock_trim = state.clock_trim
    debugwait = state.debug_wait
    measure_energy = state.measure_energy
    pll = state.pll
    pipeline = state.pipeline
    clock_switch = pll and pipeline and state.balance_power
    clock_speed = f'PLL ({tc.dev.PLL_SPEED} MHz)' if pll else f'APB ({tc.dev.APB_SPEED} MHz)'
    sleep = state.sleep
    softmax = state.softmax
    input_csv = state.input_csv is not None and main_code

    mfile = apifile or memfile

    assert groups is not None
    mask = 0
    for group in groups:
        mask |= 1 << group
    unmask = ~mask & ((1 << tc.dev.P_NUMGROUPS_ALL) - 1)

    if unload and not softmax:
        write_ml_data(memfile, output_width)
        memfile.write('\n')

    if arm_code_wrapper:
        if not state.wfi:
            memfile.write('static volatile int riscv_done;\n\n')
        function_header(memfile, prefix='', function='WakeISR', return_type='void')
        if embedded_arm or tc.dev.MODERN_SIM:
            memfile.write('  MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN & ~MXC_F_SEMA_IRQ0_CM4_IRQ;\n')
        else:
            memfile.write('  MXC_SEMA->irqr0 = MXC_S_SEMA_IRQR_IRQEN_EN | '
                          'MXC_S_SEMA_IRQR_IRQ_CLR;\n')
        if not state.wfi:
            memfile.write('  riscv_done = 1;\n')
        function_footer(memfile, return_value='void')  # WakeISR()

    # Add this to RTL simulations where it's missing from the SDK
    if arm_code_wrapper and sleep and not (embedded_code or embedded_arm):
        function_header(memfile, prefix='', function='_MXC_LP_ClearWakeStatus', return_type='void')
        memfile.write('  /* Write 1 to clear */\n'
                      '  MXC_PWRSEQ->lpwkst0 = 0xFFFFFFFF;\n'
                      '  MXC_PWRSEQ->lpwkst1 = 0xFFFFFFFF;\n'
                      '  MXC_PWRSEQ->lpwkst2 = 0xFFFFFFFF;\n'
                      '  MXC_PWRSEQ->lpwkst3 = 0xFFFFFFFF;\n')
        if embedded_code or tc.dev.MODERN_SIM:
            memfile.write('  MXC_PWRSEQ->lppwst  = 0xFFFFFFFF;\n')
        else:
            memfile.write('  MXC_PWRSEQ->lpwkst  = 0xFFFFFFFF;\n')
        function_footer(memfile, return_value='void')  # _MXC_LP_ClearWakeStatus

    function_header(memfile, prefix='', function='main')
    if clock_trim is not None and not riscv:
        memfile.write('  uint32_t trim;\n')
    if embedded_code and softmax or oneshot > 0 or measure_energy and not arm_code_wrapper:
        memfile.write('  int i;\n')
    if embedded_code and not forever and softmax:
        memfile.write('  int digs, tens;\n')
        if output_width != 32:
            memfile.write(f'int{output_width}_t *ml_data = '
                          f'(int{output_width}_t *) ml_data32;\n')
    if embedded_code and softmax or oneshot > 0:
        memfile.write('\n')

    bbfc = 'BBFC' if not (embedded_code or embedded_arm) and not tc.dev.MODERN_SIM \
        or not tc.dev.SUPPORT_GCFR else 'GCFR'

    if arm_code:
        if embedded_code or embedded_arm:
            memfile.write('  MXC_ICC_Enable(MXC_ICC0); // Enable cache\n\n')
            if clock_trim is not None:
                memfile.write('  // Manual clock trim override:\n')
                memfile.write('  *((volatile uint32_t *) 0x40000c00) = 1; '
                              '// Set TME\n')
                # pylint: disable=unsubscriptable-object
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
                # pylint: enable=unsubscriptable-object

            memfile.write(f'  // Switch to {tc.dev.IPO_SPEED} MHz clock\n'
                          '  MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);\n')
            if pll:
                memfile.write('  MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN;'
                              ' // Enable IPLL\n')
            memfile.write('  SystemCoreClockUpdate();\n')
        else:
            memfile.write('  icache_enable();\n\n')
            memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000001; // Set TME\n')
            memfile.write('  *((volatile uint32_t *) 0x40006c04) = 0x000001a0; // 96M trim\n')
            memfile.write('  *((volatile uint32_t *) 0x40000c00) = 0x00000000; '
                          '// Clear TME\n\n')
            if tc.dev.SUPPORT_GCFR and tc.dev.MODERN_SIM:
                memfile.write('  MXC_GCR->clkctrl |= MXC_F_GCR_CLKCTRL_IPO_EN;'
                              ' // Enable internal primary osc (IPO)\n')
                if pll:
                    memfile.write('  MXC_GCR->ipll_ctrl |= MXC_F_GCR_IPLL_CTRL_EN;'
                                  ' // Enable IPLL\n')
                memfile.write('  while ((MXC_GCR->clkctrl & MXC_F_GCR_CLKCTRL_IPO_RDY) == 0) ;'
                              ' // Wait for osc\n'
                              '  MXC_GCR->clkctrl |= MXC_S_GCR_CLKCTRL_SYSCLK_SEL_IPO;'
                              ' // Select osc\n')

                if not tc.dev.MODERN_SIM:
                    memfile.write('\n  // Reset all domains, restore power to CNN\n')
                    memfile.write('  MXC_GCFR->reg3 = 0xf; // Reset\n')
                    memfile.write(f'  MXC_GCFR->reg1 = 0x{mask:01x}; // Mask memory\n')
                    memfile.write(f'  MXC_GCFR->reg0 = 0x{mask:01x}; // Power\n')
                    memfile.write(f'  MXC_GCFR->reg2 = 0x{unmask:01x}; // Iso\n')
                    memfile.write('  MXC_GCFR->reg3 = 0x0; // Reset\n\n')

                    memfile.write('  MXC_GCR->pclkdiv &= ~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | '
                                  'MXC_F_GCR_PCLKDIV_CNNCLKSEL);\n'
                                  '  MXC_GCR->pclkdiv |= MXC_S_GCR_PCLKDIV_'
                                  f'CNNCLKDIV_DIV{state.clock_divider}; '
                                  f'// CNN clock: APB div {state.clock_divider}\n')
                    memfile.write('  MXC_GCR->pclkdis0 &= ~MXC_F_GCR_PCLKDIS0_CNN;'
                                  ' // Enable CNN clock\n')
            else:
                memfile.write('  MXC_GCR->clkcn |= MXC_F_GCR_CLKCN_HIRC96M_EN;'
                              ' // Enable 96M\n')
                memfile.write('  while ((MXC_GCR->clkcn & MXC_F_GCR_CLKCN_HIRC96M_RDY) == 0) ;'
                              ' // Wait for 96M\n')
                memfile.write('  MXC_GCR->clkcn |= MXC_S_GCR_CLKCN_CLKSEL_HIRC96;'
                              ' // Select 96M\n')

                if not tc.dev.MODERN_SIM:
                    memfile.write('\n  // Reset all domains, restore power to CNN\n')
                    memfile.write(f'  MXC_{bbfc}->reg3 = 0xf; // Reset\n')
                    memfile.write(f'  MXC_{bbfc}->reg1 = 0x{mask:01x}; // Mask memory\n')
                    memfile.write(f'  MXC_{bbfc}->reg0 = 0x{mask:01x}; // Power\n')
                    memfile.write(f'  MXC_{bbfc}->reg2 = 0x{unmask:01x}; // Iso\n')
                    memfile.write(f'  MXC_{bbfc}->reg3 = 0x0; // Reset\n\n')

                    memfile.write('  MXC_GCR->pckdiv = 0x00010000; // CNN clock 96M div 2\n')
                    memfile.write('  MXC_GCR->perckcn &= ~0x2000000; // Enable CNN clock\n')

        if riscv is not None:
            if state.riscv_cache:
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
            if state.riscv_exclusive:
                if embedded_code or embedded_arm or tc.dev.MODERN_SIM:
                    memfile.write('  MXC_FCR->urvctrl |= 0x00000001; '
                                  '// Exclusive SRAM access for RISC-V\n')
                else:
                    memfile.write('  *((volatile uint32_t *) 0x40000814) |= 0x00000001; '
                                  '// Exclusive SRAM access for RISC-V (MXC_NBBFC->reg5)\n')
            if embedded_code or embedded_arm or tc.dev.MODERN_SIM:
                memfile.write('  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_SMPHR); '
                              '// Enable Semaphore clock\n')
                memfile.write('  MXC_NVIC_SetVector(RISCV_IRQn, WakeISR); // Set wakeup ISR\n')
                if (embedded_code or embedded_arm) and debugwait:
                    memfile.write('\n  // DO NOT DELETE THIS LINE:\n'
                                  f'  MXC_Delay(SEC({debugwait})); '
                                  '// Let debugger interrupt if needed\n\n')
                memfile.write('  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CPU1); '
                              '// Enable RISC-V clock\n')
            else:
                memfile.write('  MXC_GCR->perckcn1 &= ~MXC_F_GCR_PERCKCN1_SMPHRD; '
                              '// Clear semaphore disable\n'
                              '  MXC_NVIC_SetVector(RV32_IRQn, WakeISR); // Set wakeup ISR\n'
                              '  MXC_GCR->perckcn1 &= ~MXC_F_GCR_PERCKCN1_CPU1; '
                              '// Enable RISC-V clock\n')
        else:
            if (embedded_code or embedded_arm) and debugwait:
                memfile.write('\n  printf("Waiting...\\n");\n\n'
                              '  // DO NOT DELETE THIS LINE:\n'
                              f'  MXC_Delay(SEC({debugwait})); '
                              '// Let debugger interrupt if needed\n')
        memfile.write('\n')
    elif riscv:
        if state.riscv_debug and embedded_code:
            memfile.write('  Debug_Init(); // Set up RISCV JTAG\n')
        if state.riscv_cache:
            if not embedded_code:
                memfile.write('  icache1_enable();\n')
                memfile.write('  invalidate_icache1();\n\n')
            else:
                memfile.write('  MXC_ICC_Enable(MXC_ICC1); // Enable cache\n\n')

    if main_code and measure_energy and tc.dev.REQUIRE_PMON_GPIO:
        memfile.write('  // Configure port pins for PMON\n'
                      '  gpio_trig1.port = MXC_GPIO1;\n'
                      '  gpio_trig1.mask = MXC_GPIO_PIN_6;\n'
                      '  gpio_trig1.pad = MXC_GPIO_PAD_NONE;\n'
                      '  gpio_trig1.func = MXC_GPIO_FUNC_OUT;\n'
                      '  MXC_GPIO_Config(&gpio_trig1);\n\n'
                      '  gpio_trig2.port = MXC_GPIO1;\n'
                      '  gpio_trig2.mask = MXC_GPIO_PIN_7;\n'
                      '  gpio_trig2.pad = MXC_GPIO_PAD_NONE;\n'
                      '  gpio_trig2.func = MXC_GPIO_FUNC_OUT;\n'
                      '  MXC_GPIO_Config(&gpio_trig2);\n\n')

    if input_csv:
        memfile.write('  enable_pcif_clock(); // Enable camera clock\n')
        memfile.write('  set_pcif_gpio_altf();\n\n')
        if state.input_csv_format == 555:
            mode = '10'
            comment = '555'
        elif state.input_csv_format == 565:
            mode = '12'
            comment = '565'
        else:
            mode = '8'  # Default
            comment = '888'
        memfile.write(f'  // Enable {comment} format single image in external timing mode\n')
        if not tc.dev.MODERN_SIM:
            memfile.write('  MXC_CAMERAIF0->ctrl = MXC_S_CAMERAIF_CTRL_READ_MODE_SINGLE_IMG +\n'
                          f'                        MXC_S_CAMERAIF_CTRL_DATA_WIDTH_{mode}BIT +\n'
                          '                        MXC_S_CAMERAIF_CTRL_DS_TIMING_EN_DIS +\n'
                          '                        MXC_S_CAMERAIF_CTRL_PCIF_SYS_EN_EN')
            if channels == 3:
                memfile.write(' +\n                        (1<<30);\n\n')
            else:
                memfile.write(';\n\n')
        else:
            memfile.write('  MXC_PCIF->ctrl = MXC_S_CAMERAIF_CTRL_READ_MODE_SINGLE_IMG +\n'
                          f'                   MXC_S_CAMERAIF_CTRL_DATA_WIDTH_{mode}BIT +\n'
                          '                   MXC_F_CAMERAIF_CTRL_PCIF_SYS')
            if channels == 3:
                memfile.write(' +\n                   MXC_F_CAMERAIF_CTRL_THREE_CH_EN;\n\n')
            else:
                memfile.write(';\n\n')

    sleep_api = 'MXC_LP_EnterLowPowerMode' if sleep else 'MXC_LP_EnterSleepMode'

    if main_code:
        if embedded_code or tc.dev.MODERN_SIM:
            if measure_energy:
                memfile.write('  cnn_disable(); // Disable clock and power to CNN\n'
                              '  // Enable primary clock\n'
                              '  MXC_SYS_ClockSourceEnable(MXC_SYS_CLOCK_IPO);\n\n'
                              '  printf("Measuring system base (idle) power...\\n");\n'
                              '  SYS_START;\n')
                if not riscv:
                    memfile.write('  MXC_Delay(SEC(1));\n')
                else:
                    memfile.write('  MXC_TMR_Delay(MXC_TMR0, 1000000);\n')
                memfile.write('  SYS_COMPLETE;\n\n')

            if embedded_code and apifile is not None:
                cdiv = 4 if pll and state.balance_power else state.clock_divider
                memfile.write('  // Enable peripheral, enable CNN interrupt, turn on CNN clock\n'
                              f'  // CNN clock: {clock_speed} div {cdiv}\n'
                              '  cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_'
                              f'{"IPLL" if pll else "PCLK"}, '
                              f'MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV{cdiv});\n')
                function_header(apifile, function='enable',
                                arguments='uint32_t clock_source, uint32_t clock_divider')

            mfile.write('  // Reset all domains, restore power to CNN\n')
            mfile.write(f'  MXC_{bbfc}->reg3 = 0xf; // Reset\n')
            mfile.write(f'  MXC_{bbfc}->reg1 = 0x{mask:01x}; // Mask memory\n')
            mfile.write(f'  MXC_{bbfc}->reg0 = 0x{mask:01x}; // Power\n')
            if embedded_code and state.enable_delay > 0:
                mfile.write(f'  MXC_Delay(MSEC({state.enable_delay})); '
                            '// Wait for load switches\n')
            mfile.write(f'  MXC_{bbfc}->reg2 = 0x{unmask:01x}; // Iso\n')
            mfile.write(f'  MXC_{bbfc}->reg3 = 0x0; // Reset\n\n')

            if tc.dev.SUPPORT_PLL:
                if embedded_code:
                    mfile.write('  if (clock_source == MXC_S_GCR_PCLKDIV_CNNCLKSEL_IPLL)\n  ')
                if embedded_code or pll:
                    mfile.write('  while ((MXC_GCR->ipll_ctrl & MXC_F_GCR_IPLL_CTRL_RDY) '
                                '!= MXC_F_GCR_IPLL_CTRL_RDY) ; // Wait for PLL\n\n')

            if embedded_code and apifile is not None:
                mfile.write('  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & '
                            '~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))\n'
                            '                     | clock_divider | clock_source;\n')
            else:
                select_clock(mfile, 'IPLL' if pll else 'PCLK', f'DIV{state.clock_divider}',
                             f'CNN clock: {clock_speed} div {state.clock_divider}')

            mfile.write('  MXC_SYS_ClockEnable(MXC_SYS_PERIPH_CLOCK_CNN); '
                        '// Enable CNN clock\n\n')

            if not riscv:
                if embedded_code:
                    mfile.write('  MXC_NVIC_SetVector(CNN_IRQn, CNN_ISR); '
                                '// Set CNN complete vector\n')
                else:
                    mfile.write('  NVIC_SetVector(CNN_IRQn, CNN_ISR); '
                                '// Set CNN complete vector\n')
            else:
                mfile.write('  // Set CNN complete vector\n'
                            '  __enable_irq();\n'
                            '  NVIC_EnableIRQ(CNN_IRQn);\n'
                            '  NVIC_EnableEVENT(CNN_IRQn);\n')

            if embedded_code and apifile is not None:
                function_footer(apifile)  # enable()
            if embedded_code and apifile is not None:
                function_header(apifile, function='boost_enable',
                                arguments='mxc_gpio_regs_t *port, uint32_t pin')

            if boost is not None or apifile is not None:
                # pylint: disable=unsubscriptable-object
                if boost is not None and apifile is None:
                    memfile.write(f'\n  // Configure P{boost[0]}.{boost[1]}, '
                                  'turn on the CNN Boost\n')
                mfile.write('  mxc_gpio_cfg_t gpio_out;\n')
                if boost is not None and apifile is None:
                    memfile.write(f'  gpio_out.port = MXC_GPIO{boost[0]};\n')
                    memfile.write(f'  gpio_out.mask = MXC_GPIO_PIN_{boost[1]};\n')
                else:
                    mfile.write('  gpio_out.port = port;\n')
                    mfile.write('  gpio_out.mask = pin;\n')
                    if boost is not None:
                        memfile.write(f'  cnn_boost_enable(MXC_GPIO{boost[0]}, '
                                      f'MXC_GPIO_PIN_{boost[1]}); // Turn on the boost circuit\n')
                # pylint: enable=unsubscriptable-object
                mfile.write('  gpio_out.pad = MXC_GPIO_PAD_NONE;\n')
                mfile.write('  gpio_out.func = MXC_GPIO_FUNC_OUT;\n')
                mfile.write('  MXC_GPIO_Config(&gpio_out);\n')
                mfile.write('  MXC_GPIO_OutSet(gpio_out.port, gpio_out.mask);\n')

            if embedded_code and apifile is not None:
                function_footer(apifile)  # boost_enable() or enable()

            memfile.write('\n')

        if embedded_code:
            memfile.write(f'  printf("\\n*** CNN Inference Test {name} ***\\n");\n\n')
            if not state.zero_sram:
                memfile.write('  cnn_init(); // Bring state machine into consistent state\n')
            else:
                memfile.write('  if (cnn_init() != CNN_OK) fail();\n')
            if measure_energy:
                memfile.write('\n  printf("Measuring weight loading...\\n");\n'
                              '  CNN_START;\n'
                              '  for (i = 0; i < 100; i++)\n'
                              '    cnn_load_weights(); // Load kernels\n'
                              '  CNN_COMPLETE;\n\n'
                              '  MXC_TMR_Delay(MXC_TMR0, 500000);\n')
                if not fifo:
                    memfile.write('  printf("Measuring input loading...\\n");\n'
                                  '  CNN_START;\n'
                                  '  for (i = 0; i < 100; i++)\n'
                                  '    load_input(); // Load data input\n'
                                  '  CNN_COMPLETE;\n\n')
                else:
                    memfile.write('  printf("Skipping input loading (using FIFO)...\\n");\n'
                                  '  CNN_START;\n'
                                  '  MXC_TMR_Delay(MXC_TMR0, 10); // Dummy delay displays as 0\n'
                                  '  CNN_COMPLETE;\n\n')
            else:
                memfile.write('  cnn_load_weights(); // Load kernels\n')
            if state.verify_kernels:
                memfile.write('  if (cnn_verify_weights() != CNN_OK) fail();\n'
                              '  printf("Weights verified successfully.\\n");\n')
            if bias:
                memfile.write('  cnn_load_bias();\n')
            else:
                memfile.write('  cnn_load_bias(); // Not used in this network\n')
            if state.verify_writes:
                memfile.write('  if (cnn_configure() != CNN_OK) fail(); '
                              '// Configure state machine\n')
            else:
                memfile.write('  cnn_configure(); // Configure state machine\n')
            if not measure_energy:
                if not fifo:
                    memfile.write('  load_input(); // Load data input\n')
                if clock_switch:
                    select_clock(memfile, 'IPLL', f'DIV{state.clock_divider}',
                                 f'CNN clock: {clock_speed} div {state.clock_divider}',
                                 pll_wait=False)
                memfile.write('  cnn_start(); // Start CNN processing\n')
                if fifo:
                    memfile.write('  load_input(); // Load data input via FIFO\n')
            memfile.write('\n')
        else:
            memfile.write('  cnn_init(); // Bring state machine into consistent state\n')
            if load_kernels:
                memfile.write('  cnn_load_weights(); // Load kernels\n')
            else:
                memfile.write('  // Kernels are pre-loaded\n')
            if state.verify_kernels:
                memfile.write('  if (cnn_verify_weights() != CNN_OK) { fail(); pass(); '
                              'return 0; }\n')
            if bias:
                memfile.write('  cnn_load_bias();\n')
            else:
                memfile.write('  // No bias values\n')
            memfile.write('  if (cnn_configure() != CNN_OK) { fail(); pass(); return 0; }\n')

        if state.stopstart:
            memfile.write('\n  cnn_stop();\n')
            memfile.write('  cnn_continue();\n\n')

        if not measure_energy:
            if embedded_code or tc.dev.MODERN_SIM:
                if state.wfi:
                    if not (riscv or embedded_code):
                        memfile.write('  SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0\n')
                    memfile.write('  while (cnn_time == 0)\n')
                    if embedded_code:
                        memfile.write(f'    {sleep_api}(); // Wait for CNN\n\n')
                    elif not riscv:
                        memfile.write('    __WFI(); // Wait for CNN\n\n')
                    else:
                        memfile.write('    asm volatile("wfi"); // Wait for CNN\n\n')
                else:
                    memfile.write('  while (cnn_time == 0); // Spin wait\n')
            else:
                memfile.write('  cnn_wait();\n\n')
        else:
            memfile.write('  MXC_TMR_Delay(MXC_TMR0, 500000);\n'
                          '  printf("Measuring input load + inference...\\n");\n'
                          '  CNN_START; // Allow capture of processing time\n'
                          '  for (i = 0; i < 100; i++) {\n')
            if not fifo:
                memfile.write('    load_input(); // Load data input\n')
            if clock_switch:
                select_clock(memfile, 'IPLL', f'DIV{state.clock_divider}',
                             f'CNN clock: {clock_speed} div {state.clock_divider}',
                             pll_wait=False, prefix='  ')
            memfile.write('    cnn_start(); // Run inference\n')
            if fifo:
                memfile.write('    load_input(); // Load data input via FIFO\n')
            if state.wfi:
                if not (riscv or embedded_code):
                    memfile.write('    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0\n')
                memfile.write('    while (cnn_time == 0)\n')
                if embedded_code:
                    memfile.write(f'      {sleep_api}(); // Wait for CNN\n')
                elif not riscv:
                    memfile.write('      __WFI(); // Wait for CNN\n')
                else:
                    memfile.write('      asm volatile("wfi"); // Wait for CNN\n')
            else:
                memfile.write('    while (cnn_time == 0); // Spin wait\n')
            if clock_switch:
                select_clock(memfile, 'IPLL', 'DIV4', f'CNN clock: {clock_speed} div 4',
                             pll_wait=False, prefix='  ')
            memfile.write('  }\n'
                          '  CNN_COMPLETE;\n\n')

        if oneshot > 0:
            memfile.write(f'  for (i = 0; i < {oneshot}; i++) {{\n')
            memfile.write('    cnn_continue();\n')
            if embedded_code or tc.dev.MODERN_SIM:
                if state.wfi:
                    if not (riscv or embedded_code):
                        memfile.write('    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0\n')
                    memfile.write('    while (cnn_time == 0)\n')
                    if embedded_code:
                        memfile.write(f'      {sleep_api}(); // Wait for CNN\n')
                    elif not riscv:
                        memfile.write('      __WFI(); // Wait for CNN\n')
                    else:
                        memfile.write('      asm volatile("wfi"); // Wait for CNN\n')
                else:
                    memfile.write('    while (cnn_time == 0); // Spin wait\n')
            else:
                memfile.write('    cnn_wait();\n')
            memfile.write('  }\n\n')

        if clock_switch:
            select_clock(memfile, 'IPLL', 'DIV4',
                         f'Switch CNN clock to {clock_speed} div 4\n',
                         pll_wait=False)

        if embedded_code and apifile is not None:
            function_header(apifile, function='boost_disable',
                            arguments='mxc_gpio_regs_t *port, uint32_t pin')
            mfile.write('  mxc_gpio_cfg_t gpio_out;\n')
            mfile.write('  gpio_out.port = port;\n')
            mfile.write('  gpio_out.mask = pin;\n')
            mfile.write('  gpio_out.pad = MXC_GPIO_PAD_NONE;\n')
            mfile.write('  gpio_out.func = MXC_GPIO_FUNC_OUT;\n')
            mfile.write('  MXC_GPIO_Config(&gpio_out);\n')
            mfile.write('  MXC_GPIO_OutClr(gpio_out.port, gpio_out.mask);\n')
            function_footer(apifile)  # boost_disable()

        # pylint: disable=unsubscriptable-object
        if not forever and boost is not None:
            if apifile is None:
                memfile.write('  // Turn off the CNN Boost\n')
                memfile.write('  MXC_GPIO_OutClr(gpio_out.port, gpio_out.mask);\n\n')
            else:
                memfile.write(f'  cnn_boost_disable(MXC_GPIO{boost[0]}, '
                              f'MXC_GPIO_PIN_{boost[1]}); // Turn off the boost circuit\n\n')
        # pylint: enable=unsubscriptable-object

        if state.generate_kat:
            memfile.write('  if (check_output() != CNN_OK) fail();\n')
        if softmax:
            memfile.write('  softmax_layer();\n')
        elif unload:
            memfile.write('  cnn_unload((uint32_t *) '
                          f'ml_data{"32" if output_width != 32 else ""});\n')

        if embedded_code:
            memfile.write('\n  printf("\\n*** PASS ***\\n\\n");\n\n'
                          '#ifdef CNN_INFERENCE_TIMER\n'
                          f'  printf("Approximate {"data loading and " if fifo else ""}'
                          'inference time: %u us\\n\\n", cnn_time);\n'
                          '#endif\n\n')
            if measure_energy:
                memfile.write('  printf("See monitor display for inference energy.\\n\\n");\n\n')

        if not forever:
            if embedded_code and apifile is not None:
                memfile.write('  cnn_disable(); // Shut down CNN clock, disable peripheral\n\n')
                function_header(apifile, function='disable')
            if embedded_code or tc.dev.MODERN_SIM:
                mfile.write('  // Disable CNN clock\n'
                            '  MXC_SYS_ClockDisable(MXC_SYS_PERIPH_CLOCK_CNN);\n\n')
            mfile.write('  // Disable power to CNN\n')
            mfile.write(f'  MXC_{bbfc}->reg3 = 0xf; // Reset\n')
            mfile.write(f'  MXC_{bbfc}->reg2 |= 0xf; // Iso\n')
            mfile.write(f'  MXC_{bbfc}->reg0 = 0x0; // Power\n')
            mfile.write(f'  MXC_{bbfc}->reg1 = 0x0; // Mask memory\n')
            mfile.write(f'  MXC_{bbfc}->reg3 = 0x0; // Reset\n')

            if embedded_code and apifile is not None:
                function_footer(apifile)  # disable()
            if pll:
                memfile.write('  MXC_GCR->ipll_ctrl &= ~MXC_F_GCR_IPLL_CTRL_EN; '
                              '// Disable IPLL\n\n')

        if not forever:
            if softmax:
                memfile.write('  printf("Classification results:\\n");\n'
                              '  for (i = 0; i < CNN_NUM_OUTPUTS; i++) {\n'
                              '    digs = (1000 * ml_softmax[i] + 0x4000) >> 15;\n'
                              '    tens = digs % 10;\n'
                              '    digs = digs / 10;\n'
                              '    printf("[%7d] -> Class %d: %d.%d%%\\n", ml_data[i], '
                              'i, digs, tens);\n'
                              '  }\n')
        else:
            memfile.write('  printf("Starting endless loop...\\n");\n\n  LED_On(1);\n\n'
                          '  while(1) {\n'
                          '    cnn_start();\n')
            if embedded_code or tc.dev.MODERN_SIM:
                if not (riscv or embedded_code):
                    memfile.write('    SCB->SCR &= ~SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=0\n')
                memfile.write('    while (cnn_time == 0)\n')
                if embedded_code:
                    memfile.write(f'      {sleep_api}(); // Wait for CNN\n')
                elif not riscv:
                    memfile.write('      __WFI(); // Wait for CNN\n')
                else:
                    memfile.write('      asm volatile("wfi"); // Wait for CNN\n')
            else:
                memfile.write('    cnn_wait();\n')

            memfile.write('  }\n')

    if riscv is not None:
        if not riscv:
            if sleep and tc.dev.REQUIRE_SEMA_LPWKEN:
                if not (embedded_code or embedded_arm or tc.dev.MODERN_SIM):
                    memfile.write('  MXC_PWRSEQ->lpwken |= 0x400; // CPU1WKEN=1\n')
                else:
                    memfile.write('  MXC_PWRSEQ->lppwen |= MXC_F_PWRSEQ_LPPWEN_CPU1;\n')
            if (embedded_code or embedded_arm) and state.wfi:
                memfile.write(f'  {sleep_api}(); // Let RISC-V run\n')
            else:
                if sleep:
                    memfile.write('  _MXC_LP_ClearWakeStatus();\n'
                                  '  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk; // SLEEPDEEP=1\n')
                if state.wfi:
                    memfile.write('  __WFI(); // Let RISC-V run\n')
                else:
                    memfile.write('  riscv_done = 0;\n'
                                  '  while (riscv_done == 0); // Let RISC-V run\n')
        elif embedded_code or tc.dev.MODERN_SIM:
            memfile.write('\n  // Signal the Cortex-M4\n'
                          '  MXC_SEMA->irq0 = MXC_F_SEMA_IRQ0_EN | MXC_F_SEMA_IRQ0_CM4_IRQ;\n')
        else:
            memfile.write('\n  // Signal the Cortex-M4\n'
                          '  MXC_SEMA->irqr0 = MXC_S_SEMA_IRQR_IRQEN_EN | '
                          'MXC_S_SEMA_IRQR_IRQ_SET;\n')

    if not embedded_code and not embedded_arm:
        memfile.write('\n  pass();')
    function_footer(memfile, return_value='0')  # Exit main - don't change from 0


def softmax_layer(
        memfile: TextIO,
        output_width: int = 8,
        shift: int = 0,
) -> None:
    """
    Write the call to the softmax layer to `memfile`.
    """
    memfile.write('// Classification layer:\n')
    write_ml_data(memfile, output_width)
    memfile.write('static q15_t ml_softmax[CNN_NUM_OUTPUTS];\n\n')

    function_header(memfile, prefix='',
                    function='softmax_layer',
                    return_type='void')
    memfile.write(f'  cnn_unload((uint32_t *) ml_data{"32" if output_width != 32 else ""});\n')

    if output_width == 32:
        if shift == 0:
            memfile.write('  softmax_q17p14_q15((const q31_t *) ml_data, '
                          'CNN_NUM_OUTPUTS, ml_softmax);\n')
        else:
            memfile.write('  softmax_shift_q17p14_q15((q31_t *) ml_data, '
                          f'CNN_NUM_OUTPUTS, {shift}, ml_softmax);\n')
    else:
        memfile.write('  arm_softmax_q7_q15((const q7_t *) ml_data32, '
                      'CNN_NUM_OUTPUTS, ml_softmax);\n')

    function_footer(memfile, return_value='void')


def c_define(
        memfile: TextIO,
        array: List,
        define_name: str,
        fmt: str,
        columns: int = 8,
        size: int = 32,
) -> None:
    """
    Write a #define to `memfile` for array `array` to `define_name`, using format `fmt` and
    creating a line break after `columns` items each.
    `fmt` can have two parts, separated by '%'. The part before the '%' sign is an optional
    prefix and can be empty, the part after the '%' is a formatting directive, e.g. '%08x'.
    """
    prefix, formatting = fmt.split('%')
    memfile.write(f'#define {define_name} {{ \\\n  ')
    for i, e in enumerate(array):
        if size == 8:
            memfile.write(f'{prefix}{e:{formatting}}')
        else:
            memfile.write(f'{prefix}{e & 0xffffffff:{formatting}}')
        if i + 1 < len(array):
            memfile.write(', ')
            if (i + 1) % columns == 0:
                memfile.write('\\\n  ')
    memfile.write(' \\\n}\n')


def select_clock(
        memfile: TextIO,
        source: str,
        divider: str,
        comment: str = '',
        pll_wait: bool = True,
        prefix: str = '',
) -> None:
    """
    Switch clock source and divider.
    """
    if comment != '':
        memfile.write(f'{prefix}  // {comment}\n')
    if source == 'IPLL' and pll_wait:
        memfile.write(f'{prefix}  while ((MXC_GCR->ipll_ctrl & MXC_F_GCR_IPLL_CTRL_RDY) != '
                      'MXC_F_GCR_IPLL_CTRL_RDY) ;\n')
    memfile.write(f'{prefix}  MXC_GCR->pclkdiv = (MXC_GCR->pclkdiv & '
                  '~(MXC_F_GCR_PCLKDIV_CNNCLKDIV | MXC_F_GCR_PCLKDIV_CNNCLKSEL))\n'
                  f'{prefix}                     | MXC_S_GCR_PCLKDIV_CNNCLKDIV_{divider} | '
                  f'MXC_S_GCR_PCLKDIV_CNNCLKSEL_{source};\n')
