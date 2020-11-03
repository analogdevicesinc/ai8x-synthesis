/*******************************************************************************
* Copyright (C) Maxim Integrated Products, Inc., All rights Reserved.
*
* This software is protected by copyright laws of the United States and
* of foreign countries. This material may also be protected by patent laws
* and technology transfer regulations of the United States and of foreign
* countries. This software is furnished under a license agreement and/or a
* nondisclosure agreement and may only be used or reproduced in accordance
* with the terms of those agreements. Dissemination of this information to
* any party or parties not specified in the license agreement and/or
* nondisclosure agreement is expressly prohibited.
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
* OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
* OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of Maxim Integrated
* Products, Inc. shall not be used except as stated in the Maxim Integrated
* Products, Inc. Branding Policy.
*
* The mere transfer of this software does not imply any licenses
* of trade secrets, proprietary technology, copyrights, patents,
* trademarks, maskwork rights, or any other form of intellectual
* property whatsoever. Maxim Integrated Products, Inc. retains all
* ownership rights.
*******************************************************************************/

#include <stdlib.h>
#include <stdint.h>
#include "global_functions.h" // For RTL Simulation


void memset32_known(uint32_t *dst, int ch, int n)
{
  uint64_t v = (uint32_t) dst + ch;
  while (n > 0) {
    *dst++ = (uint32_t) (((n % 7) * v + ~n) & 0xffffffff);
    n--;
  }
}

int memtest32_known(uint32_t *dst, int ch, int n)
{
  uint64_t v = (uint32_t) dst + ch;
  while (n > 0) {
    if (*dst++ != (uint32_t) (((n % 7) * v + ~n) & 0xffffffff)) return 0;
    n--;
  }
  return 1;
}

void load_input(void)
{
  memset32_known((uint32_t *) 0x50400000,  0, 8192);
  memset32_known((uint32_t *) 0x50408000,  4, 8192);
  memset32_known((uint32_t *) 0x50410000,  8, 8192);
  memset32_known((uint32_t *) 0x50418000, 12, 8192);

  memset32_known((uint32_t *) 0x50800000, 16, 8192);
  memset32_known((uint32_t *) 0x50808000, 20, 8192);
  memset32_known((uint32_t *) 0x50810000, 24, 8192);
  memset32_known((uint32_t *) 0x50818000, 28, 8192);

  memset32_known((uint32_t *) 0x50c00000, 32, 8192);
  memset32_known((uint32_t *) 0x50c08000, 36, 8192);
  memset32_known((uint32_t *) 0x50c10000, 40, 8192);
  memset32_known((uint32_t *) 0x50c18000, 44, 8192);

  memset32_known((uint32_t *) 0x51000000, 48, 8192);
  memset32_known((uint32_t *) 0x51008000, 52, 8192);
  memset32_known((uint32_t *) 0x51010000, 56, 8192);
  memset32_known((uint32_t *) 0x51018000, 60, 8192);
}

int cnn_load(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x50100000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50100004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x50500000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50500004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x50900000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50900004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x50d00000) = 0x00100008; // Stop SM
  *((volatile uint32_t *) 0x50d00004) = 0x0000040e; // SRAM control

  load_input(); // Load data input

  return 1;
}

int cnn_check(void)
{
  if (!memtest32_known((uint32_t *) 0x50400000,  0, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50408000,  4, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50410000,  8, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50418000, 12, 8192)) return 0;

  if (!memtest32_known((uint32_t *) 0x50800000, 16, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50808000, 20, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50810000, 24, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50818000, 28, 8192)) return 0;

  if (!memtest32_known((uint32_t *) 0x50c00000, 32, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50c08000, 36, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50c10000, 40, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x50c18000, 44, 8192)) return 0;

  if (!memtest32_known((uint32_t *) 0x51000000, 48, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x51008000, 52, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x51010000, 56, 8192)) return 0;
  if (!memtest32_known((uint32_t *) 0x51018000, 60, 8192)) return 0;
  return 1;
}

int main(void)
{
  icache_enable();

  *((volatile uint32_t *) 0x40000c00) = 0x00000001; // Set TME
  *((volatile uint32_t *) 0x40006c04) = 0x000001a0; // 96M trim
  *((volatile uint32_t *) 0x40000c00) = 0x00000000; // Clear TME

  MXC_GCR->clkcn |= MXC_F_GCR_CLKCN_HIRC96M_EN; // Enable 96M
  while ((MXC_GCR->clkcn & MXC_F_GCR_CLKCN_HIRC96M_RDY) == 0) ; // Wait for 96M
  MXC_GCR->clkcn |= MXC_S_GCR_CLKCN_CLKSEL_HIRC96; // Select 96M

  // Reset all domains, restore power to CNN
  MXC_BBFC->reg3 = 0xf; // Reset
  MXC_BBFC->reg1 = 0xf; // Mask memory
  MXC_BBFC->reg0 = 0xf; // Power
  MXC_BBFC->reg2 = 0x0; // Iso
  MXC_BBFC->reg3 = 0x0; // Reset

  MXC_GCR->pckdiv = 0x00010000; // CNN clock 96M div 2
  MXC_GCR->perckcn &= ~0x2000000; // Enable CNN clock

  if (!cnn_load()) { fail(); pass(); return 0; }

  if (!cnn_check()) fail();
  // Disable power to CNN
  MXC_BBFC->reg3 = 0xf; // Reset
  MXC_BBFC->reg1 = 0x0; // Mask memory
  MXC_BBFC->reg0 = 0x0; // Power
  MXC_BBFC->reg2 = 0xf; // Iso
  MXC_BBFC->reg3 = 0x0; // Reset

  pass();
  return 0;
}

