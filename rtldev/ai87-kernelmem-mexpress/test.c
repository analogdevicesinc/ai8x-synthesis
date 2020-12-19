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


#define MASK_WIDTH_SMALL 4096
#define MASK_WIDTH_LARGE 5120


uint32_t *wptr;
uint8_t buffer[4];
int buffered;


void buf(uint8_t u)
{
  uint32_t val;
  buffer[buffered++] = u;

  if (buffered == 4) {
     val = ((uint32_t) buffer[0] << 24) | ((uint32_t) buffer[1] << 16) | ((uint32_t) buffer[2] << 8) | (uint32_t) buffer[3];
     *wptr++ = val;
     buffered = 0;
  }
}

void memset96_known(uint32_t *dst, int p, int n)
{
  uint64_t v = (uint32_t) dst + p;
  uint32_t w0, w1, w2;

  buffered = 0;
  wptr = dst;
  *((volatile uint8_t *) ((uint32_t) wptr + 1)) = 0x01; // Set address

  while (n > 0) {
    w0 = (p ^ (uint32_t) dst) & 0xff;
    buf((uint8_t) (w0 & 0xff));
    dst++;
    w1 = (uint32_t) (((n % 7) * v + ~n) & 0xffffffff);
    buf((uint8_t) ((w1 >> 24) & 0xff));
    buf((uint8_t) ((w1 >> 16) & 0xff));
    buf((uint8_t) ((w1 >> 8) & 0xff));
    buf((uint8_t) (w1 & 0xff));
    dst++;
    w2 = (uint32_t) dst ^ (uint32_t) n;
    buf((uint8_t) ((w2 >> 24) & 0xff));
    buf((uint8_t) ((w2 >> 16) & 0xff));
    buf((uint8_t) ((w2 >> 8) & 0xff));
    buf((uint8_t) (w2 & 0xff));
    dst++;
    dst++;
    n--;
  }
  if (buffered > 0) {
    while (buffered != 0) {
      buf(0);
    }
  }
}

int memtest96_known(uint32_t *dst, int p, int n)
{
  uint64_t v = (uint32_t) dst + p;
  while (n > 0) {
    if (*dst != ((p ^ (uint32_t) dst) & 0xff)) return 0;
    dst++;
    if (*dst != ((uint32_t) (((n % 7) * v + ~n) & 0xffffffff))) return 0;
    dst++;
    if (*dst != ((uint32_t) dst ^ (uint32_t) n)) return 0;
    dst++;
    if (*dst != 0) return 0;
    dst++;
    n--;
  }
  return 1;
}

void load_input(void)
{
  memset96_known((uint32_t *) 0x51400000,  0, MASK_WIDTH_LARGE);
  memset96_known((uint32_t *) 0x51420000,  1, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51440000,  2, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51460000,  3, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51480000,  4, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x514a0000,  5, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x514c0000,  6, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x514e0000,  7, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51500000,  8, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51520000,  9, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51540000, 10, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51560000, 11, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x51580000, 12, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x515a0000, 13, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x515c0000, 14, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x515e0000, 15, MASK_WIDTH_SMALL);

  memset96_known((uint32_t *) 0x52400000, 16, MASK_WIDTH_LARGE);
  memset96_known((uint32_t *) 0x52420000, 17, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52440000, 18, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52460000, 19, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52480000, 20, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x524a0000, 21, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x524c0000, 22, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x524e0000, 23, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52500000, 24, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52520000, 25, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52540000, 26, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52560000, 27, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x52580000, 28, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x525a0000, 29, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x525c0000, 30, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x525e0000, 31, MASK_WIDTH_SMALL);

  memset96_known((uint32_t *) 0x53400000, 32, MASK_WIDTH_LARGE);
  memset96_known((uint32_t *) 0x53420000, 33, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53440000, 34, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53460000, 35, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53480000, 36, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x534a0000, 37, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x534c0000, 38, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x534e0000, 39, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53500000, 40, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53520000, 41, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53540000, 42, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53560000, 43, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x53580000, 44, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x535a0000, 45, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x535c0000, 46, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x535e0000, 47, MASK_WIDTH_SMALL);

  memset96_known((uint32_t *) 0x54400000, 48, MASK_WIDTH_LARGE);
  memset96_known((uint32_t *) 0x54420000, 49, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54440000, 50, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54460000, 51, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54480000, 52, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x544a0000, 53, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x544c0000, 54, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x544e0000, 55, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54500000, 56, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54520000, 57, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54540000, 58, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54560000, 59, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x54580000, 60, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x545a0000, 61, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x545c0000, 62, MASK_WIDTH_SMALL);
  memset96_known((uint32_t *) 0x545e0000, 63, MASK_WIDTH_SMALL);
}

int cnn_load(void)
{
  *((volatile uint32_t *) 0x50001000) = 0x00000000; // AON control
  *((volatile uint32_t *) 0x51000000) = 0x00100008; // Stop SM, enable mexpress
  *((volatile uint32_t *) 0x51000004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x52000000) = 0x00100008; // Stop SM, enable mexpress
  *((volatile uint32_t *) 0x52000004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x53000000) = 0x00100008; // Stop SM, enable mexpress
  *((volatile uint32_t *) 0x53000004) = 0x0000040e; // SRAM control

  *((volatile uint32_t *) 0x54000000) = 0x00100008; // Stop SM, enable mexpress
  *((volatile uint32_t *) 0x54000004) = 0x0000040e; // SRAM control

  load_input(); // Load data input

  return 1;
}

int cnn_check(void)
{
  if (!memtest96_known((uint32_t *) 0x51400000,  0, MASK_WIDTH_LARGE)) return 0;
  if (!memtest96_known((uint32_t *) 0x51420000,  1, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51440000,  2, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51460000,  3, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51480000,  4, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x514a0000,  5, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x514c0000,  6, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x514e0000,  7, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51500000,  8, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51520000,  9, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51540000, 10, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51560000, 11, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x51580000, 12, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x515a0000, 13, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x515c0000, 14, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x515e0000, 15, MASK_WIDTH_SMALL)) return 0;

  if (!memtest96_known((uint32_t *) 0x52400000, 16, MASK_WIDTH_LARGE)) return 0;
  if (!memtest96_known((uint32_t *) 0x52420000, 17, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52440000, 18, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52460000, 19, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52480000, 20, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x524a0000, 21, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x524c0000, 22, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x524e0000, 23, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52500000, 24, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52520000, 25, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52540000, 26, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52560000, 27, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x52580000, 28, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x525a0000, 29, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x525c0000, 30, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x525e0000, 31, MASK_WIDTH_SMALL)) return 0;

  if (!memtest96_known((uint32_t *) 0x53400000, 32, MASK_WIDTH_LARGE)) return 0;
  if (!memtest96_known((uint32_t *) 0x53420000, 33, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53440000, 34, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53460000, 35, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53480000, 36, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x534a0000, 37, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x534c0000, 38, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x534e0000, 39, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53500000, 40, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53520000, 41, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53540000, 42, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53560000, 43, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x53580000, 44, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x535a0000, 45, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x535c0000, 46, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x535e0000, 47, MASK_WIDTH_SMALL)) return 0;

  if (!memtest96_known((uint32_t *) 0x54400000, 48, MASK_WIDTH_LARGE)) return 0;
  if (!memtest96_known((uint32_t *) 0x54420000, 49, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54440000, 50, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54460000, 51, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54480000, 52, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x544a0000, 53, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x544c0000, 54, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x544e0000, 55, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54500000, 56, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54520000, 57, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54540000, 58, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54560000, 59, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x54580000, 60, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x545a0000, 61, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x545c0000, 62, MASK_WIDTH_SMALL)) return 0;
  if (!memtest96_known((uint32_t *) 0x545e0000, 63, MASK_WIDTH_SMALL)) return 0;

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

