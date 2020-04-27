#include "mxc_sys.h"
#include "bbfc_regs.h"
#include "fcr_regs.h"
#include "icc.h"

#define icache_enable() (MXC_ICC->ctrl |= MXC_F_ICC_CTRL_EN)
#define icache1_enable() (MXC_ICC1->ctrl |= MXC_F_ICC_CTRL_EN)
#define invalidate_icache1() { MXC_ICC1->invalidate = 1; while(!(MXC_ICC1->ctrl & MXC_F_ICC_CTRL_RDY)); }

#define LED_On(port)
#define LED_Off(port)

#define MXC_F_GCR_CLKCN_HIRC96M_RDY MXC_F_GCR_CLKCTRL_IPO_RDY
#define MXC_F_GCR_CLKCN_HIRC96M_EN MXC_F_GCR_CLKCTRL_IPO_EN

#define pass()

#define MXC_F_GCR_PERCKCN1_CPU1 0x80000000

extern void *_rvflash;
