###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Tornado CNN hardware constants - AI85, AI87, CMSIS-NN
"""
from . import devices
from .eprint import eprint

dev = None


class Dev:
    """
    Metaclass for all hardware devices
    """
    device = 0
    partnum = ''

    SUPPORT_STREAM_BIAS = False
    SUPPORT_DEPTHWISE = False
    SUPPORT_CALCX4 = False
    SUPPORT_GCFR = False
    SUPPORT_PIPELINE = False
    SUPPORT_PLL = False
    SUPPORT_BINARY_WEIGHTS = False
    SUPPORT_LINK_LAYER = False
    SUPPORT_READ_AHEAD = False
    SUPPORT_MULTIPASS_STRIDE = False
    SUPPORT_KERNEL_BYPASS = False
    SUPPORT_ARBITRARY_PADDING = False
    SUPPORT_ARBITRARY_STRIDE = False
    SUPPORT_ARBITRARY_OUTPUT_WIDTH = False
    REQUIRE_REG_CLEAR = False
    REQUIRE_SEMA_LPWKEN = False
    REQUIRE_ONESHOT_CLEAR = True
    REQUIRE_NEW_STREAMING = False
    REQUIRE_FIFO_CPL = True
    EMULATE_ELTWISE_MP = False
    USE_PROCESSORS = True
    MODERN_SIM = False

    MAX_DILATION = 1
    MAX_POOL_DILATION = 1

    MASK_INSTANCES = MASK_INSTANCES_EACH = 1
    C_SRAM_BASE = C_GROUP_OFFS = INSTANCE_SIZE = INSTANCE_COUNT = INSTANCE_WIDTH = 0
    MASK_WIDTH_SMALL = MASK_WIDTH_LARGE = P_NUMPRO = 0  # These will be overridden by child classes
    IPO_SPEED = 100
    APB_SPEED = IPO_SPEED // 2
    PLL_SPEED = 0

    def mask_width(self, proc):
        """
        Returns the number of kernels (x9 bytes) for processor `proc`.
        """
        return self.MASK_WIDTH_LARGE if proc % self.P_NUMPRO == 0 else self.MASK_WIDTH_SMALL

    def datainstance_from_addr(self, addr):
        """
        Unpack the address `addr` into an individual memory instance and offset and
        return group, processor, instance, and offset.
        """
        addr -= self.C_SRAM_BASE
        group = addr // self.C_GROUP_OFFS
        addr %= self.C_GROUP_OFFS
        proc = addr // (self.INSTANCE_SIZE*16)
        addr %= self.INSTANCE_SIZE*16
        addr //= 4  # Switch to 32-bit word address
        mem = addr // (self.INSTANCE_WIDTH * 4 // self.INSTANCE_COUNT)
        addr %= self.INSTANCE_WIDTH * 4 // self.INSTANCE_COUNT
        return group, proc, mem, addr

    def __str__(self):
        return self.__class__.__name__


class DevCMSISNN(Dev):
    """
    CMSIS limitations
    """
    device = devices.CMSISNN
    partnum = 'CMSIS-NN'

    SUPPORT_ARBITRARY_PADDING = True
    SUPPORT_ARBITRARY_STRIDE = True
    SUPPORT_ARBITRARY_OUTPUT_WIDTH = True
    USE_PROCESSORS = False

    APB_BASE = 0
    MAX_LAYERS = 256
    MAX_ROW_COL = 2**16
    MEM_SIZE = 2**32
    BIAS_DIV = 128

    def __str__(self):
        return self.__class__.__name__


class DevAI85(Dev):
    """
    AI85 hardware constants
    """
    device = 85
    partnum = 'MAX78000'

    SUPPORT_GCFR = True
    APB_BASE = 0x50000000
    MAX_LAYERS = 32
    MAX_STREAM_LAYERS = 8
    MAX_START_LAYER = 0
    C_CNN = 4
    C_FIFO_BASE = 0
    C_CNN_BASE = 0x100000
    P_NUMGROUPS = 4
    P_NUMGROUPS_ALL = P_NUMGROUPS
    P_NUMPRO = 16  # Processors per group
    P_SHARED = 4  # Processors sharing a data memory
    MAX_PROC = P_NUMPRO * P_NUMGROUPS
    MAX_ROW_COL = 1024

    # Per-layer registers
    LREG_RCNT = 0
    LREG_CCNT = 1
    LREG_ONED = 2
    LREG_PRCNT = 3
    LREG_PCCNT = 4
    LREG_STRIDE = 5
    LREG_WPTR_BASE = 6
    LREG_WPTR_TOFFS = 7
    LREG_WPTR_MOFFS = 8
    LREG_WPTR_CHOFFS = 9
    LREG_RPTR_BASE = 10
    LREG_LCTL = 11
    LREG_MCNT = 12
    LREG_TPTR = 13
    LREG_ENA = 14
    LREG_POST = 15
    LREG_STREAM1 = 16
    LREG_STREAM2 = 17
    LREG_FMAX = 18
    # 19 reserved for REG_IFRM
    LREG_LCTL2 = 20
    MAX_LREG = LREG_LCTL2
    LREG_RFU = None

    PAD_CNT_OFFS = 16
    XPCH_MAX_OFFS = 8
    MAX_CNT_BITS = 10
    MAX_PTR_BITS = 17
    MAX_TPTR_BITS = 12
    MAX_ISVAL_BITS = 14
    MAX_DSVAL2_BITS = 12
    MAX_FBUF_BITS = 17
    MAX_IFRM_BITS = 20
    MAX_WPTRINC_BITS = 4

    # Global registers
    REG_CTL = 0
    REG_SRAM = 1
    REG_LCNT_MAX = 2
    REG_SRAM_TEST = 3
    REG_IFRM = 612
    REG_MLAT = 1024

    FIFO_CTL = 0
    FIFO_STAT = 1
    FIFO_REG = 2

    AON_CTL = 1024

    READY_SEL = 0
    PIPELINE_READY_SEL = READY_SEL
    FIFO_READY_SEL = 0
    AON_READY_SEL = 0

    DEFAULT_WEIGHT_BITS = 8
    ACTIVATION_BITS = 8
    TRAM_SIZE = 3072
    TRAM_OFFS = 4096
    BIAS_SIZE = 512
    MASK_WIDTH_SMALL = 768
    MASK_WIDTH_LARGE = 768
    MASK_OFFS = 1024
    MCNT_SAD_OFFS = 16
    MCNT_MAX_OFFS = 0

    C_BRAM_BASE = C_CNN_BASE + 0x8000
    C_TRAM_BASE = C_CNN_BASE + 0x10000
    C_MRAM_BASE = C_CNN_BASE + 0x80000
    C_SRAM_BASE = C_CNN_BASE + 0x300000

    C_GROUP_OFFS = 0x400000

    INSTANCE_SIZE = INSTANCE_WIDTH = 2048  # x32
    INSTANCE_COUNT = 1
    INSTANCE_SHIFT = 13
    WRITE_PTR_SHIFT = 13
    MEM_SIZE = INSTANCE_SIZE * P_NUMPRO * P_NUMGROUPS // P_SHARED  # x32
    MAX_CHANNELS = 16 * MAX_PROC  # 16 x expansion

    FRAME_SIZE_MAX = 2**21  # x * y * multipass, from cnn_ctl.sv P_FRMABITS

    BIAS_DIV = 128

    FAST_FIFO_BASE = 0x400c0400
    FAST_FIFO_CR = 0  # Control register
    FAST_FIFO_SR = 1  # Status register
    FAST_FIFO_IE = 2  # Interrupt enable register
    FAST_FIFO_IS = 3  # Interrupt status (flag) register
    FAST_FIFO_DR = 4  # Data register
    FAST_FIFO_DMA = 5  # DMA register (reserved function, not yet supported)

    EMULATE_ELTWISE_MP = True

    FLASH_SIZE = 0x8000
    RISCV_SRAM_ORIGIN = 0x2001C000

    # Cycles
    C_START = 4
    C_PAD = 2

    def __str__(self):
        return self.__class__.__name__


class DevAI87(Dev):
    """
    AI85 hardware constants
    """
    device = 87
    partnum = 'MAX78002'

    APB_BASE = 0x50000000
    MAX_LAYERS = 128
    MAX_STREAM_LAYERS = 8
    MAX_START_LAYER = MAX_LAYERS - 1
    C_CNN = 0x40000
    C_FIFO_BASE = 0
    C_CNN_BASE = 0x1000000
    P_NUMGROUPS = 4
    P_NUMGROUPS_ALL = P_NUMGROUPS
    P_NUMPRO = 16  # Processors per group
    P_SHARED = 4  # Processors sharing a data memory
    MAX_PROC = P_NUMPRO * P_NUMGROUPS
    MAX_ROW_COL = 2048

    # Per-layer registers
    LREG_OFFS = 0x100

    LREG_NXTLYR = 0
    LREG_RCNT = 1
    LREG_CCNT = 2
    LREG_ONED = 3
    LREG_PRCNT = 4
    LREG_PCCNT = 5
    LREG_STRIDE = 6
    LREG_WPTR_BASE = 7
    LREG_WPTR_TOFFS = 8
    LREG_WPTR_MOFFS = 9
    LREG_WPTR_CHOFFS = 10
    LREG_RPTR_BASE = 11
    LREG_LCTL = 12
    LREG_LCTL2 = 13
    LREG_MCNT1 = 14
    LREG_MCNT2 = 15
    LREG_OCHAN = 16
    LREG_TPTR = 17
    LREG_ENA = 18
    LREG_POST = 19
    LREG_RFU = None
    MAX_LREG = LREG_POST
    LREG_STREAM1 = 0x2000
    MIN_STREAM_LREG = LREG_STREAM1
    LREG_STREAM2 = 0x2008
    LREG_FMAX = 0x2010
    MAX_STREAM_LREG = LREG_FMAX

    CTL_PIPELINE_OFFS = 5
    PAD_CNT_OFFS = 13
    PAD_ENA_OFFS = 15
    CNT_DIFF_OFFS = 16
    MAX_CNT_BITS = 11
    CNT_INC_OFFS = 4
    MP_STRIDE_OFFS = 4
    XPCH_MAX_OFFS = 12
    MAX_PTR_BITS = 21
    MAX_TPTR_BITS = 14
    MAX_ISVAL_BITS = 15
    MAX_DSVAL2_BITS = 13
    MAX_FBUF_BITS = 18
    MAX_IFRM_BITS = 22
    MAX_WPTRINC_BITS = 8

    # Global registers
    REG_CTL = 0
    REG_SRAM = 1
    REG_LCNT_MAX = 2
    REG_SRAM_TEST = 3
    REG_IFRM = 0x42018
    REG_MLAT = 0x44000

    FIFO_CTL = 0
    FIFO_STAT = 1
    FIFO_REG = 2

    AON_CTL = 1024

    READY_SEL = 0
    PIPELINE_READY_SEL = 3
    FIFO_READY_SEL = 0
    AON_READY_SEL = 0

    DEFAULT_WEIGHT_BITS = 8
    ACTIVATION_BITS = 8
    TRAM_SIZE = 12288
    TRAM_OFFS = 16384
    BIAS_SIZE = 2048
    MASK_INSTANCES = 8
    MASK_INSTANCES_EACH = 4
    MASK_WIDTH_SMALL = MASK_INSTANCES_EACH * 1024
    MASK_WIDTH_LARGE = MASK_WIDTH_SMALL + MASK_INSTANCES_EACH * 256
    MASK_OFFS = 8192
    RD_AHEAD_OFFS = 17
    CPRIME_MAX_OFFS = 18
    RPRIME_MAX_OFFS = 22

    C_BRAM_BASE = C_CNN_BASE + 0x180000
    C_TRAM_BASE = C_CNN_BASE + 0x200000
    C_MRAM_BASE = C_CNN_BASE + 0x400000
    C_SRAM_BASE = C_CNN_BASE + 0x800000

    C_GROUP_OFFS = 0x1000000

    INSTANCE_SIZE = 8192  # x32 (includes empty space)
    INSTANCE_WIDTH = 5120  # x32 (true memory size)
    INSTANCE_COUNT = 5
    INSTANCE_SHIFT = 17
    WRITE_PTR_SHIFT = 15
    MEM_SIZE = INSTANCE_WIDTH * P_NUMPRO * P_NUMGROUPS // P_SHARED  # x32
    MAX_CHANNELS = 16 * MAX_PROC  # 16 x expansion

    FRAME_SIZE_MAX = 2**21  # x * y * multipass, from cnn_ctl.sv P_FRMABITS

    BIAS_DIV = 128

    FAST_FIFO_BASE = 0x400c0400
    FAST_FIFO_CR = 0  # Control register
    FAST_FIFO_SR = 1  # Status register
    FAST_FIFO_IE = 2  # Interrupt enable register
    FAST_FIFO_IS = 3  # Interrupt status (flag) register
    FAST_FIFO_DR = 4  # Data register
    FAST_FIFO_DMA = 5  # DMA register (reserved function, not yet supported)

    SUPPORT_STREAM_BIAS = True
    SUPPORT_DEPTHWISE = True
    SUPPORT_CALCX4 = True
    SUPPORT_PIPELINE = True
    SUPPORT_PLL = True
    SUPPORT_BINARY_WEIGHTS = True
    SUPPORT_LINK_LAYER = True
    SUPPORT_READ_AHEAD = True
    SUPPORT_MULTIPASS_STRIDE = True
    SUPPORT_KERNEL_BYPASS = True
    REQUIRE_REG_CLEAR = True
    REQUIRE_SEMA_LPWKEN = True
    REQUIRE_ONESHOT_CLEAR = False
    REQUIRE_NEW_STREAMING = True
    REQUIRE_FIFO_CPL = False

    MAX_POOL_DILATION = 2**4

    SUPPORT_GCFR = True
    MODERN_SIM = True

    FLASH_SIZE = 0x28000
    RISCV_SRAM_ORIGIN = 0x2005C000

    # Cycles
    C_START = 4
    C_PAD = 2

    # PLL Speed in MHz
    PLL_SPEED = 240

    def __str__(self):
        return self.__class__.__name__


def lreg_addr(group, reg, layer=0):
    """
    Return the address of a layer register given group `group`, register `reg`, and
    layer `layer`.
    """
    if hasattr(dev, 'LREG_OFFS'):
        if reg <= dev.MAX_LREG:
            addr = dev.C_GROUP_OFFS*group + dev.C_CNN_BASE \
                + dev.C_CNN*4 + reg*4 + layer*dev.LREG_OFFS
        else:
            addr = dev.C_GROUP_OFFS*group + dev.C_CNN_BASE \
                + dev.C_CNN*4 + reg*4 + layer*4
    else:
        addr = dev.C_GROUP_OFFS*group + dev.C_CNN_BASE \
            + dev.C_CNN*4 + reg*4 * dev.MAX_LAYERS + layer*4

    return addr


def ctl_addr(group, reg):
    """
    Return the address of control register `reg` in group `group`.
    """
    return dev.C_GROUP_OFFS*group + dev.C_CNN_BASE + reg*4


def get_device(
        device,
):
    """
    Change implementation configuration to match, depending on the `device`
    integer input value.
    """
    if device == 85:
        d = DevAI85()
    elif device == 87:
        d = DevAI87()
    elif device == devices.CMSISNN:
        d = DevCMSISNN()
    else:
        eprint(f'Unknown device code `{device}`')

    print('Configuring device:', d.partnum)

    return d
