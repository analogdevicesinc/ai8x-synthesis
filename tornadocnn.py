###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Tornado CNN hardware constants - AI84, AI85, AI87
"""
import sys

import devices
from eprint import eprint


dev = None


class Dev:
    """
    Metaclass for all hardware devices
    """
    FIX_STREAM_BIAS = False
    MASK_WIDTH_SMALL = MASK_WIDTH_LARGE = P_NUMPRO = 0  # These will be overridden by child classes

    def mask_width(self, proc):
        """
        Returns the number of kernels (x9 bytes) for processor `proc`.
        """
        return self.MASK_WIDTH_LARGE if proc % self.P_NUMPRO == 0 else self.MASK_WIDTH_SMALL

    def __init__(self, part_no):
        self.part_no = part_no

    def __str__(self):
        return self.__class__.__name__


class DevAI84(Dev):
    """
    AI84 hardware constants
    """
    APB_BASE = 0x50100000
    MAX_LAYERS = 32
    MAX_STREAM_LAYERS = None
    MAX_START_LAYER = 0
    C_CNN = 4
    C_CNN_BASE = 0

    P_NUMGROUPS = 4
    P_NUMPRO = 16  # Processors per group
    P_SHARED = 4  # Processors sharing a data memory
    MAX_PROC = P_NUMPRO * P_NUMGROUPS
    MAX_ROW_COL = 256

    # Per-layer registers
    LREG_RCNT = 0
    LREG_CCNT = 1
    LREG_RFU = 2
    LREG_PRCNT = 3
    LREG_PCCNT = 4
    LREG_STRIDE = 5
    LREG_WPTR_BASE = 6
    LREG_WPTR_OFFS = 7
    LREG_RPTR_BASE = 8
    LREG_LCTL = 9
    LREG_MCNT = 10
    LREG_TPTR = 11
    LREG_ENA = 12
    MAX_LREG = LREG_ENA

    MAX_PTR_BITS = 17
    MAX_TPTR_BITS = 12

    # Global registers
    REG_CTL = 0
    REG_SRAM = 1
    REG_LCNT_MAX = 2

    READY_SEL = 0x03

    DEFAULT_WEIGHT_BITS = 8
    ACTIVATION_BITS = 8

    TRAM_SIZE = 256
    TRAM_OFFS = 256
    BIAS_SIZE = 256
    MASK_WIDTH_SMALL = 128
    MASK_WIDTH_LARGE = 128
    MASK_OFFS = 128
    MCNT_SAD_OFFS = 8
    MCNT_MAX_OFFS = 0

    C_TRAM_BASE = C_CNN_BASE + 0x800
    C_MRAM_BASE = C_CNN_BASE + 0x4800
    C_BRAM_BASE = C_CNN_BASE + 0xC800
    C_SRAM_BASE = C_CNN_BASE + 0x10000

    C_GROUP_OFFS = 0x100000

    INSTANCE_SIZE = INSTANCE_WIDTH = 1024  # x32
    INSTANCE_SHIFT = 12
    WRITE_PTR_SHIFT = 12
    MEM_SIZE = INSTANCE_SIZE * P_NUMPRO * P_NUMGROUPS // P_SHARED  # x32
    MAX_CHANNELS = MAX_PROC

    FRAME_SIZE_MAX = 2**14  # x * y * multipass

    BIAS_DIV = 1

    # Cycles
    C_START = 4
    C_POOL = 3

    def __str__(self):
        return self.__class__.__name__


class DevAI85(Dev):
    """
    AI85 hardware constants
    """
    APB_BASE = 0x50000000
    MAX_LAYERS = 32
    MAX_STREAM_LAYERS = 8
    MAX_START_LAYER = 0
    C_CNN = 4
    C_FIFO_BASE = 0
    C_CNN_BASE = 0x100000
    P_NUMGROUPS = 4
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
    MAX_CNT_BITS = 10
    MAX_PTR_BITS = 17
    MAX_TPTR_BITS = 12
    MAX_ISVAL_BITS = 14
    MAX_DSVAL2_BITS = 12
    MAX_FBUF_BITS = 17
    MAX_IFRM_BITS = 20

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

    FIX_STREAM_BIAS = True

    # Cycles
    C_START = 4
    C_PAD = 2

    def __str__(self):
        return self.__class__.__name__


class DevAI87(Dev):
    """
    AI85 hardware constants
    """
    APB_BASE = 0x50000000
    MAX_LAYERS = 128
    MAX_STREAM_LAYERS = 8
    MAX_START_LAYER = MAX_LAYERS - 1
    C_CNN = 0x40000
    C_FIFO_BASE = 0
    C_CNN_BASE = 0x1000000
    P_NUMGROUPS = 4
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
    LREG_MCNT = 14
    LREG_TPTR = 15
    LREG_ENA = 16
    LREG_POST = 17
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
    MAX_PTR_BITS = 21
    MAX_TPTR_BITS = 14
    MAX_ISVAL_BITS = 15
    MAX_DSVAL2_BITS = 13
    MAX_FBUF_BITS = 18
    MAX_IFRM_BITS = 22

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
    FIFO_READY_SEL = 0
    AON_READY_SEL = 0

    DEFAULT_WEIGHT_BITS = 8
    ACTIVATION_BITS = 8
    TRAM_SIZE = 12288
    TRAM_OFFS = 16384
    BIAS_SIZE = 2048
    MASK_WIDTH_SMALL = 4096
    MASK_WIDTH_LARGE = 5120
    MASK_OFFS = 8192
    MCNT_SAD_OFFS = 16
    MCNT_MAX_OFFS = 0
    OCHAN_CNT_OFFS = 17
    CPRIME_MAX_OFFS = 18
    RPRIME_MAX_OFFS = 22

    C_BRAM_BASE = C_CNN_BASE + 0x180000
    C_TRAM_BASE = C_CNN_BASE + 0x200000
    C_MRAM_BASE = C_CNN_BASE + 0x400000
    C_SRAM_BASE = C_CNN_BASE + 0x800000

    C_GROUP_OFFS = 0x1000000

    INSTANCE_SIZE = 8192  # x32 (includes empty space)
    INSTANCE_WIDTH = 6144  # x32 (true memory size)
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

    FIX_STREAM_BIAS = True

    # Cycles
    C_START = 4
    C_PAD = 2

    def __str__(self):
        return self.__class__.__name__


def lreg_addr(group, reg, layer=0):
    """
    Return the address of a layer register given group `group`, register `reg`, and
    layer `layer`.
    """
    if hasattr(dev, 'LREG_OFFS'):
        addr = dev.C_GROUP_OFFS*group + dev.C_CNN_BASE \
            + dev.C_CNN*4 + reg*4 + layer*dev.LREG_OFFS
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
    part = devices.partnum(device)
    print('Configuring device:', part)

    if device == 84:
        d = DevAI84(part)
    elif device == 85:
        d = DevAI85(part)
    elif device == 87:
        d = DevAI87(part)
    else:
        eprint(f'Unknown device code `{device}`')
        sys.exit(1)

    return d
