###################################################################################################
# Copyright (C) 2019-2020 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Tornado CNN hardware constants - AI84, AI85
"""
import sys


dev = None


class Dev:
    """
    Metaclass for all hardware devices
    """


class DevAI84(Dev):
    """
    AI84 hardware constants
    """
    APB_BASE = 0x50100000
    MAX_LAYERS = 32
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
    MASK_WIDTH = 128
    MASK_OFFS = 128
    MCNT_SAD_OFFS = 8
    MCNT_MAX_OFFS = 0

    C_TRAM_BASE = C_CNN_BASE + 0x800
    C_MRAM_BASE = C_CNN_BASE + 0x4800
    C_BRAM_BASE = C_CNN_BASE + 0xC800
    C_SRAM_BASE = C_CNN_BASE + 0x10000

    C_GROUP_OFFS = 0x100000

    INSTANCE_SIZE = 1024  # x32
    INSTANCE_SHIFT = 12
    MEM_SIZE = INSTANCE_SIZE * P_NUMPRO * P_NUMGROUPS // P_SHARED  # x32
    MAX_CHANNELS = MAX_PROC

    BIAS_DIV = 1


class DevAI85(Dev):
    """
    AI85 hardware constants
    """
    APB_BASE = 0x50000000
    MAX_LAYERS = 32
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
    MASK_WIDTH = 768
    MASK_OFFS = 1024
    MCNT_SAD_OFFS = 16
    MCNT_MAX_OFFS = 0

    C_BRAM_BASE = C_CNN_BASE + 0x8000
    C_TRAM_BASE = C_CNN_BASE + 0x10000
    C_MRAM_BASE = C_CNN_BASE + 0x80000
    C_SRAM_BASE = C_CNN_BASE + 0x300000

    C_GROUP_OFFS = 0x400000

    INSTANCE_SIZE = 2048  # x32
    INSTANCE_SHIFT = 13
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


def get_device(
        device,
):
    """
    Change implementation configuration to match the AI84 or AI85, depending on the `device`
    integer input value.
    """
    print(f'Configuring device: AI{device}.')

    if device == 84:
        d = DevAI84()
    elif device == 85:
        d = DevAI85()
    elif device == 86:
        d = DevAI85()  # For now, no differences from AI85
    else:
        print(f'Unknown device code `{device}`')
        sys.exit(1)

    return d
