###################################################################################################
# Copyright (C) 2020-2021 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
RISC-V defines
"""

RISCV_FLASH = \
    '__attribute__ ((section(".rvflash_section")))\n'

RISCV_CODE_ORIGIN = 0x10001000
