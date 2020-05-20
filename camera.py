###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Simulated camera data
"""


VSYNC_LEADIN = 10
VSYNC_HIGH = 5000
VSYNC_LOW = 2000
RETRACE = 318
FINAL = 10


def write(f, vsync, href, d, stretch=0):
    """
    Write `vsync`/`href` and data `d` to CSV file `f`.
    """
    f.write(f'{vsync},{href},0,{d:02x}\n')
    f.write(f'{vsync},{href},1,{d:02x}\n')
    f.write(f'{vsync},{href},1,{d:02x}\n')
    for _ in range(stretch):
        f.write(f'{vsync},{href},0,{d:02x}\n')
        f.write(f'{vsync},{href},0,{d:02x}\n')
        f.write(f'{vsync},{href},0,{d:02x}\n')


def header(f, leader=VSYNC_LEADIN, high=VSYNC_HIGH, low=VSYNC_LOW):
    """
    Write header (VSYNC low/high/low) to CSV file `f`.
    """
    f.write('vsync,href,pclk,d\n')

    # Lead-in VSYNC low
    for _ in range(leader):
        write(f, 0, 0, 0)

    # VSYNC high
    for _ in range(high):
        write(f, 1, 0, 0)

    # VSYNC low
    for _ in range(low):
        write(f, 0, 0, 0)


def finish_row(f, retrace=RETRACE):
    """
    Finish row by (HREF low) for CSV file `f`.
    """
    for _ in range(retrace):
        write(f, 0, 0, 0)


def finish_image(f, num=FINAL):
    """
    Finish image (VSYNC low/HREF low) for CSV file `f`.
    """
    for _ in range(num):
        write(f, 0, 0, 0)


def pixel(f, val):
    """
    Write pixel data `val` (HREF high) to CSV file `f`.
    """
    write(f, 0, 1, val, stretch=1)
