/*******************************************************************************
 * Copyright (C) 2020 Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
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
 *
 ******************************************************************************/

/* RVC :: ????? :: ROM        (rx) : ORIGIN = 0x00000000, LENGTH = 0x00001000 */ /* 4kB ROM */
/* RVC :: ????? :: FLASH1     (rx) : ORIGIN = 0x10080000, LENGTH = 0x00280000 */ /* 2.5MB Flash */
MEMORY {
    FLASH0  (rwx) : ORIGIN = 0x10000000, LENGTH = 0x00001000
    FLASH1  (rwx) : ORIGIN = 0x10001000, LENGTH = 0x0027F000
    SRAM6I  (rwx) : ORIGIN = 0x20050000, LENGTH = 0x00001000 /*  4kB SRAM6 :: 4KB Start used for const */
    SRAM6   (rwx) : ORIGIN = 0x20051000, LENGTH = 0x0000B000 /* 44kB SRAM6 :: SRAM6_0=8KB*/
    SRAM7   (rwx) : ORIGIN = 0x2005C000, LENGTH = 0x00004000 /* 16kB SRAM7 */
}

SECTIONS {
    .rvflash :
    {
      _rvflash = .;
        KEEP(*(.rvflash_section))   /* program code */
      _ervflash = .;
    } > FLASH1
    
    .text :
    {
        _text = .;
        KEEP(*(.isr_vector))
        *(.text*)    /* program code */

        KEEP(*(.init))
        KEEP(*(.fini))

        /* C++ Exception handling */
        KEEP(*(.eh_frame*))
        _etext = .;
    } > SRAM7

    .consts :
    {
        *(.rodata*)  /* read-only data: "const" */
        *(.srodata*) /* read-only data: "const" */
    } > SRAM6I 

    .sram3_code :
    {
        . = ALIGN(16);
        _sran_code = .;
        *(.sram3_code_section)
        _esran_code = .;
    } > SRAM7 


    /* it's used for C++ exception handling      */
    /* we need to keep this to avoid overlapping */
    .ARM.exidx :
    {
        __exidx_start = .;
        *(.ARM.exidx*)
        __exidx_end = .;
    } > SRAM7

    /* short/global data section */
    .sdata :
    {
        _sdata = ALIGN(., 4);
        __sdata_start = .; 
        PROVIDE(__global_pointer$ = . + 0x800);        
        *(.sdata*)  /*read-write initialized data: initialized global variable*/

        _esdata = ALIGN(., 4);
        __sdata_end = .;
	
    } > SRAM6 AT>SRAM6I
    __load_sdata = LOADADDR(.sdata);

    .data :
    {
        _data = ALIGN(., 4);
        __data_start__ = . ;            /* Standard newlib definition. */
	
        *(.data*)           /*read-write initialized data: initialized global variable*/
        *(.spix_config*)    /* SPIX configuration functions need to be run from SRAM7 */

        /* These array sections are used by __libc_init_array to call static C++ constructors */
        . = ALIGN(4);
        /* preinit data */
        PROVIDE_HIDDEN (__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN (__preinit_array_end = .);

        . = ALIGN(4);
        /* init data */
        PROVIDE_HIDDEN (__init_array_start = .);
        KEEP(*(SORT(.init_array.*)))
        KEEP(*(.init_array))
        PROVIDE_HIDDEN (__init_array_end = .);

        . = ALIGN(4);
        /* finit data */
        PROVIDE_HIDDEN (__fini_array_start = .);
        KEEP(*(SORT(.fini_array.*)))
        KEEP(*(.fini_array))
        PROVIDE_HIDDEN (__fini_array_end = .);

        _edata = ALIGN(., 4);
       __data_end__ = . ;                /* Standard newlib definition. */
    } > SRAM6 AT>SRAM6I
    __load_data = LOADADDR(.data);

    .sbss :
    {
        . = ALIGN(4);
        __sbss_start = .;                 /* Standard newlib definition. */
	__sbss_start__ = .;                 /* Standard newlib definition. */
        _sbss = .;
        *(.sbss*)     /*read-write zero initialized data: uninitialzed global variable*/
        *(COMMON)
        _esbss = ALIGN(., 4);
       __sbss_end__ = .;                /* Standard newlib definition. */
       __sbss_end = .;                    /* Standard newlib definition. */
    } > SRAM6

    .bss :
    {
        . = ALIGN(4);
        __bss_start = .;                 /* Standard newlib definition. */
	__bss_start__ = .;                 /* Standard newlib definition. */
        _bss = .;
        *(.bss*)     /*read-write zero initialized data: uninitialzed global variable*/
        *(COMMON)
        _ebss = ALIGN(., 4);
        __bss_end__ = .;                /* Standard newlib definition. */
	__bss_end = .;                    /* Standard newlib definition. */
     } > SRAM6

    /* Set stack top to end of RAM, and stack limit move down by
     * size of stack_dummy section */
    __StackTop = ORIGIN(SRAM6) + LENGTH(SRAM6);
    __StackLimit = __StackTop - SIZEOF(.stack_dummy);

    /* .stack_dummy section doesn't contains any symbols. It is only
     * used for linker to calculate size of stack sections, and assign
     * values to stack symbols later */
    .stack_dummy (COPY):
    {
        *(.stack*)
    } > SRAM6

    .heap (COPY):
    {
        . = ALIGN(4);
        *(.heap*)
        __HeapLimit = ABSOLUTE(__StackLimit);
    } > SRAM6

    PROVIDE(__stack = __StackTop);

    /* Check if data + heap + stack exceeds RAM limit */
    ASSERT(__StackLimit >= _ebss, "region RAM overflowed with stack")
}
