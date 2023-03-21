###################################################################################################
# Copyright (C) 2019-2023 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Routines to read and write the APB peripherals.
"""
import os
from typing import Optional, TextIO

import numpy as np

from . import datamem, state, toplevel
from . import tornadocnn as tc
from . import unload
from .eprint import wprint

READ_TIME_NS = 230
WRITE_TIME_NS = 280


class APB():
    """
    APB read and write functionality.
    """

    def __init__(
            self,
            memfile: TextIO,
            verify_writes: bool = False,
            weight_header: Optional[TextIO] = None,
            sampledata_header: Optional[TextIO] = None,
            sampleoutput_header:  Optional[TextIO] = None,
            embedded_code: bool = False,
            write_zero_registers: bool = False,
            master=None,
            riscv=None,
            fast_fifo=False,
            input_chan=None,
            apifile=None,
            forever=False,
            fifo=False,
            embedded_arm=False,
            groups=None,
            oneshot=0,
            num_classes=None,
            output_width=8,
            bias=False,
            test_name=None,
    ):
        """
        Create an APB class object that writes to memfile.
        """
        assert tc.dev is not None

        self.memfile = memfile
        self.apifile = apifile
        self.verify_writes = verify_writes
        self.weight_header = weight_header
        self.sampledata_header = sampledata_header
        self.sampleoutput_header = sampleoutput_header
        self.embedded_code = embedded_code
        self.write_zero_regs = write_zero_registers
        self.master = master
        self.riscv = riscv
        self.fast_fifo = fast_fifo
        self.input_chan = input_chan
        self.forever = forever
        self.fifo = fifo
        self.embedded_arm = embedded_arm
        self.groups = groups
        self.oneshot = oneshot
        self.num_classes = num_classes
        self.output_width = output_width
        self.bias = bias
        self.test_name = test_name

        self.data = 0
        self.num = 0
        self.data_offs = 0
        self.mem = datamem.allocate()
        self.writes = 0
        self.reads = 0
        self.fifo_writes = 0
        self.fifo_reads = 0
        self.fastfifo_writes = 0
        self.fastfifo_reads = 0
        self.verify_listdata = []
        self.verify_text = []

        self.out_offset = 0
        self.layer = 0
        self.rollover = 0

        self.data_mem = self.kernel_mem = self.output_data_mem = None

        if state.rtl_preload_weights or state.new_kernel_loader:
            if not state.compact_weights:
                self.kernel_mem = np.empty(
                    (tc.dev.P_NUMGROUPS, tc.dev.P_NUMPRO, tc.dev.MASK_INSTANCES),
                    dtype=list,
                )
                for i in range(tc.dev.P_NUMGROUPS):
                    for j in range(tc.dev.P_NUMPRO):
                        for k in range(tc.dev.MASK_INSTANCES):
                            self.kernel_mem[i][j][k] = []

        if embedded_arm or embedded_code:
            return

        procs = (tc.dev.P_NUMPRO + tc.dev.P_SHARED - 1) // tc.dev.P_SHARED
        if state.rtl_preload:
            if not (state.compact_data or fifo or fast_fifo):
                self.data_mem = [[[[] for mem in range(tc.dev.INSTANCE_COUNT)]
                                  for proc in range(procs)]
                                 for group in range(tc.dev.P_NUMGROUPS)]
        if state.result_output:
            self.output_data_mem = [[[[] for mem in range(tc.dev.INSTANCE_COUNT)]
                                     for proc in range(procs)]
                                    for group in range(tc.dev.P_NUMGROUPS)]

    def write_mem(
            self,
            base_directory,
            test_name,
    ):
        """
        Write used kernel memories and data memories to disk
        """
        procs = (tc.dev.P_NUMPRO + tc.dev.P_SHARED - 1) // tc.dev.P_SHARED
        if self.data_mem is not None:
            target_dir = os.path.join(base_directory, test_name, 'data')
            os.makedirs(target_dir, exist_ok=True)
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(procs):
                    for mem in range(tc.dev.INSTANCE_COUNT):
                        if self.data_mem[group][proc][mem]:
                            self.data_mem[group][proc][mem].sort()
                            with open(
                                os.path.join(target_dir,
                                             f'DRAM_x16_{group}_proc_{proc*4}_ram_{mem}.dat'),
                                mode='w',
                                encoding='utf-8',
                            ) as f:
                                for (addr, val) in self.data_mem[group][proc][mem]:
                                    f.write(f'@{addr:04x} {val}\n')

        def sort_addr(val):
            return val[0]

        if self.kernel_mem is not None and not state.rtl_preload_weights:
            # Build a list of sequential kernel "chunks" so the loader code can use compact
            # memcpy instructions of streaming copy
            input_list = []
            val = []
            addr = -1
            offs = -1
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(tc.dev.P_NUMPRO):
                    for mem in range(tc.dev.mask_count(proc)):
                        if self.kernel_mem[group][proc][mem]:
                            self.kernel_mem[group][proc][mem].sort(key=sort_addr)
                            for (naddr, nval) in self.kernel_mem[group][proc][mem]:
                                if mem >= tc.dev.MASK_INSTANCES_EACH:
                                    phys_addr = state.apb_base + tc.dev.C_GROUP_OFFS * group \
                                        + tc.dev.C_MRAM_BASE + proc * tc.dev.MASK_OFFS * 16 \
                                        + tc.dev.MASK_WIDTH_SMALL * 16 \
                                        + (mem - tc.dev.MASK_INSTANCES_EACH) * 16 \
                                        * (tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL) \
                                        // tc.dev.MASK_INSTANCES_EACH \
                                        + naddr * 16
                                else:
                                    phys_addr = state.apb_base + tc.dev.C_GROUP_OFFS * group \
                                        + tc.dev.C_MRAM_BASE + proc * tc.dev.MASK_OFFS * 16 \
                                        + mem * 16 \
                                        * tc.dev.MASK_WIDTH_SMALL // tc.dev.MASK_INSTANCES_EACH \
                                        + naddr * 16
                                # Flush what we have, if anything
                                if offs > 0 and phys_addr != addr + offs * 16:
                                    input_list.append((addr, val))
                                    addr = -1
                                # Set new starting point
                                if addr == -1:
                                    addr = phys_addr
                                    offs = 0
                                    val = []
                                # Append current value
                                val.append(nval)
                                offs += 1
            # Flush remainder
            if addr != -1:
                assert len(val) == offs
                input_list.append((addr, val))

            # Create a header file of "chunks" (address, length, data)
            if input_list is not None:
                kl = []
                for (addr, val) in input_list:
                    assert len(val) > 0
                    # Address (u32), word length
                    if not state.mexpress:
                        kl.append(addr)
                    else:
                        kl.append(addr & ~(tc.dev.MASK_OFFS * 16 - 1) & 0xffffffff
                                  | ((addr & (tc.dev.MASK_OFFS * 16 - 1)) >> 2))
                    kl.append(len(val) * 4 if not state.mexpress else (len(val) * 9 + 3) // 4)
                    u = 0
                    count = 0
                    if not state.mexpress:
                        for k in val:
                            kl.append(k[0] & 0xff)
                            kl.append(((k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                                       (k[3] & 0xff) << 8 | k[4] & 0xff) & 0xffffffff)
                            kl.append(((k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                                       (k[7] & 0xff) << 8 | k[8] & 0xff) & 0xffffffff)
                            kl.append(0x00000000)
                    else:
                        for k in val:
                            for i in range(9):
                                u = (u << 8) & 0xffffffff
                                u |= k[i] & 0xff
                                count += 1
                                if count == 4:
                                    kl.append(u)
                                    u = 0
                                    count = 0
                        if count > 0:
                            kl.append(u << (4 - count) * 8)
                kl.append(0)  # EOF
                self.output_define(kl, 'KERNELS', '0x%08x', 8)

        if self.kernel_mem is not None and state.rtl_preload_weights:
            try:
                target_dir = os.path.join(base_directory, test_name, 'masks')
                os.makedirs(target_dir, exist_ok=False)
            except OSError:
                wprint(target_dir, 'exists')
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(tc.dev.P_NUMPRO):
                    for mem in range(tc.dev.MASK_INSTANCES):
                        if self.kernel_mem[group][proc][mem]:
                            self.kernel_mem[group][proc][mem].sort(key=sort_addr)
                            with open(
                                os.path.join(target_dir,
                                             f'MRAM_x16_{group}_proc_{proc}_ram_{mem}.dat'),
                                mode='w',
                                encoding='utf-8',
                            ) as f:
                                for (addr, val) in self.kernel_mem[group][proc][mem]:
                                    f.write(f'@{addr:04x} {val}\n')

        if self.output_data_mem is not None:
            target_dir = os.path.join(base_directory, test_name, 'data-output')
            os.makedirs(target_dir, exist_ok=True)
            try:
                target_dir = os.path.join(base_directory, test_name, 'data-expected')
                os.makedirs(target_dir, exist_ok=False)
            except OSError:
                wprint(target_dir, 'exists')
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(procs):
                    for mem in range(tc.dev.INSTANCE_COUNT):
                        if self.output_data_mem[group][proc][mem]:
                            self.output_data_mem[group][proc][mem].sort()
                            with open(
                                os.path.join(target_dir,
                                             f'DRAM_x16_{group}_proc_{proc*4}_ram_{mem}.dat'),
                                mode='w',
                                encoding='utf-8',
                            ) as f:
                                for (addr, val) in self.output_data_mem[group][proc][mem]:
                                    f.write(f'@{addr:04x} {val}\n')

    def get_time(
            self,
    ):
        """
        Return total bus access time in ms based on number of writes and reads
        """
        return (WRITE_TIME_NS * (self.writes + self.fifo_writes + self.fastfifo_writes) +
                READ_TIME_NS * (self.reads + self.fifo_reads + self.fastfifo_reads)) // 1000000

    def get_access_count(
            self,
    ):
        """
        Return total number of bus reads, writes, FIFO reads, writes, and Fast FIFO reads, writes
        """
        return self.reads, self.writes, self.fifo_reads, self.fifo_writes, self.fastfifo_reads, \
            self.fastfifo_writes

    def write(
            self,
            addr,
            val,
            comment='',
            indent='  ',
            no_verify=False,
            fifo=None,
            base=None,
            fifo_wait=True,
    ):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the output file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        raise NotImplementedError

    def inc_writes(
            self,
            count,
            fifo=None,
            fifo_wait=False,
    ):
        """
        Increase write count by `count`.
        """
        if fifo is None:
            self.writes += count
        elif not self.fast_fifo:
            if fifo_wait:
                self.fifo_reads += count
            self.fifo_writes += count
        else:
            if fifo_wait:
                self.fastfifo_reads += count
            self.fastfifo_writes += count

    def write_data(
            self,
            addr,
            val,
            comment='',
            indent='  ',
            no_verify=False,
            fifo=None,
            base=None,
    ):
        """
        Write address `addr` and data `val` to the output file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        return self.write(addr, val, comment, indent, no_verify, fifo, base)

    def verify(
            self,
            addr,
            val,
            mask=None,
            num_bytes=4,
            first_proc=0,
            comment='',
            rv=False,
            api=False,
            data=False,
            use_list=False,
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        """
        raise NotImplementedError

    def verify_list(
            self,
            addr,
            val,
            mask=None,
            num_bytes=4,
            first_proc=0,
            comment='',
            rv=False,
            api=False,
            data=False,
    ):
        """
        Verify that memory at address `addr` contains data `val`.
        """
        return self.verify(
            addr,
            val,
            mask=mask,
            num_bytes=num_bytes,
            first_proc=first_proc,
            comment=comment,
            rv=rv,
            api=api,
            data=data,
            use_list=self.embedded_code or state.result_filename is not None,
        )

    def wait(
            self,
            addr,
            mask,
            val,
            comment='',
    ):  # pylint: disable=unused-argument
        """
        Wait until memory at address `addr` masked with `mask` equals `val`.
        """
        raise NotImplementedError

    def set_memfile(
            self,
            memfile,
    ):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        self.memfile = memfile

    def write_fifo_ctl(
            self,
            reg,
            val,
            force_write=False,
            comment='',
    ):
        """
        Set FIFO control register `reg` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // fifo ctl {reg}'
        if val == 0 and not force_write:
            comment += ' *'
        addr = tc.dev.C_FIFO_BASE + reg*4
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, comment)
        if state.verbose:
            reg = f'{reg:02}'
            print(f'F{reg:<5}({addr:08x}): {val:08x}{comment}')

    def write_fast_fifo_ctl(
            self,
            reg,
            val,
            force_write=False,
            comment='',
    ):
        """
        Set fast FIFO control register `reg` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // fast fifo ctl {reg}'
        if val == 0 and not force_write:
            comment += ' *'
        addr = tc.dev.FAST_FIFO_BASE + reg*4
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, comment, base=0)
        if state.verbose:
            reg = f'{reg:02}'
            print(f'F{reg:<5}({addr:08x}): {val:08x}{comment}')

    def write_ctl(
            self,
            group,
            reg,
            val,
            force_write=False,
            comment='',
            no_verify=False,
    ):
        """
        Set global control register `reg` in group `group` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // global ctl {reg}'
        if val == 0 and not force_write:
            comment += ' *'
        addr = tc.ctl_addr(group, reg)
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, comment, no_verify=no_verify)
        if state.verbose:
            reg = f'{reg:02}'
            print(f'R{reg:<5}({addr:08x}): {val:08x}{comment}')

    def wait_ctl(
            self,
            group,
            reg,
            mask,
            val,
            comment='',
    ):
        """
        Reads from global control register `reg` in group `group` until `mask`ed value is `val`.
        An optional `comment` can be added to the output.
        """
        self.wait(tc.ctl_addr(group, reg), mask, val, comment)

    def verify_ctl(
            self,
            group,
            reg,
            mask,
            val,
            comment='',
    ):
        """
        Reads from global control register `reg` in group `group`, comparing to value `val`.
        An optional `comment` can be added to the output.
        """
        self.verify(tc.ctl_addr(group, reg), val, mask=mask, comment=comment,
                    api=self.embedded_code)

    def write_lreg(
            self,
            group,
            layer,
            reg,
            val,
            force_write=False,
            no_verify=False,
            comment='',
    ):
        """
        Set layer `layer` register `reg` in group `group` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // reg {reg}'
        if val == 0 and not force_write:
            comment += ' *'
        addr = tc.lreg_addr(group, reg, layer)
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, no_verify=no_verify, comment=comment)
        if state.verbose:
            print(f'L{layer} G{group} R{reg:02} ({addr:08x}): {val:08x}{comment}')

    def write_bias(
            self,
            group,
            offs,
            bias,
    ):
        """
        Write bias value `bias` to offset `offs` in bias memory #`group`.
        """
        addr = tc.dev.C_GROUP_OFFS*group + tc.dev.C_BRAM_BASE + offs * 4
        self.write(addr, bias & 0xff, ' // Bias')

    def write_tram(
            self,
            group,
            proc,
            offs,
            d,
            comment='',
    ):
        """
        Write value `d` to TRAM in group `group` and processor `proc` to offset `offs`.
        """
        addr = tc.dev.C_GROUP_OFFS*group + tc.dev.C_TRAM_BASE \
            + proc * tc.dev.TRAM_OFFS * 4 + offs * 4
        self.write(addr, d, f' // {comment}TRAM G{group} P{proc} #{offs}')

    def write_kern(
            self,
            ll,
            p,
            idx,
            k,
            size=9,
            verify_only=False,
            calc_x4=False,
            kern_offs=None,
            count=None,
    ):
        """
        Write single kernel `k` of length `size` for layer `ll`, processor `p` to index `idx` in
        weight memory.
        """
        assert p < tc.dev.MAX_PROC
        assert idx < tc.dev.mask_width(p)

        if calc_x4:
            start = kern_offs[ll]
            mem, rem = divmod((idx - start), (count + 3) // 4)
            start //= 4
            if idx < tc.dev.MASK_WIDTH_SMALL:
                assert 0 <= mem < 4
                idx_x4 = mem * (tc.dev.MASK_WIDTH_SMALL // 4) + rem + start
            else:
                idx_x4 = idx - tc.dev.MASK_WIDTH_SMALL
                idx_x4 = mem * ((tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL) // 4) + rem \
                    + tc.dev.MASK_WIDTH_SMALL + start
        else:
            idx_x4 = idx

        addr = tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
            + tc.dev.C_MRAM_BASE \
            + (p % tc.dev.P_NUMPRO) * tc.dev.MASK_OFFS * 16 + idx_x4 * 16

        if not verify_only:
            if self.kernel_mem is not None:
                if idx_x4 < tc.dev.MASK_WIDTH_SMALL:
                    mem, offs = divmod(idx_x4,
                                       tc.dev.MASK_WIDTH_SMALL // tc.dev.MASK_INSTANCES_EACH)
                else:
                    idx_x4 -= tc.dev.MASK_WIDTH_SMALL
                    mem, offs = divmod(idx_x4,
                                       (tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL)
                                       // tc.dev.MASK_INSTANCES_EACH)
                    mem += tc.dev.MASK_INSTANCES_EACH
                if state.rtl_preload_weights:
                    if size != 1:
                        val = f'{k[0] & 0xff:02x}_{k[1] & 0xff:02x}{k[2] & 0xff:02x}' \
                            f'{k[3] & 0xff:02x}{k[4] & 0xff:02x}_{k[5] & 0xff:02x}' \
                            f'{k[6] & 0xff:02x}{k[7] & 0xff:02x}{k[8] & 0xff:02x}'
                    else:
                        val = f'{k[0] & 0xff:02x}_00000000_00000000'
                else:
                    if size != 1:
                        val = np.empty(9, dtype=np.ubyte)
                        for i in range(9):
                            val[i] = k[i] & 0xff
                    else:
                        val = np.zeros(9, dtype=np.ubyte)
                        val[0] = k[0] & 0xff
                self.kernel_mem[p // tc.dev.P_NUMPRO][p % tc.dev.P_NUMPRO][mem]. \
                    append((offs, val))
            else:
                self.write(addr, k[0] & 0xff, no_verify=True,
                           comment=f' // Layer {ll}: processor {p} kernel #{idx}')
                if size != 1:
                    self.write(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                               (k[3] & 0xff) << 8 | k[4] & 0xff, no_verify=True)
                    self.write(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                               (k[7] & 0xff) << 8 | k[8] & 0xff, no_verify=True)
                else:
                    self.write(addr+4, 0, no_verify=True)
                    self.write(addr+8, 0, no_verify=True)
                self.write(addr+12, 0, no_verify=True)  # Execute write
        if not state.new_kernel_loader and (self.verify_writes or verify_only):
            self.verify(addr, k[0] & 0xff, api=True)
            if size != 1:
                self.verify(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                            (k[3] & 0xff) << 8 | k[4] & 0xff, api=True)
                self.verify(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                            (k[7] & 0xff) << 8 | k[8] & 0xff, api=True)
            else:
                self.verify(addr+4, 0, api=True)
                self.verify(addr+8, 0, api=True)
            self.verify(addr+12, 0, api=True)

    def write_byte_flush(
            self,
            offs,
            comment='',
            fifo=None,
    ):
        """
        Flush the contents of the internal buffer at offset `offs`, adding an optional
        `comment` to the output.
        This function also keeps track of all addresses that have been written before and
        can detect whether previous information is being overwritten.
        """
        if self.num > 0:
            woffs = self.data_offs - self.num
            datamem.store(self.mem, woffs, (-1, 0, 0, 0), check_overwrite=True)
            self.write_data(woffs, self.data, comment, fifo=fifo)
            self.num = 0
            self.data = 0
        self.data_offs = offs

    def write_byte(
            self,
            offs,
            val,
            comment='',
            fifo=None,
    ):
        """
        Add byte `val` that should be written at offset `offs` to the internal buffer.
        When reaching 4 bytes, or when the offset is not contiguous, pad with zeros and
        flush the before adding the new value to the buffer.
        An optional `comment` can be added to the output.
        """
        if offs != self.data_offs:
            self.write_byte_flush(offs)

        # Collect and write if multiple of 4 (little endian byte order)
        self.data |= (val & 0xff) << (8*self.num)
        self.num += 1
        self.data_offs += 1
        if self.num == 4:
            self.write_byte_flush(offs+1, comment, fifo=fifo)

    def get_mem(
            self,
    ):
        """
        Return reference to the memory array.
        """
        return self.mem

    def output(
            self,
            comment,
            api=False,
    ):
        """
        Write the string `comment` to the output file without further interpretation.
        """
        if self.memfile is None:
            return

        if api and self.apifile is not None:
            self.apifile.write(comment)
        else:
            self.memfile.write(comment)

    def copyright_header(
            self,
    ):
        """
        Write copyright headers.
        The base class does nothing.
        """
        return

    def header(
            self,
    ):
        """
        Write file headers.
        The base class does nothing.
        """
        return

    def function_header(
            self,
            dest='api',  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Write the header for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def function_footer(
            self,
            dest='api',  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Write the footer for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def main(
            self,
    ):
        """
        Write the main function.
        The base class does nothing.
        """
        return

    def softmax_layer(
            self,
            *args,  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Write the call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        The base class does nothing.
        """
        return

    def unload(
            self,
            *,
            output_layer,
            processor_map,
            input_shape,
            output_offset,
            out_expand,
            out_expand_thresh,
            output_width,
            write_gap,
    ):  # pylint: disable=unused-argument
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        The base class does nothing.
        """
        return

    def verify_unload(
            self,
            ll,
            in_map,
            out_map,
            out_buf,
            processor_map,
            input_shape,
            out_offset=0,
            out_expand=1,
            out_expand_thresh=64,
            output_width=8,
            overwrite_ok=False,
            mlator=False,
            write_gap=0,
            unload_layer=False,
            streaming=False,
            rollover=None,
    ):
        """
        Write a verification function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        self.layer = ll
        self.out_offset = out_offset
        self.rollover = rollover

        unload.verify(
            self.verify_list,
            ll,
            in_map,
            out_map,
            out_buf,
            processor_map,
            input_shape,
            out_offset,
            out_expand,
            out_expand_thresh,
            output_width,
            overwrite_ok=overwrite_ok,
            mlator=mlator,
            body=self.verify_text,
            write_gap=write_gap,
            unload_layer=unload_layer,
            embedded=self.embedded_code,
            test_name=self.test_name,
            streaming=streaming,
        )

    def verify_unload_finalize(self):
        """
        Finalize the verification function.
        """
        if len(self.verify_listdata) > 0:
            # Sort by mask, then address
            data = sorted(self.verify_listdata)

            rv = data[0][5]
            action = 'rv = CNN_FAIL;' if rv else 'return CNN_FAIL;'

            if self.sampleoutput_header is None:
                for (_, addr, mask_str, val, val_bytes, rv_item, comment) in data:
                    assert rv == rv_item
                    self.memfile.write(f'  if ((*((volatile uint32_t *) 0x{addr:08x}){mask_str})'
                                       f' != 0x{val:0{2*val_bytes}x}) {action}{comment}\n')
                    self.reads += 1
            else:
                # Output is sorted by mask. Group like masks together.
                max_count = state.max_count

                output_array = []
                val_array = []
                output_mask = 0
                output_addr = 0
                next_addr = 0
                cumulative_length = 0

                for (mask_item, addr, _, val, _, rv_item, _) in data:
                    assert rv == rv_item

                    # New mask? Output collected data and start over
                    if mask_item != output_mask or addr != next_addr:
                        if len(val_array) > 0:
                            output_array += [
                                output_addr,
                                output_mask,
                                len(val_array),
                            ]
                            output_array += val_array

                        # Start new block
                        val_array = []
                        output_mask = mask_item
                        output_addr = next_addr = addr

                    # Collect the next value
                    val_array.append(val)
                    next_addr += 4
                    cumulative_length += 1
                    if max_count is not None and cumulative_length > max_count:
                        break

                if len(val_array) > 0:
                    output_array += [
                        output_addr,
                        output_mask if output_mask is not None else 0xffffffff,
                        len(val_array),
                    ]
                    output_array += val_array
                    output_array.append(0)  # Terminator

                # Write to the header file
                toplevel.c_define(self.sampleoutput_header, output_array, 'SAMPLE_OUTPUT',
                                  '0x%08x', 8)

                # Write to the function
                self.memfile.write('  int i;\n'
                                   '  uint32_t mask, len;\n'
                                   '  volatile uint32_t *addr;\n'
                                   '  const uint32_t *ptr = sample_output;\n\n'
                                   '  while ((addr = (volatile uint32_t *) *ptr++) != 0) {\n'
                                   '    mask = *ptr++;\n'
                                   '    len = *ptr++;\n'
                                   '    for (i = 0; i < len; i++)\n'
                                   '      if ((*addr++ & mask) != *ptr++) {\n'
                                   '        printf("Data mismatch (%d/%d) at address 0x%08x: '
                                   'Expected 0x%08x, read 0x%08x.\\n",\n'
                                   '               i + 1, len, addr - 1, *(ptr - 1), '
                                   '*(addr - 1) & mask);\n'
                                   f'        {action}\n'
                                   '      }\n'
                                   '  }\n')

            self.verify_listdata = []  # Consume

        if len(self.verify_text) > 0:
            for e in self.verify_text:
                self.memfile.write(e)
            self.verify_text = []  # Consume

    def output_define(
            self,
            array,
            define_name,
            fmt,
            columns,
            weights=True,
    ):  # pylint: disable=unused-argument
        """
        Write a #define for array `array` to `define_name`, using format `fmt` and creating
        a line break after `columns` items each.
        The base class does nothing.
        """
        return


class APBBlockLevel(APB):
    """
    APB read and write functionality for block level tests.
    """
    def __init__(
            self,
            memfile,
            verify_writes=False,
            weight_header=None,
            sampledata_header=None,
            embedded_code=False,
            write_zero_registers=False,
            master=None,
            riscv=None,
            fast_fifo=False,
            input_chan=None,
            apifile=None,
            test_name=None,
            passfile=None,
    ):
        super().__init__(
            memfile,
            verify_writes=verify_writes,
            weight_header=weight_header,
            sampledata_header=sampledata_header,
            embedded_code=embedded_code,
        )
        self.foffs = 0
        self.passfile = passfile

    def write(
            self,
            addr,
            val,
            comment='',
            indent='  ',
            no_verify=False,
            fifo=None,
            base=None,
            fifo_wait=True,
    ):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the .mem file.
        """
        assert val >= 0
        assert addr >= 0
        if base is None:
            addr += state.apb_base

        self.memfile.write(f'@{self.foffs:04x} {addr:08x}\n')
        self.memfile.write(f'@{self.foffs+1:04x} {val:08x}\n')
        self.foffs += 2
        if fifo is None:
            self.writes += 1
        elif not self.fast_fifo:
            if fifo_wait:
                self.fifo_reads += 1
            self.fifo_writes += 1
        else:
            if fifo_wait:
                self.fastfifo_reads += 1
            self.fastfifo_writes += 1

    def verify(
            self,
            addr,
            val,
            mask=None,
            num_bytes=4,
            first_proc=0,
            comment='',
            rv=False,
            api=False,
            data=False,
            use_list=False,
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        For block level tests, this function ensuring the input address and data are not negative
        and then writes address and expected data in .mem file format.
        """
        assert val >= 0
        assert addr >= 0
        addr += state.apb_base

        self.memfile.write(f'@{self.foffs:04x} {addr:08x}\n')
        self.memfile.write(f'@{self.foffs+1:04x} {val:08x}\n')
        self.foffs += 2

    def wait(
            self,
            addr,
            mask,
            val,
            comment='',
    ):  # pylint: disable=unused-argument
        """
        Wait until memory at address `addr` masked with `mask` equals `val`.
        For block level tests, this function does nothing useful other than ensuring the inputs
        address and data are not negative.
        """
        assert val >= 0
        assert addr >= 0

    def set_memfile(
            self,
            memfile,
    ):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        super().set_memfile(memfile)
        self.foffs = 0

    def output(
            self,
            comment,
            api=False,
    ):  # pylint: disable=unused-argument
        """
        Do nothing.
        """


class APBDebug(APBBlockLevel):
    """
    Intended for debugging memory contents.
    """
    def verify(
            self,
            addr,
            val,
            mask=None,
            num_bytes=4,
            first_proc=0,
            comment='',
            rv=False,
            api=False,
            data=False,
            use_list=False,
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        This function ensures the input address and data are not negative
        and then writes the expected data to a file.
        """
        assert val >= 0
        assert addr >= 0
        addr -= tc.dev.C_SRAM_BASE
        assert addr >= 0, comment
        group = addr // tc.dev.C_GROUP_OFFS
        addr %= tc.dev.C_GROUP_OFFS
        proc = addr // (tc.dev.INSTANCE_SIZE*16)
        offs = addr % (tc.dev.INSTANCE_SIZE*16)

        if self.rollover is not None:
            offs = (offs - self.out_offset) % (self.rollover * 4) + self.out_offset
        else:
            offs = addr
        offs |= (proc | group * 4) * (tc.dev.INSTANCE_SIZE*16)  # proc is 0, 4, 8 or 12
        offs //= 4  # Switch to 32-bit word address
        assert offs >= 0, comment

        # Comment is " // channel,row,col" where each of the values can be an int or a range
        if comment.startswith(' // '):
            comment = comment[4:]

        self.memfile.write(f'w,{offs:x},{val:x},{self.layer},{comment}\n')
        if self.passfile is not None:
            state.write_count += 1
            self.passfile.write(f'w,{offs | 0x200000:x},{val:x},{self.layer},'
                                f'{state.write_count}\n')
        self.foffs += 1


class APBTopLevel(APB):
    """
    APB read and write functionality for top level tests.
    """
    def write(
            self,
            addr,
            val,
            comment='',
            indent='  ',
            no_verify=False,
            fifo=None,
            base=None,
            fifo_wait=True,
    ):
        """
        Write address `addr` and data `val` to the .c file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        if not isinstance(val, str):
            assert val >= 0
            val = f'0x{val:08x}'
        assert addr >= 0
        if base is None:
            addr += state.apb_base

        mfile = self.apifile or self.memfile
        if mfile is None:
            return

        if fifo is None:
            mfile.write(f'{indent}*((volatile uint32_t *) 0x{addr:08x}) = '
                        f'{val};{comment}\n')
            self.writes += 1
            if self.verify_writes and not no_verify:
                mfile.write(f'{indent}if (*((volatile uint32_t *) 0x{addr:08x}) != {val}) '
                            'return CNN_FAIL;\n')
                self.reads += 1
        else:
            if not self.fast_fifo:
                addr = state.apb_base + tc.dev.C_FIFO_BASE
                if fifo_wait:
                    self.memfile.write(f'{indent}// Remove the following line if there is no risk '
                                       'that the source would overrun the FIFO:\n'
                                       f'{indent}while (((*((volatile uint32_t *) '
                                       f'0x{addr + tc.dev.FIFO_STAT*4:08x})'
                                       f' & {1 << fifo})) != 0); // Wait for FIFO {fifo}\n')
                    if not state.compact_data:
                        self.fifo_reads += 1  # Otherwise handled by 'inc_writes()' via load.py
                self.memfile.write(f'{indent}*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FIFO_REG*4 + fifo*4:08x}) = '
                                   f'{val};{comment}\n')
                if not state.compact_data:
                    self.fifo_writes += 1  # Otherwise handled by 'inc_writes()' via load.py
            else:
                addr = tc.dev.FAST_FIFO_BASE
                if fifo_wait:
                    self.memfile.write(f'{indent}// Remove the following line if there is no risk '
                                       'that the source would overrun the FIFO:\n'
                                       f'{indent}while (((*((volatile uint32_t *) '
                                       f'0x{addr + tc.dev.FAST_FIFO_SR*4:08x})'
                                       f' & 2)) != 0); // Wait for FIFO\n')
                    if not state.compact_data:
                        self.fastfifo_reads += 1  # Otherwise handled by 'inc_writes()'
                self.memfile.write(f'{indent}*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FAST_FIFO_DR*4:08x}) = '
                                   f'{val};{comment}\n')
                if not state.compact_data:
                    self.fastfifo_writes += 1  # Otherwise handled by inc_writes() via load.py

    def write_data(
            self,
            addr,
            val,
            comment='',
            indent='  ',
            no_verify=False,
            fifo=None,
            base=None,
    ):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the output file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        The `write_data()` function is called for data memory only to allow memory-preloading
        in RTL simulation. For normal cases, it is equivalent to `write()`.
        """
        if self.data_mem is not None and fifo is None:
            group, proc, mem, offs = tc.dev.datainstance_from_addr(addr)
            self.data_mem[group][proc][mem].append((offs, f'{val:08x}'))
            return

        self.write(addr, val, comment, indent, no_verify, fifo, base)

    def verify(
            self,
            addr,
            val,
            mask=None,
            num_bytes=4,
            first_proc=0,
            comment='',
            rv=False,
            api=False,
            data=False,
            use_list=False,
    ):
        """
        Verify that memory at address `addr` contains data `val`.
        If `rv` is `True`, do not immediately return CNN_FAIL, but just set the status word.
        An optional `comment` can be added to the output.
        """
        assert val >= 0
        assert addr >= 0

        if self.output_data_mem is not None and data:
            if mask is None:
                if num_bytes == 4:
                    mask = ''
                elif num_bytes == 3:
                    mask = 0xffffff
                elif num_bytes == 2:
                    mask = 0xffff
                elif num_bytes == 1:
                    mask = 0xff
                else:
                    raise NotImplementedError
                assert first_proc + num_bytes <= 4

            if mask != '':
                mask <<= first_proc * 8

            val = f'{val:08x}'
            if mask != '':
                w = ''
                for i, e in enumerate(f'{mask:08x}'):
                    w += 'X' if e != 'f' else val[i]
                val = w
            group, proc, mem, offs = tc.dev.datainstance_from_addr(addr)
            self.output_data_mem[group][proc][mem].append((offs, val))
            return

        addr += state.apb_base

        if self.memfile is None:
            return

        if mask is None:
            if num_bytes == 3:
                mask = 0xffffff
            elif num_bytes == 2:
                mask = 0xffff
            elif num_bytes == 1:
                mask = 0xff
            elif num_bytes != 4:
                raise NotImplementedError
            assert first_proc + num_bytes <= 4

        val_bytes = num_bytes
        if mask is not None:
            mask <<= first_proc * 8
            val_bytes += first_proc
            val &= mask
            mask_str = f' & 0x{mask:0{2*val_bytes}x}'
        else:
            mask_str = ''
            mask = 0xffffffff

        action = 'rv = CNN_FAIL;' if rv else 'return CNN_FAIL;'

        if not use_list:
            s = f'  if ((*((volatile uint32_t *) 0x{addr:08x}){mask_str})' \
                f' != 0x{val:0{2*val_bytes}x}) {action}{comment}\n'
            if api:
                mfile = self.apifile or self.memfile
                mfile.write(s)
            else:
                self.verify_text.append(s)
        else:
            self.verify_listdata.append((mask, addr, mask_str, val, val_bytes, rv, comment))
        self.reads += 1

    def wait(
            self,
            addr,
            mask,
            val,
            comment='',
    ):
        """
        Waits until memory at address `addr` masked with `mask` contains data `val`.
        """
        assert val >= 0
        assert addr >= 0
        addr += state.apb_base

        mfile = self.apifile or self.memfile
        if mfile is None:
            return

        mfile.write(f'  while ((*((volatile uint32_t *) 0x{addr:08x}) & 0x{mask:0x})'
                    f' != 0x{val:0x});'
                    f'{comment}\n')

    def copyright_header(
            self,
    ):
        """
        Write copyright headers.
        """
        if self.apifile is not None:
            toplevel.copyright_header(self.apifile)
        toplevel.copyright_header(self.memfile)

    def header(
            self,
    ):
        """
        Write include files and forward definitions to .c file.
        """
        if self.apifile is not None:
            toplevel.header(
                self.apifile,
                embedded_code=self.embedded_code,
                master=self.master,
                riscv=self.riscv,
                embedded_arm=self.embedded_arm,
                groups=self.groups,
                lib=True,
                oneshot=self.oneshot,
            )

        toplevel.header(
            self.memfile,
            embedded_code=self.embedded_code,
            master=self.master,
            riscv=self.riscv,
            embedded_arm=self.embedded_arm,
            groups=self.groups,
            lib=False if self.apifile is not None else None,
            oneshot=self.oneshot,
        )

    def function_header(
            self,
            dest='api',
            **kwargs,
    ):
        """
        Write the header for a function.
        """
        toplevel.function_header(
            self.apifile or self.memfile if dest == 'api' else self.memfile,
            **kwargs,
        )

    def function_footer(
            self,
            dest='api',
            **kwargs,
    ):
        """
        Write the footer for a function.
        """
        toplevel.function_footer(
            self.apifile or self.memfile if dest == 'api' else self.memfile,
            **kwargs,
        )

    def main(
            self,
    ):
        """
        Write the main function.
        """
        toplevel.main(
            self.memfile,
            self.apifile,
            embedded_code=self.embedded_code,
            riscv=self.riscv,
            channels=self.input_chan,
            unload=self.embedded_code,
            load_kernels=not state.rtl_preload_weights,
            forever=self.forever,
            fifo=self.fifo,
            groups=self.groups,
            embedded_arm=self.embedded_arm,
            output_width=self.output_width,
            bias=self.bias,
            oneshot=self.oneshot,
            name=self.test_name,
        )

    def softmax_layer(
            self,
            *args,
            **kwargs,
    ):
        """
        Write call to the softmax layer.
        """
        toplevel.softmax_layer(self.memfile, *args, **kwargs)

    def unload(
            self,
            *,
            output_layer,
            processor_map,
            input_shape,
            output_offset,
            out_expand,
            out_expand_thresh,
            output_width,
            write_gap,
    ):
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        unload.unload(
            memfile=self.apifile or self.memfile,
            output_layer=output_layer,
            processor_map=processor_map,
            input_shape=input_shape,
            out_offset=output_offset,
            out_expand=out_expand,
            out_expand_thresh=out_expand_thresh,
            output_width=output_width,
            write_gap=write_gap,
        )

    def output_define(
            self,
            array,
            define_name,
            fmt,
            columns,
            weights=True,
    ):
        """
        Write a #define for array `array` to `define_name`, using format `fmt` and creating
        a line break after `columns` items each.
        If `weight`, write to the `weights.h` file, else to `sampledata.h`.
        """
        if weights:
            toplevel.c_define(self.weight_header, array, define_name, fmt, columns)
        else:
            toplevel.c_define(self.sampledata_header, array, define_name, fmt, columns)

    def select_clock(
            self,
            source,
            divider,
            comment='',
    ):
        """
        Switch clock source and divider.
        """
        toplevel.select_clock(self.apifile or self.memfile, source, divider, comment)


def apbwriter(
        *args,
        debug_mem=False,
        **kwargs,
):
    """
    Depending on `block_level` and `debug_mem`, return a block level .mem file writer,
    a top level .c file writer or a debug writer.
    """
    if not debug_mem:
        APBClass = APBBlockLevel if state.block_mode or debug_mem else APBTopLevel
    else:
        APBClass = APBDebug
    return APBClass(
        *args,
        **kwargs,
    )
