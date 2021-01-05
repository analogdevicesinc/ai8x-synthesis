###################################################################################################
# Copyright (C) Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
###################################################################################################
"""
Routines to read and write the APB peripherals.
"""
import os

import toplevel
import tornadocnn as tc
import unload
from eprint import eprint, wprint

READ_TIME_NS = 230
WRITE_TIME_NS = 280


class APB():
    """
    APB read and write functionality.
    """

    def __init__(
            self,
            memfile,
            apb_base,
            verify_writes=False,
            no_error_stop=False,
            weight_header=None,
            sampledata_header=None,
            embedded_code=False,
            compact_weights=False,
            compact_data=False,
            write_zero_registers=False,
            weight_filename=None,
            sample_filename=None,
            device=84,
            verify_kernels=False,
            master=None,
            riscv=None,
            riscv_exclusive=False,
            riscv_flash=False,
            riscv_cache=False,
            riscv_debug=False,
            debugwait=1,
            fast_fifo=False,
            input_csv=None,
            input_csv_format=888,
            input_chan=None,
            sleep=False,
            blocklevel=False,
            mexpress=False,
            mem_output=False,
            mem_output_final=False,
            apifile=None,
            measure_energy=False,
            timer=None,
            pll=False,
            boost=None,
            forever=False,
            fifo=False,
            fail_indicator=False,
            embedded_arm=False,
            groups=None,
            clock_trim=None,
            oneshot=0,
            softmax=False,
            stopstart=False,
            num_classes=None,
            output_width=8,
            bias=False,
            wfi=True,
    ):
        """
        Create an APB class object that writes to memfile.
        """
        self.memfile = memfile
        self.apifile = apifile
        self.apb_base = apb_base
        self.verify_writes = verify_writes
        self.no_error_stop = no_error_stop
        self.weight_header = weight_header
        self.sampledata_header = sampledata_header
        self.embedded_code = embedded_code
        self.compact_weights = compact_weights
        self.compact_data = compact_data
        self.write_zero_regs = write_zero_registers
        self.weight_filename = weight_filename
        self.sample_filename = sample_filename
        self.device = device
        self.verify_kernels = verify_kernels
        self.master = master
        self.riscv = riscv
        self.riscv_exclusive = riscv_exclusive
        self.riscv_flash = riscv_flash
        self.riscv_cache = riscv_cache
        self.riscv_debug = riscv_debug
        self.debugwait = debugwait
        self.fast_fifo = fast_fifo
        self.input_csv = input_csv
        self.input_csv_format = input_csv_format
        self.input_chan = input_chan
        self.sleep = sleep
        self.blocklevel = blocklevel
        self.measure_energy = measure_energy
        self.timer = timer
        self.mexpress = mexpress
        self.pll = pll
        self.boost = boost
        self.forever = forever
        self.fifo = fifo
        self.fail_indicator = fail_indicator
        self.embedded_arm = embedded_arm
        self.groups = groups
        self.clock_trim = clock_trim
        self.oneshot = oneshot
        self.softmax = softmax
        self.stopstart = stopstart
        self.num_classes = num_classes
        self.output_width = output_width
        self.bias = bias
        self.wfi = wfi

        self.data = 0
        self.num = 0
        self.data_offs = 0
        self.mem = [None] * tc.dev.C_GROUP_OFFS * tc.dev.P_NUMGROUPS
        self.writes = 0
        self.reads = 0

        self.data_mem = self.kernel_mem = self.output_data_mem = None

        if embedded_arm or embedded_code:
            return

        procs = (tc.dev.P_NUMPRO + tc.dev.P_SHARED - 1) // tc.dev.P_SHARED
        if mem_output:
            if not (compact_data or fifo or fast_fifo):
                self.data_mem = [[[[] for mem in range(tc.dev.INSTANCE_COUNT)]
                                  for proc in range(procs)]
                                 for group in range(tc.dev.P_NUMGROUPS)]
            if not (compact_weights or mexpress or verify_kernels):
                self.kernel_mem = [[[[] for mem in range(tc.dev.MASK_INSTANCES)]
                                    for proc in range(tc.dev.P_NUMPRO)]
                                   for group in range(tc.dev.P_NUMGROUPS)]
        if mem_output_final:
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
                                mode='w'
                            ) as f:
                                for (addr, val) in self.data_mem[group][proc][mem]:
                                    f.write(f'@{addr:04x} {val}\n')

        if self.kernel_mem is not None:
            try:
                target_dir = target_dir = os.path.join(base_directory, test_name, 'masks')
                os.makedirs(target_dir, exist_ok=False)
            except OSError:
                wprint(target_dir, 'already exists')
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(tc.dev.P_NUMPRO):
                    for mem in range(tc.dev.MASK_INSTANCES):
                        if self.kernel_mem[group][proc][mem]:
                            self.kernel_mem[group][proc][mem].sort()
                            with open(
                                os.path.join(target_dir,
                                             f'MRAM_x16_{group}_proc_{proc}_ram_{mem}.dat'),
                                mode='w'
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
                wprint(target_dir, 'already exists')
            for group in range(tc.dev.P_NUMGROUPS):
                for proc in range(procs):
                    for mem in range(tc.dev.INSTANCE_COUNT):
                        if self.output_data_mem[group][proc][mem]:
                            self.output_data_mem[group][proc][mem].sort()
                            with open(
                                os.path.join(target_dir,
                                             f'DRAM_x16_{group}_proc_{proc*4}_ram_{mem}.dat'),
                                mode='w'
                            ) as f:
                                for (addr, val) in self.output_data_mem[group][proc][mem]:
                                    f.write(f'@{addr:04x} {val}\n')

    def get_time(
            self,
    ):
        """
        Return total bus access time in ms based on number of writes and reads
        """
        return (WRITE_TIME_NS * self.writes + READ_TIME_NS * self.reads) // 1000000

    def write(
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
        """
        raise NotImplementedError

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
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        """
        raise NotImplementedError

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
            debug=False,
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
        if debug:
            reg = f'{reg:02}'
            print(f'F{reg:<5}({addr:08x}): {val:08x}{comment}')

    def write_fast_fifo_ctl(
            self,
            reg,
            val,
            debug=False,
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
        if debug:
            reg = f'{reg:02}'
            print(f'F{reg:<5}({addr:08x}): {val:08x}{comment}')

    def write_ctl(
            self,
            group,
            reg,
            val,
            debug=False,
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
        if debug:
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
        self.verify(tc.ctl_addr(group, reg), val, mask=mask, comment=comment)

    def write_lreg(
            self,
            group,
            layer,
            reg,
            val,
            debug=False,
            force_write=False,
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
            self.write(addr, val, comment)
        if debug:
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
            calcx4=False,
    ):
        """
        Write single kernel `k` of length `size` for layer `ll`, processor `p` to index `idx` in
        weight memory.
        """
        assert p < tc.dev.MAX_PROC
        assert idx < tc.dev.mask_width(p)
        if not calcx4:
            addr = tc.dev.C_GROUP_OFFS * (p // tc.dev.P_NUMPRO) \
                + tc.dev.C_MRAM_BASE \
                + (p % tc.dev.P_NUMPRO) * tc.dev.MASK_OFFS * 16 + idx * 16
            idx_x4 = idx
        else:
            if idx < tc.dev.MASK_WIDTH_SMALL:
                idx_x4 = (idx % 4) * (tc.dev.MASK_WIDTH_SMALL // 4) + idx // 4
            else:
                idx -= tc.dev.MASK_WIDTH_SMALL
                idx_x4 = (idx % 4) * ((tc.dev.MASK_WIDTH_LARGE - tc.dev.MASK_WIDTH_SMALL) // 4) \
                    + idx // 4
                idx += tc.dev.MASK_WIDTH_SMALL
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
                if size != 1:
                    val = f'{k[0] & 0xff:02x}_{k[1] & 0xff:02x}{k[2] & 0xff:02x}' \
                          f'{k[3] & 0xff:02x}{k[4] & 0xff:02x}_{k[5] & 0xff:02x}' \
                          f'{k[6] & 0xff:02x}{k[7] & 0xff:02x}{k[8] & 0xff:02x}'
                else:
                    val = f'{k[0] & 0xff:02x}_00000000_00000000'
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
        if self.verify_writes or verify_only:
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

    def check_overwrite(
            self,
            offs,
    ):
        """
        Check whether we're overwriting location `offs`.
        """
        if self.mem[offs >> 2]:
            eprint(f'Overwriting location {offs:08x}', error=not self.no_error_stop)

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
            self.check_overwrite(woffs)
            self.write_data(woffs, self.data, comment, fifo=fifo)
            self.mem[woffs >> 2] = True
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

    def copyright_header(  # pylint: disable=no-self-use
            self,
    ):
        """
        Write copyright headers.
        The base class does nothing.
        """
        return

    def header(  # pylint: disable=no-self-use
            self,
    ):
        """
        Write file headers.
        The base class does nothing.
        """
        return

    def function_header(  # pylint: disable=no-self-use
            self,
            dest='api',  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Write the header for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def function_footer(  # pylint: disable=no-self-use
            self,
            dest='api',  # pylint: disable=unused-argument
            **kwargs,  # pylint: disable=unused-argument
    ):
        """
        Write the footer for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def main(  # pylint: disable=no-self-use
            self,
    ):
        """
        Write the main function.
        The base class does nothing.
        """
        return

    def fc_layer(  # pylint: disable=no-self-use
            self,
            *args,
            **kwargs,
    ):  # pylint: disable=unused-argument
        """
        Write the call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        The base class does nothing.
        """
        return

    def fc_verify(  # pylint: disable=no-self-use
            self,
            data,
    ):  # pylint: disable=unused-argument
        """
        Write the code to verify the fully connected layer against `data`.
        The base class does nothing.
        """
        return

    def unload(  # pylint: disable=no-self-use
            self,
            processor_map,
            input_shape,
            output_offset=0,
            out_expand=1,
            out_expand_thresh=64,
            output_width=8,
            pool=None,
            pool_stride=1,
            mlator=False,
            write_gap=0,
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
            pool=None,
            pool_stride=1,
            overwrite_ok=False,
            no_error_stop=False,
            mlator=False,
            max_count=None,
            write_gap=0,
            final_layer=0,
    ):
        """
        Write a verification function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        unload.verify(
            self.verify,
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
            no_error_stop=no_error_stop,
            mlator=mlator,
            apb_base=self.apb_base,
            stream=self.memfile,
            max_count=max_count,
            write_gap=write_gap,
            final_layer=final_layer,
        )

    def output_define(  # pylint: disable=no-self-use
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
            apb_base,
            verify_writes=False,
            no_error_stop=False,
            weight_header=None,
            sampledata_header=None,
            compact_weights=False,
            compact_data=False,
            embedded_code=False,
            write_zero_registers=False,
            weight_filename=None,
            sample_filename=None,
            device=84,
            verify_kernels=False,
            master=None,
            riscv=None,
            riscv_exclusive=False,
            riscv_flash=False,
            riscv_cache=False,
            fast_fifo=False,
            input_csv=None,
            input_csv_format=None,
            input_chan=None,
            sleep=False,
            apifile=None,
    ):
        super().__init__(
            memfile,
            apb_base,
            verify_writes=verify_writes,
            no_error_stop=no_error_stop,
            weight_header=weight_header,
            sampledata_header=sampledata_header,
            embedded_code=embedded_code,
            compact_weights=False,
            compact_data=False,
            weight_filename=None,
            sample_filename=None,
            blocklevel=True,
        )
        self.foffs = 0

    def write(
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
        Write address `addr` and data `val` to the .mem file.
        """
        assert val >= 0
        assert addr >= 0
        if base is None:
            addr += self.apb_base

        self.memfile.write(f'@{self.foffs:04x} {addr:08x}\n')
        self.memfile.write(f'@{self.foffs+1:04x} {val:08x}\n')
        self.foffs += 2

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
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        For block level tests, this function ensuring the input address and data are not negative
        and then writes address and expected data in .mem file format.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

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
    ):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        This function ensuring the input address and data are not negative
        and then writes the expected data to a file.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        self.memfile.write(f'{val:08x}\n')
        self.foffs += 2


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
            addr += self.apb_base

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
                addr = self.apb_base + tc.dev.C_FIFO_BASE
                self.memfile.write(f'{indent}while (((*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FIFO_STAT*4:08x})'
                                   f' & {1 << fifo})) != 0); // Wait for FIFO {fifo}\n')
                self.memfile.write(f'{indent}*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FIFO_REG*4 + fifo*4:08x}) = '
                                   f'{val};{comment}\n')
            else:
                addr = tc.dev.FAST_FIFO_BASE
                self.memfile.write(f'{indent}while (((*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FAST_FIFO_SR*4:08x})'
                                   f' & 2)) != 0); // Wait for FIFO\n')
                self.memfile.write(f'{indent}*((volatile uint32_t *) '
                                   f'0x{addr + tc.dev.FAST_FIFO_DR*4:08x}) = '
                                   f'{val};{comment}\n')
            self.writes += 1

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

        addr += self.apb_base

        if self.memfile is None:
            return

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

        val_bytes = num_bytes
        if mask != '':
            mask <<= first_proc * 8
            val_bytes += first_proc
            val &= mask
            mask = f' & 0x{mask:0{2*val_bytes}x}'

        if rv:
            action = 'rv = CNN_FAIL;'
        else:
            action = 'return CNN_FAIL;'

        mfile = self.apifile or self.memfile if api else self.memfile
        mfile.write(f'  if ((*((volatile uint32_t *) 0x{addr:08x}){mask})'
                    f' != 0x{val:0{2*val_bytes}x}) '
                    f'{action}{comment}\n')
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
        addr += self.apb_base

        if self.memfile is None:
            return

        self.memfile.write(f'  while ((*((volatile uint32_t *) 0x{addr:08x}) & 0x{mask:0x})'
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
                self.apb_base,
                embedded_code=self.embedded_code,
                compact_weights=self.compact_weights,
                compact_data=self.compact_data,
                weight_filename=self.weight_filename,
                sample_filename=self.sample_filename,
                master=self.master,
                verify_kernels=self.verify_kernels,
                riscv=self.riscv,
                camera=self.input_csv is not None,
                embedded_arm=self.embedded_arm,
                fail_indicator=self.fail_indicator,
                measure_energy=self.measure_energy,
                timer=self.timer,
                groups=self.groups,
                lib=True,
                oneshot=self.oneshot,
            )

        toplevel.header(
            self.memfile,
            self.apb_base,
            embedded_code=self.embedded_code,
            compact_weights=self.compact_weights,
            compact_data=self.compact_data,
            weight_filename=self.weight_filename,
            sample_filename=self.sample_filename,
            master=self.master,
            verify_kernels=self.verify_kernels,
            riscv=self.riscv,
            camera=self.input_csv is not None,
            embedded_arm=self.embedded_arm,
            fail_indicator=self.fail_indicator,
            measure_energy=self.measure_energy,
            timer=self.timer,
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
            riscv_flash=self.riscv_flash and not self.riscv_cache,
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
            riscv_exclusive=self.riscv_exclusive,
            riscv_flash=self.riscv_flash,
            riscv_cache=self.riscv_cache,
            riscv_debug=self.riscv_debug,
            debugwait=self.debugwait,
            device=self.device,
            camera=self.input_csv is not None,
            camera_format=self.input_csv_format,
            channels=self.input_chan,
            sleep=self.sleep,
            unload=self.embedded_code,
            load_kernels=self.kernel_mem is None,
            compact_weights=self.compact_weights,
            measure_energy=self.measure_energy,
            timer=self.timer,
            mexpress=self.mexpress,
            pll=self.pll,
            boost=self.boost,
            forever=self.forever,
            fifo=self.fifo,
            groups=self.groups,
            embedded_arm=self.embedded_arm,
            clock_trim=self.clock_trim,
            oneshot=self.oneshot,
            softmax=self.softmax,
            stopstart=self.stopstart,
            num_classes=self.num_classes,
            output_width=self.output_width,
            bias=self.bias,
            verify_kernels=self.verify_kernels,
            wfi=self.wfi,
        )

    def fc_layer(
            self,
            *args,
            **kwargs,
    ):
        """
        Write call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        """
        toplevel.fc_layer(self.memfile, self.weight_header, *args, **kwargs)

    def fc_verify(
            self,
            data,
    ):
        """
        Write the code to verify the fully connected layer against `data`.
        """
        toplevel.fc_verify(self.memfile, self.sampledata_header, data)

    def unload(
            self,
            processor_map,
            input_shape,
            output_offset=0,
            out_expand=1,
            out_expand_thresh=64,
            output_width=8,
            pool=None,
            pool_stride=1,
            mlator=False,
            write_gap=0,
    ):
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        unload.unload(self.apifile or self.memfile, self.apb_base, processor_map, input_shape,
                      output_offset, out_expand, out_expand_thresh, output_width,
                      pool=pool, pool_stride=pool_stride, device=self.device,
                      mlator=mlator, blocklevel=self.blocklevel)

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
        block_level=False,
        debug_mem=False,
        **kwargs,
):
    """
    Depending on `block_level` and `debug_mem`, return a block level .mem file writer,
    a top level .c file writer or a debug writer.
    """
    if not debug_mem:
        APBClass = APBBlockLevel if block_level else APBTopLevel
    else:
        APBClass = APBDebug
    return APBClass(
        *args,
        **kwargs,
    )
