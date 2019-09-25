###################################################################################################
# Copyright (C) 2018-2019 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Routines to read and write the APB peripherals.
"""
import sys

import toplevel
import tornadocnn
import unload


READ_TIME_NS = 230
WRITE_TIME_NS = 280


class APB(object):
    """
    APB read and write functionality.
    """

    def __init__(self,
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
                 device=84):
        """
        Create an APB class object that writes to memfile.
        """
        self.memfile = memfile
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

        self.data = 0
        self.num = 0
        self.data_offs = 0
        self.mem = [None] * tornadocnn.dev.C_GROUP_OFFS * tornadocnn.dev.P_NUMGROUPS
        self.writes = 0
        self.reads = 0

    def get_time(self):
        """
        Return total bus access time in ms based on number of writes and reads
        """
        return (WRITE_TIME_NS * self.writes + READ_TIME_NS * self.reads) // 1000000

    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the output file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        raise NotImplementedError

    def verify(self,
               addr,
               val,
               num_bytes=4,
               comment='',
               rv=False):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        """
        raise NotImplementedError

    def set_memfile(self,
                    memfile):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        self.memfile = memfile

    def write_ctl(self, group, reg, val, debug=False, force_write=False, comment=''):
        """
        Set global control register `reg` in group `group` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // global ctl {reg}'
        if val == 0:
            comment += ' *'
        addr = tornadocnn.dev.C_GROUP_OFFS*group + tornadocnn.dev.C_CNN_BASE + reg*4
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, comment)
        if debug:
            print(f'R{reg:02} ({addr:08x}): {val:08x}{comment}')

    def write_lreg(self, group, layer, reg, val, debug=False, force_write=False, comment=''):
        """
        Set layer `layer` register `reg` in group `group` to value `val`.
        Unless `force_write` is set, zero values will not be written.
        """
        if comment is None:
            comment = f' // reg {reg}'
        if val == 0:
            comment += ' *'
        addr = tornadocnn.dev.C_GROUP_OFFS*group + tornadocnn.dev.C_CNN_BASE \
            + tornadocnn.dev.C_CNN*4 + reg*4 * tornadocnn.dev.MAX_LAYERS + layer*4
        if force_write or val != 0 or self.write_zero_regs:
            self.write(addr, val, comment)
        if debug:
            print(f'L{layer} G{group} R{reg:02} ({addr:08x}): {val:08x}{comment}')

    def write_bias(self, group, offs, bias):
        """
        Write bias value `bias` to offset `offs` in bias memory #`group`.
        """
        addr = tornadocnn.dev.C_GROUP_OFFS*group + tornadocnn.dev.C_BRAM_BASE + offs * 4
        self.write(addr, bias & 0xff, f' // Bias')

    def write_tram(self, group, proc, offs, d, comment=''):
        """
        Write value `d` to TRAM in group `group` and processor `proc` to offset `offs`.
        """
        addr = tornadocnn.dev.C_GROUP_OFFS*group + tornadocnn.dev.C_TRAM_BASE \
            + proc * tornadocnn.dev.TRAM_OFFS * 4 + offs * 4
        self.write(addr, d, f' // {comment}TRAM G{group} P{proc} #{offs}')

    def write_kern(self, ll, p, idx, k, size=9):
        """
        Write single kernel `k` of length `size` for layer `ll`, processor `p` to index `idx` in
        weight memory.
        """
        assert p < tornadocnn.dev.MAX_PROC
        assert idx < tornadocnn.dev.MASK_WIDTH
        addr = tornadocnn.dev.C_GROUP_OFFS * (p // tornadocnn.dev.P_NUMPRO) \
            + tornadocnn.dev.C_MRAM_BASE \
            + (p % tornadocnn.dev.P_NUMPRO) * tornadocnn.dev.MASK_OFFS * 16 + idx * 16

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
        if self.verify_writes:
            self.verify(addr, k[0] & 0xff)
            if size != 1:
                self.verify(addr+4, (k[1] & 0xff) << 24 | (k[2] & 0xff) << 16 |
                            (k[3] & 0xff) << 8 | k[4] & 0xff)
                self.verify(addr+8, (k[5] & 0xff) << 24 | (k[6] & 0xff) << 16 |
                            (k[7] & 0xff) << 8 | k[8] & 0xff)
            else:
                self.verify(addr+4, 0)
                self.verify(addr+8, 0)
            self.verify(addr+12, 0)

    def check_overwrite(self, offs):
        """
        Check whether we're overwriting location `offs`.
        """
        if self.mem[offs >> 2]:
            print(f'Overwriting location {offs:08x}')
            if not self.no_error_stop:
                sys.exit(1)

    def write_byte_flush(self, offs, comment=''):
        """
        Flush the contents of the internal buffer at offset `offs`, adding an optional
        `comment` to the output.
        This function also keeps track of all addresses that have been written before and
        can detect whether previous information is being overwritten.
        """
        if self.num > 0:
            woffs = self.data_offs - self.num
            self.check_overwrite(woffs)
            self.write(woffs, self.data, comment)
            self.mem[woffs >> 2] = True
            self.num = 0
            self.data = 0
        self.data_offs = offs

    def write_byte(self, offs, val, comment=''):
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
            self.write_byte_flush(offs+1, comment)

    def get_mem(self):
        """
        Return reference to the memory array.
        """
        return self.mem

    def output(self, comment):
        """
        Write the string `comment` to the output file without further interpretation.
        """
        if self.memfile is None:
            return

        self.memfile.write(comment)

    def copyright_header(self):
        """
        Write copyright headers.
        The base class does nothing.
        """
        return

    def header(self):
        """
        Write file headers.
        The base class does nothing.
        """
        return

    def verify_header(self):
        """
        Write the header for the CNN verification function.
        The base class does nothing.
        """
        return

    def verify_footer(self):
        """
        Write the footer for the CNN verification function.
        The base class does nothing.
        """
        return

    def load_header(self):
        """
        Write the header for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def load_footer(self):
        """
        Write the footer for the CNN configuration loader function.
        The base class does nothing.
        """
        return

    def main(self, classification_layer=False):  # pylint: disable=unused-argument
        """
        Write the main function, including an optional call to the fully connected layer if
        `classification_layer` is `True`.
        The base class does nothing.
        """
        return

    def fc_layer(self, weights, bias):  # pylint: disable=unused-argument
        """
        Write the call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        The base class does nothing.
        """
        return

    def fc_verify(self, data):  # pylint: disable=unused-argument
        """
        Write the code to verify the fully connected layer against `data`.
        The base class does nothing.
        """
        return

    def unload(self, processor_map, input_shape,  # pylint: disable=unused-argument
               output_offset=0, out_expand=1,  # pylint: disable=unused-argument
               out_expand_thresh=64, output_width=8,  # pylint: disable=unused-argument
               pool=None, pool_stride=1):  # pylint: disable=unused-argument
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        The base class does nothing.
        """
        return

    def verify_unload(self, ll, in_map, out_map, out_buf,  # pylint: disable=unused-argument
                      processor_map, input_shape,  # pylint: disable=unused-argument
                      out_offset=0, out_expand=1,  # pylint: disable=unused-argument
                      out_expand_thresh=64, output_width=8,  # pylint: disable=unused-argument
                      pool=None, pool_stride=1,  # pylint: disable=unused-argument
                      overwrite_ok=False, no_error_stop=False):  # pylint: disable=unused-argument
        """
        Write a verification function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        The base class does nothing.
        """
        return

    def output_define(self, array, define_name,  # pylint: disable=unused-argument
                      fmt, columns, weights=True):  # pylint: disable=unused-argument
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
    def __init__(self,
                 memfile,
                 apb_base,
                 verify_writes=False,
                 no_error_stop=False,
                 weight_header=None,
                 sampledata_header=None,
                 compact_weights=False,
                 compact_data=False,
                 embedded_code=False,
                 weight_filename=None,
                 sample_filename=None):
        super(APBBlockLevel, self).__init__(memfile,
                                            apb_base,
                                            verify_writes=verify_writes,
                                            no_error_stop=no_error_stop,
                                            weight_header=weight_header,
                                            sampledata_header=sampledata_header,
                                            embedded_code=embedded_code,
                                            compact_weights=False,
                                            compact_data=False,
                                            weight_filename=None,
                                            sample_filename=None)
        self.foffs = 0

    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):  # pylint: disable=unused-argument
        """
        Write address `addr` and data `val` to the .mem file.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        self.memfile.write(f'@{self.foffs:04x} {addr:08x}\n')
        self.memfile.write(f'@{self.foffs+1:04x} {val:08x}\n')
        self.foffs += 2

    def verify(self,
               addr,
               val,
               num_bytes=4,
               comment='',
               rv=False):  # pylint: disable=unused-argument
        """
        Verify that memory at address `addr` contains data `val`.
        For block level tests, this function does nothing useful other than ensuring the inputs
        address and data are not negative.
        """
        assert val >= 0
        assert addr >= 0

    def set_memfile(self, memfile):
        """
        Change the file handle to `memfile` and reset the .mem output location to 0.
        """
        super(APBBlockLevel, self).set_memfile(memfile)
        self.foffs = 0


class APBTopLevel(APB):
    """
    APB read and write functionality for top level tests.
    """
    def write(self,
              addr,
              val,
              comment='',
              no_verify=False):
        """
        Write address `addr` and data `val` to the .c file.
        if `no_verify` is `True`, do not check the result of the write operation, even if
        `verify_writes` is globally enabled.
        An optional `comment` can be added to the output.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        if self.memfile is None:
            return

        self.memfile.write(f'  *((volatile uint32_t *) 0x{addr:08x}) = 0x{val:08x};{comment}\n')
        self.writes += 1
        if self.verify_writes and not no_verify:
            self.memfile.write(f'  if (*((volatile uint32_t *) 0x{addr:08x}) != 0x{val:08x}) '
                               'return 0;\n')
            self.reads += 1

    def verify(self,
               addr,
               val,
               num_bytes=4,
               comment='',
               rv=False):
        """
        Verify that memory at address `addr` contains data `val`.
        If `rv` is `True`, do not immediately return 0, but just set the status word.
        An optional `comment` can be added to the output.
        """
        assert val >= 0
        assert addr >= 0
        addr += self.apb_base

        if self.memfile is None:
            return

        if num_bytes == 4:
            mask = ''
        elif num_bytes == 3:
            mask = ' & 0xffffff'
        elif num_bytes == 2:
            mask = ' & 0xffff'
        elif num_bytes == 1:
            mask = ' & 0xff'
        else:
            raise NotImplementedError

        if rv:
            action = 'rv = 0;'
        else:
            action = 'return 0;'

        self.memfile.write(f'  if ((*((volatile uint32_t *) 0x{addr:08x}){mask})'
                           f' != 0x{val:0{2*num_bytes}x}) '
                           f'{action}{comment}\n')
        self.reads += 1

    def copyright_header(self):
        """
        Write copyright headers.
        """
        toplevel.copyright_header(self.memfile)

    def header(self):
        """
        Write include files and forward definitions to .c file.
        """
        toplevel.header(self.memfile,
                        self.apb_base,
                        embedded_code=self.embedded_code,
                        compact_weights=self.compact_weights,
                        compact_data=self.compact_data,
                        weight_filename=self.weight_filename,
                        sample_filename=self.sample_filename)

    def verify_header(self):
        """
        Write the header for the CNN verification function.
        """
        if self.memfile is None:
            return
        toplevel.verify_header(self.memfile)

    def verify_footer(self):
        """
        Write the footer for the CNN verification function.
        """
        if self.memfile is None:
            return
        toplevel.verify_footer(self.memfile)

    def load_header(self):
        """
        Write the header for the CNN configuration loader function.
        """
        toplevel.load_header(self.memfile)

    def load_footer(self):
        """
        Write the footer for the CNN configuration loader function.
        """
        toplevel.load_footer(self.memfile, embedded_code=self.embedded_code)

    def main(self, classification_layer=False):
        """
        Write the main function, including an optional call to the fully connected layer if
        `classification_layer` is `True`.
        """
        toplevel.main(self.memfile, classification_layer=classification_layer,
                      embedded_code=self.embedded_code)

    def fc_layer(self, weights, bias):
        """
        Write call to the fully connected layer for the given `weights` and
        `bias`. The `bias` argument can be `None`.
        """
        toplevel.fc_layer(self.memfile, self.weight_header, weights, bias)

    def fc_verify(self, data):
        """
        Write the code to verify the fully connected layer against `data`.
        """
        toplevel.fc_verify(self.memfile, self.sampledata_header, data)

    def unload(self, processor_map, input_shape, output_offset=0,
               out_expand=1, out_expand_thresh=64, output_width=8,
               pool=None, pool_stride=1):
        """
        Write the unload function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        """
        unload.unload(self.memfile, self.apb_base, processor_map, input_shape,
                      output_offset, out_expand, out_expand_thresh, output_width,
                      pool=pool, pool_stride=pool_stride, device=self.device)

    def verify_unload(self, ll, in_map, out_map, out_buf,
                      processor_map, input_shape,
                      out_offset=0, out_expand=1,
                      out_expand_thresh=64, output_width=8,
                      pool=None, pool_stride=1,
                      overwrite_ok=False, no_error_stop=False):
        """
        Write a verification function. The layer to unload has the shape `input_shape`,
        and the optional `output_offset` argument can shift the output.
        The base class does nothing.
        """
        unload.verify(self.verify, ll, in_map, out_map, out_buf,
                      processor_map, input_shape, out_offset, out_expand, out_expand_thresh,
                      output_width, pool=pool, pool_stride=pool_stride,
                      overwrite_ok=overwrite_ok, no_error_stop=no_error_stop,
                      device=self.device)

    def output_define(self, array, define_name, fmt, columns, weights=True):
        """
        Write a #define for array `array` to `define_name`, using format `fmt` and creating
        a line break after `columns` items each.
        If `weight`, write to the `weights.h` file, else to `sampledata.h`.
        """
        if weights:
            toplevel.c_define(self.weight_header, array, define_name, fmt, columns)
        else:
            toplevel.c_define(self.sampledata_header, array, define_name, fmt, columns)


def apbwriter(memfile,
              apb_base,
              block_level=False,
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
              device=84):
    """
    Depending on `block_level`, return a block level .mem file writer or a top level .c file
    writer to the file `memfile` with APB base address `apb_base`.
    If `verify_writes` is set, read back everything that was written.
    If `no_error_stop` is set, continue in the case when the data is trying to overwrite
    previously written data.
    """
    APBClass = APBBlockLevel if block_level else APBTopLevel
    return APBClass(memfile,
                    apb_base,
                    verify_writes=verify_writes,
                    no_error_stop=no_error_stop,
                    weight_header=weight_header,
                    sampledata_header=sampledata_header,
                    embedded_code=embedded_code,
                    compact_weights=compact_weights,
                    compact_data=compact_data,
                    write_zero_registers=write_zero_registers,
                    weight_filename=weight_filename,
                    sample_filename=sample_filename,
                    device=device)
