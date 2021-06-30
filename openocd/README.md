# Pre-built OpenOCD

For your convenience, this folder contains the files and binaries needed to run OpenOCD for MAX78000 on macOS Catalina/Big Sur and on Ubuntu Linux 20.04 LTS, as well as 64-bit Raspberry Pi OS 10 “Buster”.



#### Using the Pre-built OpenOCD Binaries

The `run-openocd-maxdap` and `run-openocd-olimex` scripts automatically select an appropriate binary and run OpenOCD with all arguments required for MAX78000.

On Linux, several packages are required:

```shell
$ sudo apt-get install libusb-1.0 libusb-0.1 libhidapi-libusb0 libhidapi-hidraw0
```

On macOS:

```shell
% brew install libusb-compat libftdi hidapi libusb
```



#### Building from Scratch

To obtain the full OpenOCD source code, and to re-build the binaries, please use `git clone https://github.com/MaximIntegratedMicros/openocd.git`. Then:

```shell
$ ./bootstrap
$ ./configure  # Linux only, see below for macOS
$ make
$ sudo make install
```

On macOS, you may need to add arguments to `./configure`. The exact paths may vary depending on the MacOSX.sdk release.
```shell
% CFLAGS=-D_FTDI_DISABLE_DEPRECATED ./configure --prefix=/Library/Developer/CommandLineTools/usr \
--with-gxx-include-dir=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
```

