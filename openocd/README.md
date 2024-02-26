# Pre-built OpenOCD

The [MSDK](https://analogdevicesinc.github.io/msdk/) includes debugger support.

For custom installations, and your convenience, this folder contains the files and binaries needed to run OpenOCD for MAX7800X on macOS Catalina/Big Sur/Ventura/Sonoma and on Ubuntu Linux 20.04/22.04 LTS, as well as 32-bit and 64-bit Raspberry Pi OS 10 “Buster” or 11 “Bullseye”.



## Using the Pre-built OpenOCD Binaries

The `run-openocd-maxdap` and `run-openocd-olimex` scripts automatically select an appropriate binary and run OpenOCD with all arguments required for MAX78000. `run-openocd-maxdap-02` is designed for MAX78002.

### Linux

On Linux, several packages are required:

```shell
$ sudo apt-get install libusb-1.0 libusb-0.1 libhidapi-libusb0 libhidapi-hidraw0
```

### macOS

On macOS, both the command line developer tools and [Homebrew](https://brew.sh) must be installed. Follow the instructions to set up your shell.

The following additional packages are also required:

```shell
% brew install libusb-compat libftdi hidapi libusb
```



## Building from Scratch

To obtain the full OpenOCD source code, and to re-build the binaries, please use `git clone -b release https://github.com/analogdevicesinc/openocd.git`.

### Linux

```shell
$ ./bootstrap
$ ./configure  # Linux only, see below for macOS
$ make
$ sudo make install  # or use binary from src/
```

### macOS

On macOS, the following packages are needed to build the binary:

```shell
% brew install autoconf autoconf-archive automake libtool pkg-config
```

You may need to add arguments to `./configure`. The exact paths may vary depending on the MacOSX.sdk release.

```shell
% autoreconf -i -I /opt/homebrew/share/aclocal  # on M1 only, and ignore errors
% ./bootstrap
% CFLAGS=-D_FTDI_DISABLE_DEPRECATED ./configure --prefix=/Library/Developer/CommandLineTools/usr \
--with-gxx-include-dir=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1
% make
% sudo make install  # or use binary from src/
```

