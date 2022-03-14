# Development on Raspberry Pi 4 and 400

July 1, 2021

## Requirements and Performance

Software development for MAX78000 is possible on a 64-bit Raspberry Pi OS with the following restrictions:

* A 64-bit OS is required, since the embedded C compiler and PyTorch are only available for 64-bit systems. While the Raspberry Pi 3 does have a 64-bit CPU, the Raspberry Pi 4 and 400 with fan are strongly recommended for performance reasons.
  As operating system, use either <https://downloads.raspberrypi.org/raspios_lite_arm64/images/>
  or <https://downloads.raspberrypi.org/raspios_arm64/images/>.
* Performance for running the synthesis tools is bounded by disk I/O. Consider using a USB 3-connected SSD as mass storage, since I/O performance on SD cards and USB Flash drives will suffer from long latencies.
* The installation will take more time than on an x86_64 system, since more Python “wheels” have to be built from source during installation compared to x86_64.
* Since there is no CUDA on Raspberry Pi, model training will be very slow and a CPU fan is required.

## Installation

First, install the following additional packages on the Raspberry Pi:

```shell
$ sudo apt-get install screen libtool git libusb-1.0-0-dev libgpiod-dev libhidapi-dev \
libftdi-dev zip libbz2-dev libssl-dev libreadline-dev libsqlite3-dev libatlas-base-dev \
libopenblas-dev libopenmpi-dev libomp-dev libncurses5 ninja-build cmake ccache \
libblas-dev libeigen3-dev libprotobuf-dev protobuf-compiler llvm-dev
```

Then follow the instructions for Ubuntu Linux.

### OpenOCD

The pre-built binary in the `openocd` folder of the `ai8x-synthesis` project works with MAX78000.

To avoid having to use superuser privileges to run OpenOCD, follow the instructions on <https://forgge.github.io/theCore/guides/running-openocd-without-sudo.html>.

### Embedded Arm Compiler

The embedded C compiler version 10-2020-q4-major for “aarch64” works on Raspberry Pi OS 64-bit (`gcc-arm-none-eabi-10-2020-q4-major-x86_64-linux.tar.bz2`).

### Embedded RISC-V Compiler

The embedded C compiler version v10.1.0-1.1 for “arm64” works on Raspberry Pi OS 64-bit (`xpack-riscv-none-embed-gcc-10.1.0-1.1-linux-arm64.tar.gz`).

### Training and Synthesis Projects

Both model training and C code generation will work on a Raspberry Pi 64-bit. Note that model training will be very slow without CUDA.

