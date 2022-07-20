# CMSIS-NN Code Generator

The ‘izer’ includes an unsupported CMSIS-NN code generator. To use it:

1. Understand it is incomplete and unsupported.
2. Use only networks **without any** input sequences (concatenation) and without element-wise operations. Some or more of these features could be added without too much effort, any suggestions or pull requests are welcome.
3. Understand that there is no proper build environment.

## Setup

Install a copy of CMSIS_5 at the same level as ai8x-synthesis.

```shell
(ai8x-synthesis) $ cd ..
(ai8x-synthesis) $ git clone git@github.com:ARM-software/CMSIS_5.git
(ai8x-synthesis) $ cd CMSIS_5
(ai8x-synthesis) $ git checkout master
(ai8x-synthesis) $ cd ../ai8x-synthesis
```

There are additional files in the `assets/cmsis-nn` folder of ai8x-synthesis. These files are added to all projects built by the ‘izer’.

## Generating C Code

To generate C code, use the special “CMSIS-NN” device (instead of MAX78000 or MAX78002). For example, to generate a CIFAR-10 demo for CMSIS-NN, run:

```shell
(ai8x-synthesis) $ python ai8xize.py --verbose --test-dir rtldev/cmsis-demos --prefix cifar-10 --checkpoint-file trained/ai85-cifar10.pth.tar --config-file networks/cifar10-hwc-ai85.yaml --device CMSIS-NN --display-checkpoint
```

This is very similar to generating code for MAX78000 (see `gen-demos-max7800.sh`).

Next, go to the target folder (`rtldev/cmsis-demos/cifar-10` in the above example), and execute:

```shell
(ai8x-synthesis) $ cd rtldev/cmisis-demos/cifar-10
(ai8x-synthesis) $ ./makelinks.sh
(ai8x-synthesis) $ make
```

This builds an executable file called `main` which will run a known-answer test on the host:

```shell
(ai8x-synthesis) $ ./main
*** PASS ***

Output of final layer:
 -128 -128 -128  127  -128  -61 -128 -128  -128 -128
```

To build for the Cortex-M4 core in MAX78000, instead run

```shell
(ai8x-synthesis) $ MAXIM_PATH=../../../sdk make -f Makefile.ARM
```

Note that the SIMD libraries for Arm Cortex-M4 do not implement full rounding, and results may therefore not be an exact match (known-answer test returns “FAIL”).
