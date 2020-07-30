#!/bin/sh
CLOCK_TRIM=""
# Example: CLOCK_TRIM="--clock-trim 0x173,0x5e,0x14"
SDK_TARGET="sdk/Examples/MAX78000/CNN"

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp device-ai85/softmax.c $SDK_TARGET/Common/
cp device-ai85/*.h $SDK_TARGET/Common/

<<<<<<< HEAD:gen-demos-ai85.sh
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist-riscv --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --riscv --riscv-flash --riscv-cache --riscv-debug
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-10 --checkpoint-file trained/ai85-cifar10.pth.tar --config-file networks/cifar10-hwc-ai85.yaml --device 85 --compact-data --mexpress --display-checkpoint $CLOCK_TRIM
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --boost 2.5
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix kws20 --checkpoint-file trained/ai85-kws20.pth.tar --config-file networks/kws20-hwc.yaml --device 85 --compact-data --mexpress --softmax --embedded-code --display-checkpoint $CLOCK_TRIM
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix faceid --checkpoint-file trained/ai85-faceid.pth.tar --config-file networks/faceid.yaml --device 85 --fifo --compact-data --mexpress --display-checkpoint --unload $CLOCK_TRIM
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cats-vs-dogs-chw --checkpoint-file trained/ai85-catsdogs-chw.pth.tar --config-file networks/cats-dogs-chw.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint --embedded-code --boost 2.5
=======
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device MAX78000 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix mnist-riscv --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --device MAX78000 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --riscv --riscv-flash --riscv-cache --riscv-debug $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-10 --checkpoint-file trained/ai85-cifar10.pth.tar --config-file networks/cifar10-hwc-ai85.yaml --device MAX78000 --compact-data --mexpress --display-checkpoint $CLOCK_TRIM $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device MAX78000 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --boost 2.5 $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix kws20 --checkpoint-file trained/ai85-kws20.pth.tar --config-file networks/kws20-hwc.yaml --device MAX78000 --compact-data --mexpress --softmax --embedded-code --display-checkpoint $CLOCK_TRIM $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix faceid --checkpoint-file trained/ai85-faceid.pth.tar --config-file networks/faceid.yaml --device MAX78000 --fifo --compact-data --mexpress --display-checkpoint --unload $CLOCK_TRIM $@
>>>>>>> master:gen-demos-max78000.sh
