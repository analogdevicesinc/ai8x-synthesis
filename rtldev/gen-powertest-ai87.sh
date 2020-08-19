#!/bin/sh
CLOCK_TRIM=""
BOOST=""
SDK_TARGET="sdk/Examples/MAX78002/PowerTest"

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp device-ai87/softmax.c $SDK_TARGET/Common/
cp device-ai87/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_64x64x64 --config-file tests/test-burnin_64x64x64.yaml --device 87 --compact-data --mexpress $CLOCK_TRIM --fixed-input --max-checklines 4096 --repeat-layers 32 --forever $BOOST $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_16x64x64 --config-file tests/test-burnin_16x64x64.yaml --device 87 --compact-data --mexpress $CLOCK_TRIM --fixed-input --max-checklines 4096 --repeat-layers 32 --forever $BOOST $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-rand_64x64x64 --config-file tests/test-burnin-rand_64x64x64.yaml --device 87 --compact-data --mexpress $CLOCK_TRIM --max-checklines 4096 --repeat-layers 32 --forever $BOOST $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-rand_16x64x64 --config-file tests/test-burnin-rand_16x64x64.yaml --device 87 --compact-data --mexpress $CLOCK_TRIM --max-checklines 4096 --repeat-layers 32 --forever $BOOST $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device 87 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --forever $BOOST $@
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-mnist-extrasmall --checkpoint-file trained/ai85-mnist-extrasmall.pth.tar --config-file networks/mnist-chw-extrasmall-ai85.yaml --device 87 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --forever $BOOST $@
