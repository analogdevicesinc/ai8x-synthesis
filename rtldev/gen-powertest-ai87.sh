#!/bin/sh
DEVICE="MAX78002"
SDK_TARGET="sdk/Examples/$DEVICE/PowerTest"
EXTRA_ARGS=""
BOOST=""

# Update common code and header files
mkdir -p $SDK_TARGET/Common/
cp assets/device-ai87/softmax.c $SDK_TARGET/Common/
cp assets/device-ai87/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_64x64x64 --config-file tests/test-burnin_64x64x64.yaml --device "$DEVICE" --compact-data --mexpress $EXTRA_ARGS --fixed-input --max-checklines 4096 --repeat-layers 32 --forever $BOOST "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_16x64x64 --config-file tests/test-burnin_16x64x64.yaml --device "$DEVICE" --compact-data --mexpress $EXTRA_ARGS --fixed-input --max-checklines 4096 --repeat-layers 32 --forever $BOOST "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-rand_64x64x64 --config-file tests/test-burnin-rand_64x64x64.yaml --device "$DEVICE" --compact-data --mexpress $EXTRA_ARGS --max-checklines 4096 --repeat-layers 32 --forever $BOOST "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-rand_16x64x64 --config-file tests/test-burnin-rand_16x64x64.yaml --device "$DEVICE" --compact-data --mexpress $EXTRA_ARGS --max-checklines 4096 --repeat-layers 32 --forever $BOOST "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint $EXTRA_ARGS --forever $BOOST "$@"
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-mnist-extrasmall --checkpoint-file trained/ai85-mnist-extrasmall.pth.tar --config-file networks/mnist-chw-extrasmall-ai85.yaml --device "$DEVICE" --compact-data --mexpress --softmax --display-checkpoint $EXTRA_ARGS --forever $BOOST "$@"
