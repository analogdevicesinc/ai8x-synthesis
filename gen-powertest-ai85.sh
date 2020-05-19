#!/bin/sh
CLOCK_TRIM=""
SDK_TARGET="sdk/Examples/MAX78000/PowerTest"

# Update common code and header files
cp device-ai85/softmax.c $SDK_TARGET/Common/
cp device-ai85/*.h $SDK_TARGET/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_64x64x64 --config-file tests/test-burnin_64x64x64.yaml --device 85 --compact-data --mexpress $CLOCK_TRIM --fixed-input --max-checklines 4096 --repeat-layers 32 --forever
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin_16x64x64 --config-file tests/test-burnin_16x64x64.yaml --device 85 --compact-data --mexpress $CLOCK_TRIM --fixed-input --max-checklines 4096 --repeat-layers 32 --forever
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-rand_64x64x64 --config-file tests/test-burnin-rand_64x64x64.yaml --device 85 --compact-data --mexpress $CLOCK_TRIM --max-checklines 4096 --repeat-layers 32 --forever
./ai8xize.py -e --verbose --top-level cnn -L --test-dir $SDK_TARGET --prefix burnin-cifar-100 --checkpoint-file trained/ai85-cifar100.pth.tar --config-file networks/cifar100-simple.yaml --device 85 --compact-data --mexpress --softmax --display-checkpoint $CLOCK_TRIM --forever
