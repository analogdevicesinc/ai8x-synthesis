#!/bin/sh
SDK_TARGET="../AI84SDK/Firmware/trunk/Applications/EvKitExamples"

# Update common files
cp device-ai84/*.c $SDK_TARGET/Common/
cp device-ai84/tornadocnn.h $SDK_TARGET/Common/

# Create SDK demos
./ai8xize.py --verbose -L --top-level cnn --test-dir $SDK_TARGET --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --fc-layer --embedded-code
./ai8xize.py --verbose -L --top-level cnn --test-dir $SDK_TARGET --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code
./ai8xize.py --verbose -L --top-level cnn --test-dir $SDK_TARGET --prefix MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code
./ai8xize.py --verbose -L --top-level cnn --test-dir $SDK_TARGET --prefix MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file networks/mnist-chw-smallnet.yaml --fc-layer --embedded-code
./ai8xize.py --verbose -L --top-level cnn --test-dir $SDK_TARGET --prefix speechcom --checkpoint-file trained/ai84-speechcom-net7.pth.tar --config-file networks/speechcom-chw.yaml --fc-layer --embedded-code
