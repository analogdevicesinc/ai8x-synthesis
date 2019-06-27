#!/bin/sh
cp Device/*.c ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/Common/
cp Device/tornadocnn.h ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/Common/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --fc-layer --embedded-code
cp demos/MNIST/* ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/MNIST/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code
cp demos/CIFAR-10/* ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/CIFAR-10/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code
cp demos/MNIST-ExtraSmall/* ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/MNIST-ExtraSmall/

./cnn-gen.py --verbose -L --top-level cnn --test-dir demos --prefix MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file networks/mnist-chw-smallnet.yaml --fc-layer --embedded-code
cp demos/MNIST-Small/* ../AI84SDK/Firmware/trunk/Applications/EvKitExamples/MNIST-Small/
