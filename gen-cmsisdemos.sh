#!/bin/sh
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file networks/mnist-chw-smallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
