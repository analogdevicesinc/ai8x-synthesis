#!/bin/sh
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file networks/mnist-chw-smallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-SpeechCom --checkpoint-file trained/ai84-speechcom-net7.pth.tar --config-file networks/speechcom-chw.yaml --fc-layer --embedded-code --cmsis-software-nn
./cnn-gen.py --top-level cnn --test-dir demos --prefix CMSIS-CIFAR-Conv1x1 --checkpoint-file trained/ai85-cifar10-1x1.pth.tar --config-file tests/test-ai85-cifar10-hwc-1x1.yaml --embedded-code --cmsis-software-nn --ai85
