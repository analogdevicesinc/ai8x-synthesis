#!/bin/sh
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-CIFAR-10 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --fc-layer --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-MNIST-ExtraSmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-MNIST-Small --checkpoint-file trained/ai84-mnist-smallnet.pth.tar --config-file networks/mnist-chw-smallnet.yaml --fc-layer --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-MNIST --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --fc-layer --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-SpeechCom --checkpoint-file trained/ai84-speechcom-net7.pth.tar --config-file networks/speechcom-chw.yaml --fc-layer --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-CIFAR-Conv1x1 --checkpoint-file trained/ai85-cifar10-1x1.pth.tar --config-file tests/test-ai85-cifar10-hwc-1x1.yaml --embedded-code --cmsis-software-nn --ai85

./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-Conv1D --config-file tests/test-conv1d.yaml --embedded-code --cmsis-software-nn
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-Conv1x1 --config-file tests/test-conv1x1.yaml --embedded-code --cmsis-software-nn --ai85

./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-Nonsquare --config-file tests/test-nonsquare.yaml --embedded-code --cmsis-software-nn --ai85
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-NonsquarePool --config-file tests/test-nonsquare-pool.yaml --embedded-code --cmsis-software-nn --ai85
./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-NonsquarePoolNonsquare --config-file tests/test-nonsquare-nonsquarepool.yaml --embedded-code --cmsis-software-nn --ai85

./ai8xize.py --verbose -L --top-level cnn --test-dir demos --prefix CMSIS-Energy --config-file tests/test-energy.yaml --embedded-code --cmsis-software-nn

