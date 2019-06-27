#!/bin/sh
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix mnist --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml --stop-after 0 --verify-writes
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix fmnist --checkpoint-file trained/ai84-fashionmnist.pth.tar --config-file networks/fashionmnist-hwc.yaml
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar-bias --checkpoint-file trained/ai84-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix cifar-bias --checkpoint-file trained/ai84-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix shift1-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-hwc-shift1.yaml --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix shift2-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-hwc-shift2.yaml --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix outoffs-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-hwc-outputoffset.yaml

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 2
