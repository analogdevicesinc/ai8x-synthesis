#!/bin/sh
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 2 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-q4-16x16avgpool --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-3x3s2p2avgpool --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file tests/test-ai85-cifar10-hwc-3x3s2p2avgpool.yaml --stop-after 0 --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-3x3s1avgpool --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling3x3s1.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-4x4s2avgpool --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling4x4s2.yaml --stop-after 0 --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-wideout --checkpoint-file trained/ai85-mnist-wide.pth.tar --config-file tests/test-ai85-mnistwide.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-80wideout --checkpoint-file trained/ai85-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-80wideout-q4 --checkpoint-file trained/ai85-mnist-80wide-q4.pth.tar --config-file tests/test-ai85-mnist80wide-q4.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-80wideout --checkpoint-file trained/ai85-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 1 --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-80wideout-q4-32bit --checkpoint-file trained/ai85-mnist-80expansion-q4.pth.tar --config-file tests/test-ai85-80expansion-q4-32bitout.yaml --stop-after 1 --ai85
