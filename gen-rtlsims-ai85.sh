#!/bin/sh
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-fmnist --checkpoint-file trained/ai85-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml --stop-after 0 --ai85 --verify-writes

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-q4-cifar-bias --checkpoint-file trained/ai85-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 1 --ai85
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

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d --config-file tests/test-ai85-conv1d.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-1 --config-file tests/test-conv1d-1.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-2 --config-file tests/test-conv1d-2.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-3 --config-file tests/test-conv1d-3.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-4 --config-file tests/test-conv1d-4.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-5 --config-file tests/test-conv1d-5.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-6 --config-file tests/test-conv1d-6.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-7 --config-file tests/test-conv1d-7.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-8 --config-file tests/test-conv1d-8.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1d-9 --config-file tests/test-conv1d-9.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-conv1x1 --config-file tests/test-conv1x1.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar-conv1x1 --checkpoint-file trained/ai85-cifar10-1x1.pth.tar --config-file tests/test-ai85-cifar10-hwc-1x1.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-nonsquare --config-file tests/test-nonsquare.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-nonsquare-pool --config-file tests/test-nonsquare-pool.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-nonsquare-nonsquarepool --config-file tests/test-nonsquare-nonsquarepool.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-mnist --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-speechcom --checkpoint-file trained/ai84-speechcom-net7.pth.tar --config-file networks/speechcom-chw.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 0 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 1 --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 2 --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-energy --config-file tests/test-energy.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-16x16s4 --config-file tests/test-pooling16x16s4.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-16x16s4wide --config-file tests/test-pooling16x16s4wide.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-16x16s7 --config-file tests/test-pooling16x16s7.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-16x16s16 --config-file tests/test-pooling16x16s16.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool2-16x16s7 --config-file tests/test-pooling16x16s7-II.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool2-16x16s16 --config-file tests/test-pooling16x16s16-II.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-1x13s13 --config-file tests/test-pooling1x13s13.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool2-1x13s13 --config-file tests/test-pooling1x13s13-II.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-13x1s1 --config-file tests/test-pooling13x1s1.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool2-13x1s1 --config-file tests/test-pooling13x1s1-II.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-2x3s2 --config-file tests/test-pooling2x3s2.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-3x2s3 --config-file tests/test-pooling3x2s3.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-3x3s2 --config-file tests/test-pooling3x3s2.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-4x3s3 --config-file tests/test-pooling3x3s3.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-4x4s1 --config-file tests/test-pooling4x4s1.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-4x4s3 --config-file tests/test-pooling4x4s3.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-pool-4x4s4 --config-file tests/test-pooling4x4s4.yaml --ai85

./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --ai85
./cnn-gen.py --verbose --autogen rtlsim --top-level cnn -L --test-dir rtlsim --prefix ai85-singlebyte-chw --config-file tests/test-singlebyte-chw.yaml --ai85
