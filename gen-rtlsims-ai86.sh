#!/bin/sh
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fmnist --checkpoint-file trained/ai86-fashionmnist.pth.tar --config-file networks/fashionmnist-chw.yaml --stop-after 0 --ai86 --verify-writes

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-bias --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-bias --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-q4-cifar-bias --checkpoint-file trained/ai86-cifar10-bias-quant4.pth.tar --config-file tests/test-ai86-cifar10-hwc-quant4.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-q4-cifar-bias --checkpoint-file trained/ai86-cifar10-bias-quant4.pth.tar --config-file tests/test-ai86-cifar10-hwc-quant4.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 1 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 2 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-q4-16x16avgpool --checkpoint-file trained/ai86-cifar10-bias-quant4.pth.tar --config-file tests/test-ai86-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-3x3s2p2avgpool --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file tests/test-ai86-cifar10-hwc-3x3s2p2avgpool.yaml --stop-after 0 --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-3x3s1avgpool --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling3x3s1.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-4x4s2avgpool --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling4x4s2.yaml --stop-after 0 --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wideout --checkpoint-file trained/ai86-mnist-wide.pth.tar --config-file tests/test-ai86-mnistwide.yaml --stop-after 0 --ai86 --timeout 16
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-80wideout --checkpoint-file trained/ai86-mnist-80wide.pth.tar --config-file tests/test-ai86-mnist80wide.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-80wideout-q4 --checkpoint-file trained/ai86-mnist-80wide-q4.pth.tar --config-file tests/test-ai86-mnist80wide-q4.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-80wideout --checkpoint-file trained/ai86-mnist-80wide.pth.tar --config-file tests/test-ai86-mnist80wide.yaml --stop-after 1 --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-80wideout-q4-32bit --checkpoint-file trained/ai86-mnist-80expansion-q4.pth.tar --config-file tests/test-ai86-80expansion-q4-32bitout.yaml --stop-after 1 --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d --config-file tests/test-ai86-conv1d.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-1 --config-file tests/test-conv1d-1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-2 --config-file tests/test-conv1d-2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-3 --config-file tests/test-conv1d-3.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-4 --config-file tests/test-conv1d-4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-5 --config-file tests/test-conv1d-5.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-6 --config-file tests/test-conv1d-6.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-7 --config-file tests/test-conv1d-7.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-8 --config-file tests/test-conv1d-8.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-9 --config-file tests/test-conv1d-9.yaml --ai86

# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1dII --config-file tests/test-ai86-conv1dII.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-3II --config-file tests/test-conv1d-3II.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-4II --config-file tests/test-conv1d-4II.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-5II --config-file tests/test-conv1d-5II.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-6II --config-file tests/test-conv1d-6II.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-7II --config-file tests/test-conv1d-7II.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-1 --config-file tests/test-conv1d-pool-1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-2 --config-file tests/test-conv1d-pool-2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-3 --config-file tests/test-conv1d-pool-3.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-4 --config-file tests/test-conv1d-pool-4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-5 --config-file tests/test-conv1d-pool-5.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-6 --config-file tests/test-conv1d-pool-6.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-7 --config-file tests/test-conv1d-pool-7.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-8 --config-file tests/test-conv1d-pool-8.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-9 --config-file tests/test-conv1d-pool-9.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-1-q1 --config-file tests/test-conv1d-pool-1-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-2-q1 --config-file tests/test-conv1d-pool-1-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-3-q1 --config-file tests/test-conv1d-pool-3-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-4-q1 --config-file tests/test-conv1d-pool-4-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-5-q1 --config-file tests/test-conv1d-pool-5-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-6-q1 --config-file tests/test-conv1d-pool-6-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-7-q1 --config-file tests/test-conv1d-pool-7-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-8-q1 --config-file tests/test-conv1d-pool-8-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-9-q1 --config-file tests/test-conv1d-pool-9-q1.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-1-q2 --config-file tests/test-conv1d-pool-1-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-2-q2 --config-file tests/test-conv1d-pool-1-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-3-q2 --config-file tests/test-conv1d-pool-3-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-4-q2 --config-file tests/test-conv1d-pool-4-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-5-q2 --config-file tests/test-conv1d-pool-5-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-6-q2 --config-file tests/test-conv1d-pool-6-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-7-q2 --config-file tests/test-conv1d-pool-7-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-8-q2 --config-file tests/test-conv1d-pool-8-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-9-q2 --config-file tests/test-conv1d-pool-9-q2.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-1-q4 --config-file tests/test-conv1d-pool-1-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-2-q4 --config-file tests/test-conv1d-pool-1-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-3-q4 --config-file tests/test-conv1d-pool-3-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-4-q4 --config-file tests/test-conv1d-pool-4-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-5-q4 --config-file tests/test-conv1d-pool-5-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-6-q4 --config-file tests/test-conv1d-pool-6-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-7-q4 --config-file tests/test-conv1d-pool-7-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-8-q4 --config-file tests/test-conv1d-pool-8-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-9-q4 --config-file tests/test-conv1d-pool-9-q4.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-1-wide --config-file tests/test-conv1d-pool-1-wide.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-3-wide --config-file tests/test-conv1d-pool-3-wide.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-5-wide --config-file tests/test-conv1d-pool-5-wide.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-pool-9-wide --config-file tests/test-conv1d-pool-9-wide.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1x1 --config-file tests/test-conv1x1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-conv1x1 --checkpoint-file trained/ai86-cifar10-1x1.pth.tar --config-file tests/test-ai86-cifar10-hwc-1x1.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare --config-file tests/test-nonsquare.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare-pool --config-file tests/test-nonsquare-pool.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare-nonsquarepool --config-file tests/test-nonsquare-nonsquarepool.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-speechcom --checkpoint-file trained/ai84-speechcom-net7.pth.tar --config-file networks/speechcom-chw.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 0 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 1 --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 2 --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-energy --config-file tests/test-energy.yaml --ai86 --timeout 40 --mexpress --compact-data

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-16x16s4 --config-file tests/test-pooling16x16s4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-16x16s4wide --config-file tests/test-pooling16x16s4wide.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-16x16s7 --config-file tests/test-pooling16x16s7.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-16x16s16 --config-file tests/test-pooling16x16s16.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool2-16x16s7 --config-file tests/test-pooling16x16s7-II.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool2-16x16s16 --config-file tests/test-pooling16x16s16-II.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-1x13s13 --config-file tests/test-pooling1x13s13.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool2-1x13s13 --config-file tests/test-pooling1x13s13-II.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-13x1s1 --config-file tests/test-pooling13x1s1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool2-13x1s1 --config-file tests/test-pooling13x1s1-II.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-2x3s2 --config-file tests/test-pooling2x3s2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-3x2s3 --config-file tests/test-pooling3x2s3.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-3x3s2 --config-file tests/test-pooling3x3s2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-4x3s3 --config-file tests/test-pooling3x3s3.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-4x4s1 --config-file tests/test-pooling4x4s1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-4x4s3 --config-file tests/test-pooling4x4s3.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-4x4s4 --config-file tests/test-pooling4x4s4.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-singlebyte-chw --config-file tests/test-singlebyte-chw.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-layers --config-file tests/test-layers.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru --config-file tests/test-passthrough.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru-2 --config-file tests/test-passthrough-2.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru-2a --config-file tests/test-passthrough-2a.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru-pool --config-file tests/test-passthrough-pool.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml --ai86
# ./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthru-2a-pool --config-file tests/test-passthrough-2a-pool.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein --config-file tests/test-widein.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-q1 --config-file tests/test-widein-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-q2 --config-file tests/test-widein-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-q4 --config-file tests/test-widein-q4.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wideout --config-file tests/test-wideout.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wideout-q1 --config-file tests/test-wideout-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wideout-q2 --config-file tests/test-wideout-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wideout-q4 --config-file tests/test-wideout-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512out --config-file tests/test-wide512out.yaml --ai86 --compact-weights --autogen None

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-bias --config-file tests/test-widein-bias.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-bias-q1 --config-file tests/test-widein-bias-q1.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-bias-q2 --config-file tests/test-widein-bias-q2.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-bias-q4 --config-file tests/test-widein-bias-q4.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide256in-bias-q1 --config-file tests/test-wide256in-bias-q1.yaml --ai86 --timeout 128
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512in --config-file tests/test-wide512in.yaml --ai86 --compact-weights --timeout 128 --autogen None
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512in-bias-q2 --config-file tests/test-wide512in-bias-q2.yaml --ai86 --timeout 128
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512in-q1 --config-file tests/test-wide512in-q1.yaml --ai86 --timeout 128 --autogen None
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512in-q2 --config-file tests/test-wide512in-q2.yaml --ai86 --compact-weights --timeout 128 --autogen None
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide512in-q4 --config-file tests/test-wide512in-q4.yaml --ai86 --compact-weights --timeout 128 --autogen None

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-dataonexone --config-file tests/test-dataonexone.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-dataonexone2 --config-file tests/test-dataonexone2.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-maxproc4 --config-file tests/test-widein-maxproc4.yaml --ai86 --max-proc 4
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-maxproc4-q4 --config-file tests/test-widein-maxproc4-q4.yaml --ai86 --max-proc 4

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv2Dk1x1 --config-file tests/test-conv2Dk1x1.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv2Dk1x1-b --config-file tests/test-conv2Dk1x1-b.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv2Dk1x1-b-pool --config-file tests/test-conv2Dk1x1-b-pool.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlp12to2 --config-file tests/test-mlp12to2.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml --ai86 --debug-computation --debug

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-wide3to508to3 --config-file tests/test-wide3to508to3.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai86 --stop-after 1 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai86 --stop-after 2 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai86 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai86 --stop-after 1 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai86 --stop-after 2 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai86 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-transition --config-file tests/test-stream-transition.yaml --ai86 --overwrite-ok

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add --config-file tests/test-eltwiseadd.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-sub --config-file tests/test-eltwisesub.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-xor --config-file tests/test-eltwisexor.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-or --config-file tests/test-eltwiseor.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add-multipass --config-file tests/test-eltwiseadd-multipass.yaml --ai86 --max-proc 2 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add7 --config-file tests/test-eltwiseadd-7ch.yaml --ai86 --legacy-test

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add4-7ch --config-file tests/test-eltwiseadd4-7ch.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add7-conv2d --config-file tests/test-eltwiseaddconv2d-7ch.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add31 --config-file tests/test-eltwiseadd-31ch.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add126 --config-file tests/test-eltwiseadd-126ch.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-poolafter --config-file tests/test-eltwiseadd-poolafter.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add5op31 --config-file tests/test-eltwiseadd-5op-31ch.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-add5op31-32bit --config-file tests/test-eltwiseadd-5op-31ch-32bit.yaml --ai86 --legacy-test

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-passthroughmp --config-file tests/test-passthroughmultipass.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-widein-1x1 --config-file tests/test-widein-1x1.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-pool-avg --config-file tests/test-eltwiseadd-pool-avg.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml --ai86 --legacy-test
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-q4-16x16avgpool-round --checkpoint-file trained/ai86-cifar10-bias-quant4.pth.tar --config-file tests/test-ai86-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai86 --avg-pool-rounding

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai86 --fifo --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai86 --fifo --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2 --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai86 --fifo --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-mlp --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-chw-mlp.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar-transition --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --ai86 --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-stream-cifar-transition-zeroize --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --ai86 --zero-sram --overwrite-ok
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml --ai86 --fifo --stop-after 5

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlator --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai86 --mlator --mexpress --mlator-noverify --compact-data
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlator --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 1 --ai86 --mlator --mexpress --mlator-noverify --compact-data
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlator --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 2 --ai86 --mlator

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-abs --config-file tests/test-conv1d-abs.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-abs --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-cifar10-abs.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten12to17 --config-file tests/test-mlpflatten12to17.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten12to17big --config-file tests/test-mlpflatten12to17-big.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten12to100 --config-file tests/test-mlpflatten12to100.yaml --ai86 --debug-computation --debug
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten12to100big --config-file tests/test-mlpflatten12to100-big.yaml --ai86 --debug-computation --debug

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten192to10 --config-file tests/test-mlpflatten192to10.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten192to10big --config-file tests/test-mlpflatten192to10-big.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten768to10 --config-file tests/test-mlpflatten768to10.yaml --ai86 --debug --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten768to10big --config-file tests/test-mlpflatten768to10-big.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten768to100 --config-file tests/test-mlpflatten768to100.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten768to100big --config-file tests/test-mlpflatten768to100-big.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall-oneshot --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 2 --ai86 --one-shot
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall-stopstart --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 2 --ai86 --stop-start

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall-cweight --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 0 --ai86 --compact-weights --verify-kernels
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mnist-extrasmall-mexpress --checkpoint-file trained/ai84-mnist-extrasmallnet.pth.tar --config-file networks/mnist-chw-extrasmallnet.yaml --stop-after 0 --ai86 --mexpress --verify-kernels
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-cifar-bias-mexpress --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --stop-after 0 --ai86 --mexpress --verify-kernels
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-q4-cifar-bias-mexpress --checkpoint-file trained/ai86-cifar10-bias-quant4.pth.tar --config-file tests/test-ai86-cifar10-hwc-quant4.yaml --stop-after 0 --ai86 --mexpress --verify-kernels

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-mlpflatten768to100big-q4 --config-file tests/test-mlpflatten768to100-big-q4.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml --ai86 --fifo --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml --ai86 --fifo --debug-computation --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-nonsquare-hwc --config-file tests/test-fifostream-nonsquare-hwc.yaml --ai86 --fifo --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-pool --config-file tests/test-fifostream-pool.yaml --ai86 --fifo --debug-computation --override-start 0x08 --override-rollover 0x24 --override-delta2 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-pool-hwc --config-file tests/test-fifostream-pool-hwc.yaml --ai86 --fifo --debug-computation --override-start 0x1b --override-rollover 0x1c --override-delta2 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-15ch-hwc --config-file tests/test-fifostream-15ch-hwc.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-16ch-hwc --config-file tests/test-fifostream-16ch-hwc.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640 --config-file tests/test-fifostream-640.yaml --ai86 --fifo --mexpress --timeout 40
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640 --config-file tests/test-fifostream-640.yaml --ai86 --fifo --stop-after 1 --mexpress --timeout 40
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai86 --fifo --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai86 --fifo --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare --config-file tests/test-nonsquare.yaml --ai86 --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare --config-file tests/test-nonsquare.yaml --ai86 --debug-computation --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-pool-stride --config-file tests/test-fifostream-pool-stride.yaml --ai86 --fifo --debug-computation --override-start 0x07 --override-rollover 0x38 --override-delta2 0x04
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-pool-stride-hwc --config-file tests/test-fifostream-pool-stride-hwc.yaml --ai86 --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 3
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 4
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai86 --stop-after 5
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 2
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 3
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 4
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --stop-after 5

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-pool-stride-hwc-slow --config-file tests/test-fifostream-pool-stride-hwc.yaml --ai86 --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b --slow-load 8

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-outputshift --config-file tests/test-outputshift.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai86 --fifo --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai86 --fifo --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml --ai86 --fifo --mexpress --timeout 40
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml --ai86 --fifo --stop-after 1 --mexpress --timeout 40
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-pool-4high --config-file tests/test-pool-4high.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upsample --config-file tests/test-upsample.yaml --ai86 --debug --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale --config-file tests/test-upscale.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale --config-file tests/test-upscale.yaml --ai86 --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale --config-file tests/test-upscale.yaml --ai86 --stop-after 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale-pro --config-file tests/test-upscale-pro.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale-pro --config-file tests/test-upscale-pro.yaml --ai86 --stop-after 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-upscale-pro --config-file tests/test-upscale-pro.yaml --ai86 --stop-after 1

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-helloworld --config-file tests/test-pooling13x1s1.yaml --ai86 --riscv-flash
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml --ai86 --compact-data --mexpress --riscv-flash --riscv-cache
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-flash-cache-helloworld --config-file tests/test-pooling13x1s1.yaml --ai86 --riscv-cache
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-flash-cache-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml --ai86 --riscv-cache --mexpress --compact-data

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifo-nonsquare --config-file tests/test-fifo-nonsquare.yaml --ai86 --fifo --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai86 --fifo --debug-computation
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai86 --fifo --input-csv input.csv

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-hwc-nonsquare --config-file tests/test-fifostream-nonsquare-hwc.yaml --ai86 --fifo --input-csv input.csv --riscv-cache
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml --ai86 --fifo --stop-after 5 --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-vga-hwc --config-file tests/test-fifostream-vga-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1 
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-vga2-hwc --config-file tests/test-fifostream-vga2-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-group0-pool4 --config-file tests/test-group0-pool4.yaml --ai86 --fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-group0-pool4 --config-file tests/test-streaming-group0-pool4.yaml --ai86 --fifo

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml --ai86 --fifo --increase-start 1
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-multilayer-eltwise --config-file tests/test-multilayer-eltwise.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --input-fifo
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 2 --fast-fifo --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai86 --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress --stop-after 0

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-chan1024 --config-file tests/test-chan1024.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-chan1024chan1024 --config-file tests/test-chan1024-1024.yaml --ai86

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga80x60-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --stop-after 1 
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga128x96-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-sync
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga190x120-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name medium --timeout 2500 --input-csv-period 180 --input-sync --increase-start 4 --increase-delta2 4
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-qfastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-group-notvga-hwc --config-file tests/test-fifostream-group-notvga64x48-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-csv-fastfifostream-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-csv-period 180 --timeout 60
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai86 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --ignore-streaming
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai86 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 1 --ignore-streaming
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai86 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 2 --ignore-streaming

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-simple1b-widein-q1 --config-file tests/test-widein-q1.yaml --ai86 --simple1b
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-simple1b-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai86 --simple1b

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-deepsleep-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --deepsleep --autogen None
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-deepsleep-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo --deepsleep --autogen None

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-verify-layers --config-file tests/test-layers.yaml --ai86 --verify-writes --write-zero-registers
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-verify-cifar-bias --checkpoint-file trained/ai86-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --ai86 --verify-writes --compact-data --mexpress

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-exclusivesram-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --riscv-exclusive

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-555-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 555
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-565-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 565
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-555-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 555
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-565-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 565

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-noretrace-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai86 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-csv-retrace 0
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-powerdown-group0-pool4 --config-file tests/test-group0-pool4.yaml --ai86 --fifo --powerdown

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fire --config-file tests/test-ai86-fire-cifar10.yaml --ai86 --checkpoint-file trained/ai86-firetestnet-cifar10.pth.tar
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fire --config-file tests/test-ai86-fire-cifar10.yaml --ai86 --checkpoint-file trained/ai86-firetestnet-cifar10.pth.tar --stop-after 4
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fire2 --config-file tests/test-ai86-fire2-cifar10.yaml --ai86 --checkpoint-file trained/ai86-firetestnet-cifar10.pth.tar
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-fire2 --config-file tests/test-ai86-fire2-cifar10.yaml --ai86 --checkpoint-file trained/ai86-firetestnet-cifar10.pth.tar --stop-after 4

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-kmax_bmax_dmax --config-file tests/test-max.yaml --ai86
./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-riscv-fastfifo-simple --config-file tests/test-fifo-hwc.yaml --ai86 --fifo --riscv --riscv-flash --fast-fifo

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-rdysel --config-file tests/test-pooling13x1s1.yaml --ai86 --ready-sel 3 --ready-sel-fifo 3 --ready-sel-aon 3

./cnn-gen.py --verbose --autogen rtlsim-ai86 --top-level cnn -L --test-dir rtlsim-ai86 --prefix ai86-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml --ai86 --mexpress --mlator
