#!/bin/sh
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fmnist --checkpoint-file tests/test-fashionmnist.pth.tar --config-file tests/test-fashionmnist-chw.yaml --stop-after 0 --ai85 --verify-writes $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-q4-cifar-bias --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-q4-cifar-bias --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 1 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-q4-16x16avgpool --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-3x3s2p2avgpool --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-ai85-cifar10-hwc-3x3s2p2avgpool.yaml --stop-after 0 --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-3x3s1avgpool --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling3x3s1.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-4x4s2avgpool --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling4x4s2.yaml --stop-after 0 --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wideout --checkpoint-file tests/test-mnist-wide.pth.tar --config-file tests/test-ai85-mnistwide.yaml --stop-after 0 --ai85 --timeout 16 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-80wideout --checkpoint-file tests/test-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-80wideout-q4 --checkpoint-file tests/test-mnist-80wide-q4.pth.tar --config-file tests/test-ai85-mnist80wide-q4.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-80wideout --checkpoint-file tests/test-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 1 --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-80wideout-q4-32bit --checkpoint-file tests/test-mnist-80expansion-q4.pth.tar --config-file tests/test-ai85-80expansion-q4-32bitout.yaml --stop-after 1 --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d --config-file tests/test-ai85-conv1d.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-1 --config-file tests/test-conv1d-1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-2 --config-file tests/test-conv1d-2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-3 --config-file tests/test-conv1d-3.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-4 --config-file tests/test-conv1d-4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-5 --config-file tests/test-conv1d-5.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-6 --config-file tests/test-conv1d-6.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-7 --config-file tests/test-conv1d-7.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-8 --config-file tests/test-conv1d-8.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-9 --config-file tests/test-conv1d-9.yaml --ai85 $@

# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1dII --config-file tests/test-ai85-conv1dII.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-3II --config-file tests/test-conv1d-3II.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-4II --config-file tests/test-conv1d-4II.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-5II --config-file tests/test-conv1d-5II.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-6II --config-file tests/test-conv1d-6II.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-7II --config-file tests/test-conv1d-7II.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-1 --config-file tests/test-conv1d-pool-1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-2 --config-file tests/test-conv1d-pool-2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-3 --config-file tests/test-conv1d-pool-3.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-4 --config-file tests/test-conv1d-pool-4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-5 --config-file tests/test-conv1d-pool-5.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-6 --config-file tests/test-conv1d-pool-6.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-7 --config-file tests/test-conv1d-pool-7.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-8 --config-file tests/test-conv1d-pool-8.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-9 --config-file tests/test-conv1d-pool-9.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-1-q1 --config-file tests/test-conv1d-pool-1-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-2-q1 --config-file tests/test-conv1d-pool-1-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-3-q1 --config-file tests/test-conv1d-pool-3-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-4-q1 --config-file tests/test-conv1d-pool-4-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-5-q1 --config-file tests/test-conv1d-pool-5-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-6-q1 --config-file tests/test-conv1d-pool-6-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-7-q1 --config-file tests/test-conv1d-pool-7-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-8-q1 --config-file tests/test-conv1d-pool-8-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-9-q1 --config-file tests/test-conv1d-pool-9-q1.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-1-q2 --config-file tests/test-conv1d-pool-1-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-2-q2 --config-file tests/test-conv1d-pool-1-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-3-q2 --config-file tests/test-conv1d-pool-3-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-4-q2 --config-file tests/test-conv1d-pool-4-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-5-q2 --config-file tests/test-conv1d-pool-5-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-6-q2 --config-file tests/test-conv1d-pool-6-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-7-q2 --config-file tests/test-conv1d-pool-7-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-8-q2 --config-file tests/test-conv1d-pool-8-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-9-q2 --config-file tests/test-conv1d-pool-9-q2.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-1-q4 --config-file tests/test-conv1d-pool-1-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-2-q4 --config-file tests/test-conv1d-pool-1-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-3-q4 --config-file tests/test-conv1d-pool-3-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-4-q4 --config-file tests/test-conv1d-pool-4-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-5-q4 --config-file tests/test-conv1d-pool-5-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-6-q4 --config-file tests/test-conv1d-pool-6-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-7-q4 --config-file tests/test-conv1d-pool-7-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-8-q4 --config-file tests/test-conv1d-pool-8-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-9-q4 --config-file tests/test-conv1d-pool-9-q4.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-1-wide --config-file tests/test-conv1d-pool-1-wide.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-3-wide --config-file tests/test-conv1d-pool-3-wide.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-5-wide --config-file tests/test-conv1d-pool-5-wide.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-pool-9-wide --config-file tests/test-conv1d-pool-9-wide.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1x1 --config-file tests/test-conv1x1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-conv1x1 --checkpoint-file tests/test-cifar10-1x1.pth.tar --config-file tests/test-ai85-cifar10-hwc-1x1.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare --config-file tests/test-nonsquare.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare-pool --config-file tests/test-nonsquare-pool.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare-nonsquarepool --config-file tests/test-nonsquare-nonsquarepool.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist --checkpoint-file tests/test-mnist.pth.tar --config-file tests/test-mnist-chw.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-speechcom --checkpoint-file tests/test-speechcom-net7.pth.tar --config-file tests/test-speechcom-chw.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-hwc.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 0 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 1 --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 2 --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-energy --config-file tests/test-energy.yaml --ai85 --timeout 40 --mexpress --compact-data $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-16x16s4 --config-file tests/test-pooling16x16s4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-16x16s4wide --config-file tests/test-pooling16x16s4wide.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-16x16s7 --config-file tests/test-pooling16x16s7.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-16x16s16 --config-file tests/test-pooling16x16s16.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool2-16x16s7 --config-file tests/test-pooling16x16s7-II.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool2-16x16s16 --config-file tests/test-pooling16x16s16-II.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-1x13s13 --config-file tests/test-pooling1x13s13.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool2-1x13s13 --config-file tests/test-pooling1x13s13-II.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-13x1s1 --config-file tests/test-pooling13x1s1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool2-13x1s1 --config-file tests/test-pooling13x1s1-II.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-2x3s2 --config-file tests/test-pooling2x3s2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-3x2s3 --config-file tests/test-pooling3x2s3.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-3x3s2 --config-file tests/test-pooling3x3s2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-4x3s3 --config-file tests/test-pooling3x3s3.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-4x4s1 --config-file tests/test-pooling4x4s1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-4x4s3 --config-file tests/test-pooling4x4s3.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-4x4s4 --config-file tests/test-pooling4x4s4.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-singlebyte-chw --config-file tests/test-singlebyte-chw.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-layers --config-file tests/test-layers.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru --config-file tests/test-passthrough.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru-2 --config-file tests/test-passthrough-2.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru-2a --config-file tests/test-passthrough-2a.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru-pool --config-file tests/test-passthrough-pool.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml --ai85 $@
# ./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthru-2a-pool --config-file tests/test-passthrough-2a-pool.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein --config-file tests/test-widein.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-q1 --config-file tests/test-widein-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-q2 --config-file tests/test-widein-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-q4 --config-file tests/test-widein-q4.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wideout --config-file tests/test-wideout.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wideout-q1 --config-file tests/test-wideout-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wideout-q2 --config-file tests/test-wideout-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wideout-q4 --config-file tests/test-wideout-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512out --config-file tests/test-wide512out.yaml --ai85 --compact-weights --autogen None $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-bias --config-file tests/test-widein-bias.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-bias-q1 --config-file tests/test-widein-bias-q1.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-bias-q2 --config-file tests/test-widein-bias-q2.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-bias-q4 --config-file tests/test-widein-bias-q4.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide256in-bias-q1 --config-file tests/test-wide256in-bias-q1.yaml --ai85 --timeout 128 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512in --config-file tests/test-wide512in.yaml --ai85 --compact-weights --timeout 128 --autogen None $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512in-bias-q2 --config-file tests/test-wide512in-bias-q2.yaml --ai85 --timeout 128 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512in-q1 --config-file tests/test-wide512in-q1.yaml --ai85 --timeout 128 --autogen None $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512in-q2 --config-file tests/test-wide512in-q2.yaml --ai85 --compact-weights --timeout 128 --autogen None $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide512in-q4 --config-file tests/test-wide512in-q4.yaml --ai85 --compact-weights --timeout 128 --autogen None $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-dataonexone --config-file tests/test-dataonexone.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-dataonexone2 --config-file tests/test-dataonexone2.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-maxproc4 --config-file tests/test-widein-maxproc4.yaml --ai85 --max-proc 4 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-maxproc4-q4 --config-file tests/test-widein-maxproc4-q4.yaml --ai85 --max-proc 4 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv2Dk1x1 --config-file tests/test-conv2Dk1x1.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv2Dk1x1-b --config-file tests/test-conv2Dk1x1-b.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv2Dk1x1-b-pool --config-file tests/test-conv2Dk1x1-b-pool.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlp12to2 --config-file tests/test-mlp12to2.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml --ai85 --debug-computation --debug $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-wide3to508to3 --config-file tests/test-wide3to508to3.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai85 --stop-after 1 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai85 --stop-after 2 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml --ai85 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai85 --stop-after 1 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai85 --stop-after 2 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml --ai85 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-transition --config-file tests/test-stream-transition.yaml --ai85 --overwrite-ok --allow-streaming $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add --config-file tests/test-eltwiseadd.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-sub --config-file tests/test-eltwisesub.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-xor --config-file tests/test-eltwisexor.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-or --config-file tests/test-eltwiseor.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add-multipass --config-file tests/test-eltwiseadd-multipass.yaml --ai85 --max-proc 2 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add7 --config-file tests/test-eltwiseadd-7ch.yaml --ai85 --legacy-test $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add4-7ch --config-file tests/test-eltwiseadd4-7ch.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add7-conv2d --config-file tests/test-eltwiseaddconv2d-7ch.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add31 --config-file tests/test-eltwiseadd-31ch.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add126 --config-file tests/test-eltwiseadd-126ch.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-poolafter --config-file tests/test-eltwiseadd-poolafter.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add5op31 --config-file tests/test-eltwiseadd-5op-31ch.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-add5op31-32bit --config-file tests/test-eltwiseadd-5op-31ch-32bit.yaml --ai85 --legacy-test $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-passthroughmp --config-file tests/test-passthroughmultipass.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-widein-1x1 --config-file tests/test-widein-1x1.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-pool-avg --config-file tests/test-eltwiseadd-pool-avg.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-q4-16x16avgpool-round --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 --ai85 --avg-pool-rounding $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai85 --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai85 --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml --ai85 --fifo --stop-after 2 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 2 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-mlp --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw-mlp.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar-transition --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --ai85 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-stream-cifar-transition-zeroize --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --ai85 --zero-sram --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml --ai85 --fifo --stop-after 5 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 --ai85 --mlator --mexpress --mlator-noverify --compact-data $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 1 --ai85 --mlator --mexpress --mlator-noverify --compact-data $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 2 --ai85 --mlator $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-abs --config-file tests/test-conv1d-abs.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-abs --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-abs.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten12to17 --config-file tests/test-mlpflatten12to17.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten12to17big --config-file tests/test-mlpflatten12to17-big.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten12to100 --config-file tests/test-mlpflatten12to100.yaml --ai85 --debug-computation --debug $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten12to100big --config-file tests/test-mlpflatten12to100-big.yaml --ai85 --debug-computation --debug $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten192to10 --config-file tests/test-mlpflatten192to10.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten192to10big --config-file tests/test-mlpflatten192to10-big.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten768to10 --config-file tests/test-mlpflatten768to10.yaml --ai85 --debug --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten768to10big --config-file tests/test-mlpflatten768to10-big.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten768to100 --config-file tests/test-mlpflatten768to100.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten768to100big --config-file tests/test-mlpflatten768to100-big.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall-oneshot --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 --ai85 --one-shot $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall-stopstart --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 --ai85 --stop-start $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall-cweight --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 --ai85 --compact-weights --verify-kernels $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mnist-extrasmall-mexpress --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 --ai85 --mexpress --verify-kernels $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-cifar-bias-mexpress --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 --ai85 --mexpress --verify-kernels $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-q4-cifar-bias-mexpress --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 --ai85 --mexpress --verify-kernels $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlpflatten768to100big-q4 --config-file tests/test-mlpflatten768to100-big-q4.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml --ai85 --fifo --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml --ai85 --fifo --debug-computation --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-nonsquare-hwc --config-file tests/test-fifostream-nonsquare-hwc.yaml --ai85 --fifo --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-pool --config-file tests/test-fifostream-pool.yaml --ai85 --fifo --debug-computation --override-start 0x08 --override-rollover 0x24 --override-delta2 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-pool-hwc --config-file tests/test-fifostream-pool-hwc.yaml --ai85 --fifo --debug-computation --override-start 0x1b --override-rollover 0x1c --override-delta2 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-15ch-hwc --config-file tests/test-fifostream-15ch-hwc.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-16ch-hwc --config-file tests/test-fifostream-16ch-hwc.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640 --config-file tests/test-fifostream-640.yaml --ai85 --fifo --mexpress --timeout 40 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640 --config-file tests/test-fifostream-640.yaml --ai85 --fifo --stop-after 1 --mexpress --timeout 40 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai85 --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml --ai85 --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare --config-file tests/test-nonsquare.yaml --ai85 --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare --config-file tests/test-nonsquare.yaml --ai85 --debug-computation --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-pool-stride --config-file tests/test-fifostream-pool-stride.yaml --ai85 --fifo --debug-computation --override-start 0x07 --override-rollover 0x38 --override-delta2 0x04 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-pool-stride-hwc --config-file tests/test-fifostream-pool-stride-hwc.yaml --ai85 --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 2 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 3 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 4 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml --ai85 --stop-after 5 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 2 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 3 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 4 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 5 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-pool-stride-hwc-slow --config-file tests/test-fifostream-pool-stride-hwc.yaml --ai85 --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b --slow-load 8 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-outputshift --config-file tests/test-outputshift.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai85 --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32 --config-file tests/test-fifostream-32.yaml --ai85 --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml --ai85 --fifo --mexpress --timeout 40 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml --ai85 --fifo --stop-after 1 --mexpress --timeout 40 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-pool-4high --config-file tests/test-pool-4high.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upsample --config-file tests/test-upsample.yaml --ai85 --debug --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale --config-file tests/test-upscale.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale --config-file tests/test-upscale.yaml --ai85 --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale --config-file tests/test-upscale.yaml --ai85 --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale-pro --config-file tests/test-upscale-pro.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale-pro --config-file tests/test-upscale-pro.yaml --ai85 --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-upscale-pro --config-file tests/test-upscale-pro.yaml --ai85 --stop-after 1 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-helloworld --config-file tests/test-pooling13x1s1.yaml --ai85 --riscv-flash $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml --ai85 --compact-data --mexpress --riscv-flash --riscv-cache $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-flash-cache-helloworld --config-file tests/test-pooling13x1s1.yaml --ai85 --riscv-cache $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-flash-cache-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml --ai85 --riscv-cache --mexpress --compact-data $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifo-nonsquare --config-file tests/test-fifo-nonsquare.yaml --ai85 --fifo --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai85 --fifo --debug-computation $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai85 --fifo --input-csv input.csv $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-hwc-nonsquare --config-file tests/test-fifostream-nonsquare-hwc.yaml --ai85 --fifo --input-csv input.csv --riscv-cache $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml --ai85 --fifo --stop-after 5 --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-vga-hwc --config-file tests/test-fifostream-vga-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1  $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-vga2-hwc --config-file tests/test-fifostream-vga2-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-group0-pool4 --config-file tests/test-group0-pool4.yaml --ai85 --fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-group0-pool4 --config-file tests/test-streaming-group0-pool4.yaml --ai85 --fifo $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml --ai85 --fifo --increase-start 1 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-multilayer-eltwise --config-file tests/test-multilayer-eltwise.yaml --ai85 --legacy-test $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --input-fifo $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 2 --fast-fifo --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress --stop-after 0 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-chan1024 --config-file tests/test-chan1024.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-chan1024chan1024 --config-file tests/test-chan1024-1024.yaml --ai85 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga80x60-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --stop-after 1  $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga128x96-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-sync $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga190x120-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name medium --timeout 2500 --input-csv-period 180 --input-sync --increase-start 4 --increase-delta2 4 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-qfastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-group-notvga-hwc --config-file tests/test-fifostream-group-notvga64x48-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-csv-fastfifostream-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-csv-period 180 --timeout 60 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai85 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --ignore-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai85 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 1 --ignore-streaming $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml --ai85 --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 2 --ignore-streaming $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-simple1b-widein-q1 --config-file tests/test-widein-q1.yaml --ai85 --simple1b $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-simple1b-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --simple1b $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-deepsleep-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --deepsleep --autogen None $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-deepsleep-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo --deepsleep --autogen None $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-verify-layers --config-file tests/test-layers.yaml --ai85 --verify-writes --write-zero-registers $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-verify-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --ai85 --verify-writes --compact-data --mexpress $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-exclusivesram-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --riscv-exclusive $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-555-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 555 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-565-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 565 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-555-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 555 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-565-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 565 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-noretrace-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml --ai85 --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-csv-retrace 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-powerdown-group0-pool4 --config-file tests/test-group0-pool4.yaml --ai85 --fifo --powerdown $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fire --config-file tests/test-ai85-fire-cifar10.yaml --ai85 --checkpoint-file tests/test-firetestnet-cifar10.pth.tar $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fire --config-file tests/test-ai85-fire-cifar10.yaml --ai85 --checkpoint-file tests/test-firetestnet-cifar10.pth.tar --stop-after 4 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fire2 --config-file tests/test-ai85-fire2-cifar10.yaml --ai85 --checkpoint-file tests/test-firetestnet-cifar10.pth.tar $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-fire2 --config-file tests/test-ai85-fire2-cifar10.yaml --ai85 --checkpoint-file tests/test-firetestnet-cifar10.pth.tar --stop-after 4 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-kmax_bmax_dmax --config-file tests/test-max.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-fastfifo-simple --config-file tests/test-fifo-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-rdysel --config-file tests/test-pooling13x1s1.yaml --ai85 --ready-sel 3 --ready-sel-fifo 3 --ready-sel-aon 3 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml --ai85 --mexpress --mlator $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml --ai85 --stop-after 0 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml --ai85 --stop-after 1 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml --ai85 --mexpress --stop-after 5 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml --ai85 --mexpress $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-resnet --checkpoint-file tests/test-resnet.pth.tar --config-file tests/test-resnet-4l.yaml --device 85 --compact-data --mexpress $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-resnet --checkpoint-file tests/test-resnet.pth.tar --config-file tests/test-resnet.yaml --device 85 --compact-data --mexpress $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-burnin_64x64x64 --config-file tests/test-burnin_64x64x64.yaml --device 85 --compact-data --mexpress --fixed-input --max-checklines 4096 --timeout 60 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-burnin_16x64x64 --config-file tests/test-burnin_16x64x64.yaml --device 85 --compact-data --mexpress --fixed-input --max-checklines 4096 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-burnin-rand_64x64x64 --config-file tests/test-burnin-rand_64x64x64.yaml --device 85 --compact-data --mexpress --max-checklines 4096 --timeout 60 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-burnin-rand_16x64x64 --config-file tests/test-burnin-rand_16x64x64.yaml --device 85 --compact-data --mexpress --max-checklines 4096 $@

./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-conv1d-3-bias --config-file tests/test-conv1d-3-bias.yaml --ai85 $@
./ai8xize.py --verbose --autogen rtlsim-ai85 --top-level cnn -L --test-dir rtlsim-ai85 --prefix ai85-riscv-fastfifo-mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --ai85 --compact-data --mexpress --riscv --riscv-flash --riscv-cache --riscv-debug $@
