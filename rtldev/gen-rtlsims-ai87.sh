#!/bin/sh
DEVICE="--device 78002"
TARGET="rtlsim-ai87"
PREFIX="ai87"
SHORT_LOG="--log-last-only"

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fmnist --checkpoint-file tests/test-fashionmnist.pth.tar --config-file tests/test-fashionmnist-chw.yaml --stop-after 0 $DEVICE --verify-writes $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-q4-cifar-bias --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-q4-cifar-bias --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 1 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-q4-16x16avgpool --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-3x3s2p2avgpool --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-ai85-cifar10-hwc-3x3s2p2avgpool.yaml --stop-after 0 $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-3x3s1avgpool --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling3x3s1.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-4x4s2avgpool --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-pooling4x4s2.yaml --stop-after 0 $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wideout --checkpoint-file tests/test-mnist-wide.pth.tar --config-file tests/test-ai85-mnistwide.yaml --stop-after 0 $DEVICE --timeout 16 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-80wideout --checkpoint-file tests/test-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-80wideout-q4 --checkpoint-file tests/test-mnist-80wide-q4.pth.tar --config-file tests/test-ai85-mnist80wide-q4.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-80wideout --checkpoint-file tests/test-mnist-80wide.pth.tar --config-file tests/test-ai85-mnist80wide.yaml --stop-after 1 $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-80wideout-q4-32bit --checkpoint-file tests/test-mnist-80expansion-q4.pth.tar --config-file tests/test-ai85-80expansion-q4-32bitout.yaml --stop-after 1 $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d --config-file tests/test-ai85-conv1d.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-1 --config-file tests/test-conv1d-1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-2 --config-file tests/test-conv1d-2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-3 --config-file tests/test-conv1d-3.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-4 --config-file tests/test-conv1d-4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-5 --config-file tests/test-conv1d-5.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-6 --config-file tests/test-conv1d-6.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-7 --config-file tests/test-conv1d-7.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-8 --config-file tests/test-conv1d-8.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-9 --config-file tests/test-conv1d-9.yaml $DEVICE $@

# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1dII --config-file tests/test-ai85-conv1dII.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-3II --config-file tests/test-conv1d-3II.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-4II --config-file tests/test-conv1d-4II.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-5II --config-file tests/test-conv1d-5II.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-6II --config-file tests/test-conv1d-6II.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-7II --config-file tests/test-conv1d-7II.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-1 --config-file tests/test-conv1d-pool-1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-2 --config-file tests/test-conv1d-pool-2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-3 --config-file tests/test-conv1d-pool-3.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-4 --config-file tests/test-conv1d-pool-4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-5 --config-file tests/test-conv1d-pool-5.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-6 --config-file tests/test-conv1d-pool-6.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-7 --config-file tests/test-conv1d-pool-7.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-8 --config-file tests/test-conv1d-pool-8.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-9 --config-file tests/test-conv1d-pool-9.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-1-q1 --config-file tests/test-conv1d-pool-1-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-2-q1 --config-file tests/test-conv1d-pool-1-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-3-q1 --config-file tests/test-conv1d-pool-3-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-4-q1 --config-file tests/test-conv1d-pool-4-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-5-q1 --config-file tests/test-conv1d-pool-5-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-6-q1 --config-file tests/test-conv1d-pool-6-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-7-q1 --config-file tests/test-conv1d-pool-7-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-8-q1 --config-file tests/test-conv1d-pool-8-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-9-q1 --config-file tests/test-conv1d-pool-9-q1.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-1-q2 --config-file tests/test-conv1d-pool-1-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-2-q2 --config-file tests/test-conv1d-pool-1-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-3-q2 --config-file tests/test-conv1d-pool-3-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-4-q2 --config-file tests/test-conv1d-pool-4-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-5-q2 --config-file tests/test-conv1d-pool-5-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-6-q2 --config-file tests/test-conv1d-pool-6-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-7-q2 --config-file tests/test-conv1d-pool-7-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-8-q2 --config-file tests/test-conv1d-pool-8-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-9-q2 --config-file tests/test-conv1d-pool-9-q2.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-1-q4 --config-file tests/test-conv1d-pool-1-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-2-q4 --config-file tests/test-conv1d-pool-1-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-3-q4 --config-file tests/test-conv1d-pool-3-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-4-q4 --config-file tests/test-conv1d-pool-4-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-5-q4 --config-file tests/test-conv1d-pool-5-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-6-q4 --config-file tests/test-conv1d-pool-6-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-7-q4 --config-file tests/test-conv1d-pool-7-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-8-q4 --config-file tests/test-conv1d-pool-8-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-9-q4 --config-file tests/test-conv1d-pool-9-q4.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-1-wide --config-file tests/test-conv1d-pool-1-wide.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-3-wide --config-file tests/test-conv1d-pool-3-wide.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-5-wide --config-file tests/test-conv1d-pool-5-wide.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-pool-9-wide --config-file tests/test-conv1d-pool-9-wide.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1x1 --config-file tests/test-conv1x1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-conv1x1 --checkpoint-file tests/test-cifar10-1x1.pth.tar --config-file tests/test-ai85-cifar10-hwc-1x1.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare --config-file tests/test-nonsquare.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare-pool --config-file tests/test-nonsquare-pool.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare-nonsquarepool --config-file tests/test-nonsquare-nonsquarepool.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist --checkpoint-file tests/test-mnist.pth.tar --config-file tests/test-mnist-chw.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-speechcom --checkpoint-file tests/test-speechcom-net7.pth.tar --config-file tests/test-speechcom-chw.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-hwc.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 0 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 1 $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw.yaml --stop-after 2 $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-energy --config-file tests/test-energy.yaml $DEVICE --timeout 40 --mexpress --compact-data $SHORT_LOG $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-16x16s4 --config-file tests/test-pooling16x16s4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-16x16s4wide --config-file tests/test-pooling16x16s4wide.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-16x16s7 --config-file tests/test-pooling16x16s7.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-16x16s16 --config-file tests/test-pooling16x16s16.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool2-16x16s7 --config-file tests/test-pooling16x16s7-II.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool2-16x16s16 --config-file tests/test-pooling16x16s16-II.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-1x13s13 --config-file tests/test-pooling1x13s13.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool2-1x13s13 --config-file tests/test-pooling1x13s13-II.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-13x1s1 --config-file tests/test-pooling13x1s1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool2-13x1s1 --config-file tests/test-pooling13x1s1-II.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-2x3s2 --config-file tests/test-pooling2x3s2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-3x2s3 --config-file tests/test-pooling3x2s3.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-3x3s2 --config-file tests/test-pooling3x3s2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-4x3s3 --config-file tests/test-pooling3x3s3.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-4x4s1 --config-file tests/test-pooling4x4s1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-4x4s3 --config-file tests/test-pooling4x4s3.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-4x4s4 --config-file tests/test-pooling4x4s4.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-singlebyte-chw --config-file tests/test-singlebyte-chw.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-layers --config-file tests/test-layers.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru --config-file tests/test-passthrough.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru-2 --config-file tests/test-passthrough-2.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru-2a --config-file tests/test-passthrough-2a.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru-pool --config-file tests/test-passthrough-pool.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml $DEVICE $@
# ./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthru-2a-pool --config-file tests/test-passthrough-2a-pool.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein --config-file tests/test-widein.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-q1 --config-file tests/test-widein-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-q2 --config-file tests/test-widein-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-q4 --config-file tests/test-widein-q4.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wideout --config-file tests/test-wideout.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wideout-q1 --config-file tests/test-wideout-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wideout-q2 --config-file tests/test-wideout-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wideout-q4 --config-file tests/test-wideout-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512out --config-file tests/test-wide512out.yaml $DEVICE --compact-weights --autogen None $SHORT_LOG $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-bias --config-file tests/test-widein-bias.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-bias-q1 --config-file tests/test-widein-bias-q1.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-bias-q2 --config-file tests/test-widein-bias-q2.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-bias-q4 --config-file tests/test-widein-bias-q4.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide256in-bias-q1 --config-file tests/test-wide256in-bias-q1.yaml $DEVICE --timeout 128 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512in --config-file tests/test-wide512in.yaml $DEVICE --compact-weights --timeout 128 --autogen None $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512in-bias-q2 --config-file tests/test-wide512in-bias-q2.yaml $DEVICE --timeout 128 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512in-q1 --config-file tests/test-wide512in-q1.yaml $DEVICE --timeout 128 --autogen None $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512in-q2 --config-file tests/test-wide512in-q2.yaml $DEVICE --compact-weights --timeout 128 --autogen None $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide512in-q4 --config-file tests/test-wide512in-q4.yaml $DEVICE --compact-weights --timeout 128 --autogen None $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-dataonexone --config-file tests/test-dataonexone.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-dataonexone2 --config-file tests/test-dataonexone2.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-maxproc4 --config-file tests/test-widein-maxproc4.yaml $DEVICE --max-proc 4 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-maxproc4-q4 --config-file tests/test-widein-maxproc4-q4.yaml $DEVICE --max-proc 4 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv2Dk1x1 --config-file tests/test-conv2Dk1x1.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv2Dk1x1-b --config-file tests/test-conv2Dk1x1-b.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv2Dk1x1-b-pool --config-file tests/test-conv2Dk1x1-b-pool.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlp12to2 --config-file tests/test-mlp12to2.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml $DEVICE --debug-computation --debug $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-wide3to508to3 --config-file tests/test-wide3to508to3.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml $DEVICE --stop-after 1 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml $DEVICE --stop-after 2 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar.yaml $DEVICE --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml $DEVICE --stop-after 1 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml $DEVICE --stop-after 2 --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-hwc.yaml $DEVICE --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-transition --config-file tests/test-stream-transition.yaml $DEVICE --overwrite-ok --allow-streaming $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add --config-file tests/test-eltwiseadd.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-sub --config-file tests/test-eltwisesub.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-xor --config-file tests/test-eltwisexor.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-or --config-file tests/test-eltwiseor.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add-multipass --config-file tests/test-eltwiseadd-multipass.yaml $DEVICE --max-proc 2 --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add7 --config-file tests/test-eltwiseadd-7ch.yaml $DEVICE --legacy-test $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add4-7ch --config-file tests/test-eltwiseadd4-7ch.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add7-conv2d --config-file tests/test-eltwiseaddconv2d-7ch.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add31 --config-file tests/test-eltwiseadd-31ch.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add126 --config-file tests/test-eltwiseadd-126ch.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-poolafter --config-file tests/test-eltwiseadd-poolafter.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add5op31 --config-file tests/test-eltwiseadd-5op-31ch.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-add5op31-32bit --config-file tests/test-eltwiseadd-5op-31ch-32bit.yaml $DEVICE --legacy-test $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-passthroughmp --config-file tests/test-passthroughmultipass.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-widein-1x1 --config-file tests/test-widein-1x1.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-pool-avg --config-file tests/test-eltwiseadd-pool-avg.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-q4-16x16avgpool-round --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-16x16avgpool.yaml --stop-after 0 $DEVICE --avg-pool-rounding $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml $DEVICE --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml $DEVICE --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2 --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar.yaml $DEVICE --fifo --stop-after 2 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 2 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-mlp --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-chw-mlp.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar-transition --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml $DEVICE --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-stream-cifar-transition-zeroize --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml $DEVICE --zero-sram --overwrite-ok --allow-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml $DEVICE --fifo --stop-after 5 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 $DEVICE --mlator --mexpress --mlator-noverify --compact-data $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 1 $DEVICE --mlator --mexpress --mlator-noverify --compact-data $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlator --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 2 $DEVICE --mlator $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-abs --config-file tests/test-conv1d-abs.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-abs --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-cifar10-abs.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to17 --config-file tests/test-mlpflatten12to17.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to17big --config-file tests/test-mlpflatten12to17-big.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to100 --config-file tests/test-mlpflatten12to100.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten12to100big --config-file tests/test-mlpflatten12to100-big.yaml $DEVICE --debug-computation --debug $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten192to10 --config-file tests/test-mlpflatten192to10.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten192to10big --config-file tests/test-mlpflatten192to10-big.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten768to10 --config-file tests/test-mlpflatten768to10.yaml $DEVICE --debug --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten768to10big --config-file tests/test-mlpflatten768to10-big.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten768to100 --config-file tests/test-mlpflatten768to100.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten768to100big --config-file tests/test-mlpflatten768to100-big.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall-oneshot --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 $DEVICE --one-shot $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall-stopstart --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 $DEVICE --stop-start $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall-cweight --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 $DEVICE --compact-weights --verify-kernels $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mnist-extrasmall-mexpress --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 0 $DEVICE --mexpress --verify-kernels $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-cifar-bias-mexpress --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml --stop-after 0 $DEVICE --mexpress --verify-kernels $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-q4-cifar-bias-mexpress --checkpoint-file tests/test-cifar10-bias-quant4.pth.tar --config-file tests/test-ai85-cifar10-hwc-quant4.yaml --stop-after 0 $DEVICE --mexpress --verify-kernels $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflatten768to100big-q4 --config-file tests/test-mlpflatten768to100-big-q4.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml $DEVICE --fifo --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-nonsquare --config-file tests/test-fifostream-nonsquare.yaml $DEVICE --fifo --debug-computation --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-nonsquare-hwc --config-file tests/test-fifostream-nonsquare-hwc.yaml $DEVICE --fifo --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-pool --config-file tests/test-fifostream-pool.yaml $DEVICE --fifo --debug-computation --override-start 0x08 --override-rollover 0x24 --override-delta2 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-pool-hwc --config-file tests/test-fifostream-pool-hwc.yaml $DEVICE --fifo --debug-computation --override-start 0x1b --override-rollover 0x1c --override-delta2 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-15ch-hwc --config-file tests/test-fifostream-15ch-hwc.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-16ch-hwc --config-file tests/test-fifostream-16ch-hwc.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640 --config-file tests/test-fifostream-640.yaml $DEVICE --fifo --mexpress --timeout 40 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640 --config-file tests/test-fifostream-640.yaml $DEVICE --fifo --stop-after 1 --mexpress --timeout 40 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml $DEVICE --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640-small --config-file tests/test-fifostream-640-small.yaml $DEVICE --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare --config-file tests/test-nonsquare.yaml $DEVICE --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare --config-file tests/test-nonsquare.yaml $DEVICE --debug-computation --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-pool-stride --config-file tests/test-fifostream-pool-stride.yaml $DEVICE --fifo --debug-computation --override-start 0x07 --override-rollover 0x38 --override-delta2 0x04 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-pool-stride-hwc --config-file tests/test-fifostream-pool-stride-hwc.yaml $DEVICE --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 2 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 3 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 4 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer --config-file tests/test-conv1d-multilayer.yaml $DEVICE --stop-after 5 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 2 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 3 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 4 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --stop-after 5 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-pool-stride-hwc-slow --config-file tests/test-fifostream-pool-stride-hwc.yaml $DEVICE --fifo --debug-computation --override-start 0x1a --override-rollover 0x1b --slow-load 8 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-outputshift --config-file tests/test-outputshift.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32 --config-file tests/test-fifostream-32.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32 --config-file tests/test-fifostream-32.yaml $DEVICE --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32 --config-file tests/test-fifostream-32.yaml $DEVICE --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml $DEVICE --fifo --mexpress --timeout 40 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-640-hwc --config-file tests/test-fifostream-640-hwc.yaml $DEVICE --fifo --stop-after 1 --mexpress --timeout 40 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-pool-4high --config-file tests/test-pool-4high.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upsample --config-file tests/test-upsample.yaml $DEVICE --debug --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale --config-file tests/test-upscale.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale --config-file tests/test-upscale.yaml $DEVICE --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale --config-file tests/test-upscale.yaml $DEVICE --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale-pro --config-file tests/test-upscale-pro.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale-pro --config-file tests/test-upscale-pro.yaml $DEVICE --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upscale-pro --config-file tests/test-upscale-pro.yaml $DEVICE --stop-after 1 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-helloworld --config-file tests/test-pooling13x1s1.yaml $DEVICE --riscv-flash $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml $DEVICE --compact-data --mexpress --riscv-flash --riscv-cache $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-flash-cache-helloworld --config-file tests/test-pooling13x1s1.yaml $DEVICE --riscv-cache $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-flash-cache-mexpress-helloworld --config-file tests/test-pooling13x1s1.yaml $DEVICE --riscv-cache --mexpress --compact-data $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifo-nonsquare --config-file tests/test-fifo-nonsquare.yaml $DEVICE --fifo --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml $DEVICE --fifo --debug-computation $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml $DEVICE --fifo --input-csv input.csv $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-hwc-nonsquare --config-file tests/test-fifostream-nonsquare-hwc.yaml $DEVICE --fifo --input-csv input.csv --riscv-cache $SHORTLOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-transition-early --config-file tests/test-fifostream-transition-early.yaml $DEVICE --fifo --stop-after 5 --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-vga-hwc --config-file tests/test-fifostream-vga-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1  $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-vga2-hwc --config-file tests/test-fifostream-vga2-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name long --timeout 2500 --input-csv-period 180 --input-sync --increase-start 8 --increase-delta2 1 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-group0-pool4 --config-file tests/test-group0-pool4.yaml $DEVICE --fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-group0-pool4 --config-file tests/test-streaming-group0-pool4.yaml $DEVICE --fifo $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml $DEVICE --fifo --increase-start 1 --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-multilayer-eltwise --config-file tests/test-multilayer-eltwise.yaml $DEVICE --legacy-test $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --input-fifo $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 2 --fast-fifo --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-cifar2-hwc --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml $DEVICE --fifo --stop-after 2 --fast-fifo-quad --riscv --riscv-cache --input-csv input.csv --input-csv-period 180 --timeout 40 --input-fifo --mexpress --stop-after 0 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-chan1024 --config-file tests/test-chan1024.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-chan1024chan1024 --config-file tests/test-chan1024-1024.yaml $DEVICE $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga80x60-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --stop-after 1  $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga128x96-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-sync $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga190x120-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --queue-name medium --timeout 2500 --input-csv-period 180 --input-sync --increase-start 4 --increase-delta2 4 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo-quad --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-group-notvga-hwc --config-file tests/test-fifostream-group-notvga64x48-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 $SHORT_LOG $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-csv-fastfifostream-expandcontract --config-file tests/test-fifostream-expandcontract.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-csv-period 180 --timeout 60 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml $DEVICE --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --ignore-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml $DEVICE --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 1 --ignore-streaming $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-expandcontract --config-file tests/test-fifostream-expandcontract.yaml $DEVICE --mexpress --riscv --riscv-flash --riscv-cache --timeout 60 --stop-after 2 --ignore-streaming $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-simple1b-widein-q1 --config-file tests/test-widein-q1.yaml $DEVICE --simple1b $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-simple1b-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml $DEVICE --simple1b $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-deepsleep-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga64x48-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --deepsleep --autogen None $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-deepsleep-riscv-fastfifo-hwc-nonsquare --config-file tests/test-fifo-hwc-nonsquare.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo --deepsleep --autogen None $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-verify-layers --config-file tests/test-layers.yaml $DEVICE --verify-writes --write-zero-registers $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-verify-cifar-bias --checkpoint-file tests/test-cifar10-bias.pth.tar --config-file tests/test-cifar10-hwc.yaml $DEVICE --verify-writes --compact-data --mexpress $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-exclusivesram-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --riscv-exclusive $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-555-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 555 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-565-riscv-csv-fastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --input-fifo --input-csv-period 160 --input-csv-format 565 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-555-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 555 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-565-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 360 --input-csv-format 565 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-noretrace-riscv-csv-fastfifostream-notvga-hwc --config-file tests/test-fifostream-notvga40x30-hwc.yaml $DEVICE --fifo --mexpress --riscv --riscv-flash --fast-fifo --input-csv input.csv --riscv-cache --timeout 2500 --input-csv-period 180 --input-csv-retrace 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-powerdown-group0-pool4 --config-file tests/test-group0-pool4.yaml $DEVICE --fifo --powerdown $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fire --config-file tests/test-ai85-fire-cifar10.yaml $DEVICE --checkpoint-file tests/test-firetestnet-cifar10.pth.tar $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fire --config-file tests/test-ai85-fire-cifar10.yaml $DEVICE --checkpoint-file tests/test-firetestnet-cifar10.pth.tar --stop-after 4 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fire2 --config-file tests/test-ai85-fire2-cifar10.yaml $DEVICE --checkpoint-file tests/test-firetestnet-cifar10.pth.tar $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-fire2 --config-file tests/test-ai85-fire2-cifar10.yaml $DEVICE --checkpoint-file tests/test-firetestnet-cifar10.pth.tar --stop-after 4 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-kmax_bmax_dmax --config-file tests/test-max.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-fastfifo-simple --config-file tests/test-fifo-hwc.yaml $DEVICE --fifo --riscv --riscv-flash --fast-fifo $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-rdysel --config-file tests/test-pooling13x1s1.yaml $DEVICE --ready-sel 3 --ready-sel-fifo 3 --ready-sel-aon 3 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml $DEVICE --mexpress --mlator $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml $DEVICE --stop-after 0 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml $DEVICE --stop-after 1 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml $DEVICE --mexpress --stop-after 5 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlp-multilayer --config-file tests/test-mlp-multilayer208.yaml $DEVICE --mexpress $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-resnet --checkpoint-file tests/test-resnet.pth.tar --config-file tests/test-resnet-4l.yaml $DEVICE --compact-data --mexpress $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-resnet --checkpoint-file tests/test-resnet.pth.tar --config-file tests/test-resnet.yaml $DEVICE --compact-data --mexpress $SHORT_LOG $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-burnin_64x64x64 --config-file tests/test-burnin_64x64x64.yaml $DEVICE --compact-data --mexpress --fixed-input --max-checklines 4096 --timeout 60 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-burnin_16x64x64 --config-file tests/test-burnin_16x64x64.yaml $DEVICE --compact-data --mexpress --fixed-input --max-checklines 4096 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-burnin-rand_64x64x64 --config-file tests/test-burnin-rand_64x64x64.yaml $DEVICE --compact-data --mexpress --max-checklines 4096 --timeout 60 $SHORT_LOG $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-burnin-rand_16x64x64 --config-file tests/test-burnin-rand_16x64x64.yaml $DEVICE --compact-data --mexpress --max-checklines 4096 $@

./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-conv1d-3-bias --config-file tests/test-conv1d-3-bias.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflattenpool12to2 --config-file tests/test-mlpflattenpool12to2.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflattenpoolavg12to2 --config-file tests/test-mlpflattenpoolavg12to2.yaml $DEVICE --debug-computation --debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-riscv-fastfifo-mnist --checkpoint-file trained/ai85-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml $DEVICE --compact-data --mexpress --riscv --riscv-flash --riscv-cache --riscv-debug $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-mlpflattenpool --config-file tests/test-mlpflattenpool.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-upsample-nonsquare --config-file tests/test-upsample-nonsquare.yaml $DEVICE --debug --debug-computation $@

# MAX78002 only
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-layers128 --config-file tests/test-layers128.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-startlayer28 --config-file tests/test-layers.yaml $DEVICE --start-layer 28 $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-maxkern --config-file tests/test-maxkern.yaml $DEVICE --reshape-inputs $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-maxkern0123 --config-file tests/test-maxkern0123.yaml $DEVICE --reshape-inputs $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-largeinput --config-file tests/test-largeinput.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-largeinputw --config-file tests/test-largeinputw.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-largeinputh --config-file tests/test-largeinputh.yaml $DEVICE $@
./ai8xize.py --verbose --autogen $TARGET --top-level cnn -L --test-dir $TARGET --prefix $PREFIX-linklayer --checkpoint-file tests/test-mnist-extrasmallnet.pth.tar --config-file tests/test-mnist-chw-extrasmallnet.yaml --stop-after 2 --link-layer $DEVICE $@
