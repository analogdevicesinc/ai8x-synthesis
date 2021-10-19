#!/bin/sh
TARGET=demos
COMMON_ARGS="--device CMSIS-NN"

python ai8xize.py --verbose --log --test-dir $TARGET --prefix mnist --checkpoint-file trained/ai85-mnist-qat8-q.pth.tar --config-file networks/mnist-chw-ai85.yaml $COMMON_ARGS "$@"
python ai8xize.py --verbose --log --test-dir $TARGET --prefix cifar-10 --checkpoint-file tests/ai85-cifar10-qat8-q.pth.tar --config-file networks/cifar10-hwc-ai85.yaml $COMMON_ARGS "$@"
python ai8xize.py --verbose --log --test-dir $TARGET --prefix kws20_v2 --checkpoint-file trained/ai85-kws20_v2-qat8-q.pth.tar --config-file networks/kws20-v2-hwc.yaml $COMMON_ARGS "$@"
python ai8xize.py --verbose --log --test-dir $TARGET --prefix faceid --checkpoint-file trained/ai85-faceid-qat8-q.pth.tar --config-file networks/faceid.yaml $COMMON_ARGS "$@"
python ai8xize.py --verbose --log --test-dir $TARGET --prefix cats-dogs --checkpoint-file trained/ai85-catsdogs-qat8-q.pth.tar --config-file networks/cats-dogs-chw.yaml $COMMON_ARGS "$@"

python ai8xize.py --verbose --log --test-dir $TARGET --prefix CMSIS-Conv1D --config-file tests/test-conv1d.yaml --device CMSIS-NN
python ai8xize.py --verbose --log --test-dir $TARGET --prefix CMSIS-Conv1x1 --config-file tests/test-conv1x1.yaml --device CMSIS-NN

python ai8xize.py --verbose --log --test-dir $TARGET --prefix CMSIS-Nonsquare --config-file tests/test-nonsquare.yaml --device CMSIS-NN
python ai8xize.py --verbose --log --test-dir $TARGET --prefix CMSIS-NonsquarePool --config-file tests/test-nonsquare-pool.yaml --device CMSIS-NN
python ai8xize.py --verbose --log --test-dir $TARGET --prefix CMSIS-NonsquarePoolNonsquare --config-file tests/test-nonsquare-nonsquarepool.yaml --device CMSIS-NN
