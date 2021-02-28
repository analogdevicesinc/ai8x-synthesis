#!/bin/sh
AUTOGEN_LIST=untested_autogen_list

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-conv2d-id --config-file tests/test-conv2d-id.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-x4-readahead-cifar --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-readahead-cifar10-hwc.yaml --stop-after 1 --device "$DEVICE" --calcx4 --read-ahead "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-nopad-multipass5 --config-file tests/test-ffsx4readahead-nopad-multipass5.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-nopad-multipass5 --config-file tests/test-ffsx4readahead-nopad-multipass5.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-nopad-multipass5-bias --config-file tests/test-ffsx4readahead-nopad-multipass5-bias.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-nopad-multipass5-bias --config-file tests/test-ffsx4readahead-nopad-multipass5-bias.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-multipass5 --config-file tests/test-multipass5.yaml --device "$DEVICE" "$@"
