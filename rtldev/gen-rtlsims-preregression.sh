#!/bin/sh
AUTOGEN_LIST=untested_autogen_list

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-x4-likecifar --config-file tests/test-ffsx4-likecifar10-hwc.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-readahead-multipass --config-file tests/test-ffsreadahead-multipass.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-readahead-multipass --config-file tests/test-ffsreadahead-multipass.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-multipass --config-file tests/test-ffsx4readahead-multipass.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-multipass --config-file tests/test-ffsx4readahead-multipass.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-multipass3 --config-file tests/test-ffsx4readahead-multipass3.yaml --fast-fifo --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-readahead-multipass-bias --config-file tests/test-ffsreadahead-multipass-bias.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-readahead-multipass-bias --config-file tests/test-ffsreadahead-multipass-bias.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-multipass-bias --config-file tests/test-ffsx4readahead-multipass-bias.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-multipass-bias --config-file tests/test-ffsx4readahead-multipass-bias.yaml --fast-fifo --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-stream-cifar-transition-zeroize --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --device "$DEVICE" --zero-sram --allow-streaming --queue-name long "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-fifostream-cifar-transition-zeroize --checkpoint-file tests/test-cifar10.pth.tar --config-file tests/test-stream-cifar-transition.yaml --device "$DEVICE" --zero-sram --fifo "$@"
