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

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad0a --config-file tests/test-upsample-nonsquare-pad0A.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad0b --config-file tests/test-upsample-nonsquare-pad0B.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad0c --config-file tests/test-upsample-nonsquare-pad0C.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad1a --config-file tests/test-upsample-nonsquare-pad1A.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad1b --config-file tests/test-upsample-nonsquare-pad1B.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad1c --config-file tests/test-upsample-nonsquare-pad1C.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad2a --config-file tests/test-upsample-nonsquare-pad2A.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad2b --config-file tests/test-upsample-nonsquare-pad2B.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --autogen_list $AUTOGEN_LIST --log --test-dir $TARGET --prefix $PREFIX-upsample-pad2c --config-file tests/test-upsample-nonsquare-pad2C.yaml --device "$DEVICE" "$@"
