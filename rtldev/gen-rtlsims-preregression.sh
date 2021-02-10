#!/bin/sh

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mpflatten-128 --config-file tests/test-mpflatten-128.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mpflatten-192 --config-file tests/test-mpflatten-256.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mpflatten-256 --config-file tests/test-mpflatten-256.yaml --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mppool-1x1 --config-file tests/test-mppool-1x1.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mppool-128 --config-file tests/test-mppool-128.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mppool-192 --config-file tests/test-mppool-192.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-mppool-256 --config-file tests/test-mppool-256.yaml --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-flatten-bias --config-file tests/test-flatten-bias.yaml --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-tfrock-bias --config-file tests/test-tfrock-bias.yaml --ignore-bias-groups --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-riscv-csv-qfastfifostream-x4-likecifar --config-file tests/test-ffsx4-likecifar10-hwc.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-readahead-multipass --config-file tests/test-ffsreadahead-multipass.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-readahead-multipass --config-file tests/test-ffsreadahead-multipass.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-multipass --config-file tests/test-ffsx4readahead-multipass.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-multipass --config-file tests/test-ffsx4readahead-multipass.yaml --fast-fifo --riscv --device "$DEVICE" "$@"

./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-readahead-multipass-bias --config-file tests/test-ffsreadahead-multipass-bias.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-readahead-multipass-bias --config-file tests/test-ffsreadahead-multipass-bias.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-qfastfifostream-x4-readahead-multipass-bias --config-file tests/test-ffsx4readahead-multipass-bias.yaml --fast-fifo-quad --riscv --device "$DEVICE" "$@"
./ai8xize.py --rtl"$PRELOAD" --verbose --autogen $TARGET --log --test-dir $TARGET --prefix $PREFIX-fastfifostream-x4-readahead-multipass-bias --config-file tests/test-ffsx4readahead-multipass-bias.yaml --fast-fifo --riscv --device "$DEVICE" "$@"
