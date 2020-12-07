#!/bin/sh
export DEVICE="MAX78000"
export TARGET="rtldev/rtlsim-ai85"
export SHORT_LOG="--log-last-only"
export PRELOAD=""

export PREFIX="ai85"
rtldev/gen-rtlsims.sh "$@"

