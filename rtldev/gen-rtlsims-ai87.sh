#!/bin/sh
export DEVICE="MAX78002"
export TARGET="rtldev/rtlsim-ai87"
export SHORT_LOG="--log-last-only"
export PRELOAD="-preload"

export PREFIX="ai87"
rtldev/gen-rtlsims.sh --no-pll --no-pipeline --result-output "$@"

export PREFIX="ai87-pipeline"

rtldev/gen-rtlsims.sh --result-output "$@"

