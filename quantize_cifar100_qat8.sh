#!/bin/sh
./quantize.py trained/ai85-cifar100-unquantized.pth.tar trained/ai85-cifar100.pth.tar --device MAX78000 -v --qat_model --qat_weight_bits 8 $@
