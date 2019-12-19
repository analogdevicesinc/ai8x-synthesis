#!/bin/sh
./cnn-gen.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --ai85
./cnn-gen.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --ai85
./cnn-gen.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-kmax_bmax_dmax --config-file tests/test-max.yaml --ai85
