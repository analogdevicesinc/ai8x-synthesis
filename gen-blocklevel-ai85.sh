#!/bin/sh
./cnn-gen.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --ai85
