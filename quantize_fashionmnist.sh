#!/bin/sh
./quantize.py /home/chenot/checkpoint.pth.tar trained/ai85-fashionmnist.pth.tar --device MAX78000 -v -c networks/fashionmnist-chw-tf.yaml --scale 0.85 $@
