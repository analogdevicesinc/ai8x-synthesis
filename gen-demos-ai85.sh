#!/bin/sh
cp Device/*.c ../AI85SDK/Firmware/trunk/Applications/EvKitExamples/Common/
cp Device/tornadocnn.h ../AI85SDK/Firmware/trunk/Applications/EvKitExamples/Common/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir demos --prefix ai85-mnist --checkpoint-file trained/ai84-mnist.pth.tar --config-file networks/mnist-chw-ai85.yaml --ai85 --compact-data --mexpress
cp demos/ai85-mnist/* ../AI85SDK/Firmware/trunk/Applications/EvKitExamples/ai85-mnist/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir demos --prefix ai85-verify-cifar-bias --checkpoint-file trained/ai85-cifar10-bias.pth.tar --config-file networks/cifar10-hwc.yaml --ai85 --verify-writes --compact-data --mexpress
cp demos/ai85-verify-cifar-bias/* ../AI85SDK/Firmware/trunk/Applications/EvKitExamples/ai85-verify-cifar-bias/

./ai8xize.py -e --verbose --top-level cnn -L --test-dir demos --prefix ai85-fifostream-cifar2-hwc --checkpoint-file trained/ai84-cifar10.pth.tar --config-file tests/test-fifostream-cifar-hwc.yaml --ai85 --fifo
cp demos/ai85-fifostream-cifar2-hwc/* ../AI85SDK/Firmware/trunk/Applications/EvKitExamples/ai85-fifostream-cifar2-hwc/
