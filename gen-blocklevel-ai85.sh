#!/bin/sh
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-cifar --checkpoint-file trained/ai84-cifar10.pth.tar --config-file networks/cifar10-hwc.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-singlebyte-hwc --config-file tests/test-singlebyte-hwc.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-kmax_bmax_dmax --config-file tests/test-max.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-passthru-2-pool --config-file tests/test-passthrough-2-pool.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-conv1d-multilayer-q1248 --config-file tests/test-conv1d-multilayer-q1248.yaml --ai85 --stop-after 2
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-eltwise-pool --config-file tests/test-eltwiseadd-pool.yaml --ai85 --legacy-test
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-eltwise-poolafter-avg --config-file tests/test-eltwiseadd-poolafter-avg.yaml --ai85 --legacy-test
# ./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-fifostream-eltwise --config-file tests/test-fifostream-eltwise.yaml --ai85 --fifo --increase-start 1
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-passthroughmp --config-file tests/test-passthroughmultipass.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-wide3to508to3 --config-file tests/test-wide3to508to3.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-mlpflatten12to2 --config-file tests/test-mlpflatten12to2.yaml --ai85 --debug-computation --debug
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-conv1d-2 --config-file tests/test-conv1d-2.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-conv1x1 --config-file tests/test-conv1x1.yaml --ai85
./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-upscale --config-file tests/test-upscale.yaml --ai85 --stop-after 0
# ./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-nonsquare-mexpress-mlator --config-file tests/test-nonsquare.yaml --ai85 --mexpress --mlator
# ./ai8xize.py --verbose --autogen blocklevel -L --test-dir blocklevel --prefix ai85-riscv-qfastfifostream-32-hwc --config-file tests/test-fifostream-32-hwc.yaml --ai85 --fifo --riscv --riscv-flash --fast-fifo-quad --riscv-cache --input-fifo
