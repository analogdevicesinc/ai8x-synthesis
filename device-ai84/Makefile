CC=gcc
CFLAGS=-I. -ICMSIS/Core/Include -ICMSIS/NN/Include -ICMSIS/DSP/Include -Wall -D__ARM_ARCH_6M__

.PHONY: all
all: main

main: main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_q8p7_q15.o arm_softmax_q8p7_q15_frac.o arm_fully_connected_q7_q8p7_opt.o arm_convolve_HWC_q7_fast.o arm_convolve_HWC_q7_basic_nonsquare.o arm_pool_q7_HWC_nonsquare.o arm_pool_nonsquare_q7_HWC_nonsquare.o arm_relu32_q7.o arm_softmax_q17p14_q15.o
	$(CC) -o main main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_q8p7_q15.o arm_softmax_q8p7_q15_frac.o arm_fully_connected_q7_q8p7_opt.o arm_convolve_HWC_q7_fast.o arm_convolve_HWC_q7_basic_nonsquare.o arm_pool_q7_HWC_nonsquare.o arm_pool_nonsquare_q7_HWC_nonsquare.o arm_relu32_q7.o arm_softmax_q17p14_q15.o

DEPS=sampledata.h weights.h tornadocnn.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean
clean:
	rm main main.o arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_softmax_q8p7_q15.o arm_softmax_q8p7_q15_frac.o arm_fully_connected_q7_q8p7_opt.o arm_convolve_HWC_q7_fast.o arm_convolve_HWC_q7_basic_nonsquare.o arm_pool_q7_HWC_nonsquare.o arm_pool_nonsquare_q7_HWC_nonsquare.o arm_relu32_q7.o arm_softmax_q17p14_q15.o
