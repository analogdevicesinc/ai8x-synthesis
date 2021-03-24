CC=gcc
CFLAGS=-I. -ICMSIS/Core/Include -ICMSIS/NN/Include -ICMSIS/DSP/Include -Wall -D__ARM_ARCH_6M__
LIB_FILES=arm_convolve_HWC_q7_basic.o arm_pool_q7_HWC.o arm_relu_q7.o arm_convolve_HWC_q7_fast.o arm_convolve_HWC_q7_basic_nonsquare.o arm_pool_q7_HWC_nonsquare.o arm_pool_nonsquare_q7_HWC_nonsquare.o arm_relu32_q7.o arm_fully_connected_q7.o arm_fully_connected_q7_q31.o

.PHONY: all
all: main

main: main.o $(LIB_FILES)
	$(CC) -o main main.o $(LIB_FILES)

DEPS=sampledata.h weights.h tornadocnn.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

.PHONY: clean
clean:
	rm main main.o $(LIB_FILES)
