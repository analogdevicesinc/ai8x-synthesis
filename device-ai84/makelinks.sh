#!/bin/sh
ln -s ../../Device/Makefile .
ln -s ../../Device/*.h .
ln -s ../../Device/*.c .
ln -s ../../CMSIS_5/CMSIS .
ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c .
ln -s CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c .
ln -s CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c .
ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c .
ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c .
