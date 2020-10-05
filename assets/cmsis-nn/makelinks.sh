#!/bin/sh
ln -s ../../../CMSIS_5/CMSIS .
# ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c .
# ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic.c .
# ln -s CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast.c .
ln -s CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC.c .
ln -s CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c .
ln -s CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c .
