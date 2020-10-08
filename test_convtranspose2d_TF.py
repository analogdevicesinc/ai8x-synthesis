#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc,All Rights Reserved.
#
# Maxim Integrated Products, Inc,Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
# Written by RM
###################################################################################################
"""
Test the convtranspose2d operator.
"""
import numpy as np
import torch
import compute
import tensorflow as tf
import os
import sys
sys.path.append(os.path.join('../ai8x-training/TensorFlow'))
import ai8xTF 

STRIDE = 2
PAD = 1
DILATION = 1
OUTPUT_PAD = 1
def clamp(x, minimum=-128, maximum=127):
    """
    clamp with max/min
    """
    return np.array(tf.clip_by_value(x, minimum, maximum))

def deconvolve(groups, data, weight,w1, expected):

    """Upsample data"""
    print('Input:\n', data)

    wflip = np.flip(weight, axis=(2, 3)).swapaxes(0, 1)
    wunflip = np.flip(wflip, axis=(2, 3)).swapaxes(0, 1)
    assert np.array_equal(wunflip, weight)
    print("W/D:",weight.shape, data.shape)

    t = torch.nn.functional.conv_transpose2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),
        torch.as_tensor(np.flip(weight, axis=(2, 3)).swapaxes(0, 1).copy(), dtype=torch.float),
        bias=None,
        stride=STRIDE,
        padding=PAD,
        output_padding=OUTPUT_PAD,
        groups=groups,
        dilation=DILATION,
    ).int().squeeze().numpy()
    print("Pytorch :")
    print(t.shape, t//128)

    test_input = data
    input_layer = tf.keras.Input(shape=(3, 5))
    reshape = tf.keras.layers.Reshape(target_shape=(3, 5, 1))(input_layer)

    conv1 = tf.keras.layers.Conv2DTranspose(
        filters=2, 
        kernel_size=3, 
        strides=STRIDE, 
        padding="same", 
        output_padding=OUTPUT_PAD,
        dilation_rate=DILATION, 
        use_bias=False,
        kernel_initializer=tf.keras.initializers.constant(w1)
    )(reshape)

    model = tf.keras.Model(inputs=[input_layer], outputs=[conv1])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    output = model.predict(test_input).astype(np.int64)
    output = np.transpose(output,(0,3,1,2))
    output = np.flip(output)
    output = np.squeeze(output,0)

    # Model output
    print('TF/Keras :')
    print(output.shape,(output//128))
    
    test_input = data
    test_input = clamp(np.floor(test_input + 0.5))/128.0
    w2 =  clamp(np.floor(w1 + 0.5))/128.0
    input_layer = tf.keras.Input(shape=(3, 5))
    reshape = tf.keras.layers.Reshape(target_shape=(3, 5, 1))(input_layer)

    conv1 = ai8xTF.FusedConv2DTranspose(
        filters=2,
        kernel_size=3,
        strides=2,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.constant(w2)
    )(reshape)

    model = tf.keras.Model(inputs=[input_layer], outputs=[conv1])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    output1 = model.predict(test_input)

    print(output1.shape)
    output1 = np.transpose(output1,(0,3,1,2))
    output1 = np.flip(output1)
    output1 = np.squeeze(output1,0)
    print('ai8xTF :')
    print(output1.shape,(output1*128).astype(np.int64))


    if groups > 1:
        weight = weight.transpose(1, 0, 2, 3)
    print("Pytorch and TensorFlow results match" if np.array_equal(output, t) else "*** FAILURE ***")
    assert np.array_equal(output, t)

def test_convtranspose2d():
    """Main program to test compute.conv2d with fractional stride."""

    # 3x4x4 (CHW)
    d0 = np.array(
           [[[-90, -77, -64, -51, -38],
            [-26, -13,   0,  13,  26],
            [ 38,  51,  64,  77,  90]]],
        dtype=np.int64,
    )

    # 3x5x3x3
    w1 = np.array(
        [[[[[-115],
            [-102]],
           [[ -88],
            [ -75]],
           [[ -61],
            [ -47]]],
          [[[ -34],
            [ -20]],
           [[  -7],
            [   7]],
           [[  20],
            [  34]]],
          [[[  47],
            [  61]],
           [[  75],
            [  88]],
           [[ 102],
            [ 115]]]]],
        dtype=np.int64,
    )
    w0 = np.squeeze(w1,0)
    w0 = np.transpose(w0,(2,3,0,1)) #v(d)
    w0 = np.flip(w0, axis=(2, 3))

    # 5x8x8
    e0 = np.array(
       [[[[  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [ 81,  72],
         [ 62,  53],
         [112,  94],
         [ 53,  45],
         [ 94,  79],
         [ 44,  38],
         [ 76,  64],
         [ 35,  30],
         [ 58,  49],
         [ 26,  22],
         [ 18,  14],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [ 24,  14],
         [  5,  -5],
         [  6, -12],
         [  4,  -4],
         [  5, -10],
         [  4,  -3],
         [  4,  -9],
         [  3,  -3],
         [  2,  -8],
         [  2,  -2],
         [ -6, -10],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [-10, -22],
         [-35, -47],
         [-76, -98],
         [-36, -45],
         [-79, -95],
         [-37, -44],
         [-81, -92],
         [-39, -43],
         [-84, -89],
         [-40, -41],
         [-43, -44],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [  7,   4],
         [  1,  -1],
         [ -1,  -5],
         [  1,  -1],
         [ -2,  -3],
         [  0,   0],
         [ -3,  -2],
         [ -1,   1],
         [ -5,  -1],
         [ -1,   1],
         [  4,   7],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [-44, -43],
         [-41, -40],
         [-89, -84],
         [-43, -39],
         [-92, -81],
         [-44, -37],
         [-95, -79],
         [-45, -36],
         [-98, -76],
         [-47, -35],
         [-22, -10],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [-10,  -6],
         [ -2,   2],
         [ -8,   2],
         [ -3,   3],
         [ -9,   4],
         [ -3,   4],
         [-10,   5],
         [ -4,   4],
         [-12,   6],
         [ -5,   5],
         [ 14,  24],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [ 14,  18],
         [ 22,  26],
         [ 49,  58],
         [ 30,  35],
         [ 64,  76],
         [ 38,  44],
         [ 79,  94],
         [ 45,  53],
         [ 94, 112],
         [ 53,  62],
         [ 72,  81],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0]],
        [[  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0],
         [  0,   0]]]],
        dtype=np.int64,
    )
    '''
        [[[[ 81,  72],
          [ 62,  53],
          [112,  94],
          [ 53,  45],
          [ 94,  79],
          [ 44,  38],
          [ 76,  64],
          [ 35,  30],
          [ 58,  49],
          [ 26,  22],
          [ 18,  14]],
         [[ 24,  14],
          [  5,  -5],
          [  6, -12],
          [  4,  -4],
          [  5, -10],
          [  4,  -3],
          [  4,  -9],
          [  3,  -3],
          [  2,  -8],
          [  2,  -2],
          [ -6, -10]],
         [[-10, -22],
          [-35, -47],
          [-76, -98],
          [-36, -45],
          [-79, -95],
          [-37, -44],
          [-81, -92],
          [-39, -43],
          [-84, -89],
          [-40, -41],
          [-43, -44]],
         [[  7,   4],
          [  1,  -1],
          [ -1,  -5],
          [  1,  -1],
          [ -2,  -3],
          [  0,   0],
          [ -3,  -2],
          [ -1,   1],
          [ -5,  -1],
          [ -1,   1],
          [  4,   7]],
         [[-44, -43],
          [-41, -40],
          [-89, -84],
          [-43, -39],
          [-92, -81],
          [-44, -37],
          [-95, -79],
          [-45, -36],
          [-98, -76],
          [-47, -35],
          [-22, -10]],
         [[-10,  -6],
          [ -2,   2],
          [ -8,   2],
          [ -3,   3],
          [ -9,   4],
          [ -3,   4],
          [-10,   5],
          [ -4,   4],
          [-12,   6],
          [ -5,   5],
          [ 14,  24]],
         [[ 14,  18],
          [ 22,  26],
          [ 49,  58],
          [ 30,  35],
          [ 64,  76],
          [ 38,  44],
          [ 79,  94],
          [ 45,  53],
          [ 94, 112],
          [ 53,  62],
          [ 72,  81]]]],
    '''

    deconvolve(1, d0, w0, w1, e0)


if __name__ == '__main__':
    test_convtranspose2d()
