#!/usr/bin/env python3
###################################################################################################
# Copyright (C) 2019 Maxim Integrated Products, Inc, All Rights Reserved.
#
# Maxim Integrated Products, Inc, Default Copyright Notice:
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

def deconvolve(groups, data, weight, w1, wl, expected):

    """Upsample data"""
    print('Input:\n', data.shape,data)
    wflip = np.flip(weight, axis=(2, 3)).swapaxes(0, 1)
    wunflip = np.flip(wflip, axis=(2, 3)).swapaxes(0, 1)
    assert np.array_equal(wunflip, weight)
    
    c = torch.nn.functional.conv_transpose2d(
        torch.as_tensor(data, dtype=torch.float).unsqueeze(0),
        torch.as_tensor(np.flip(weight, axis=(2, 3)).swapaxes(0, 1).copy(), dtype=torch.float),
        bias=None,
        stride=STRIDE,
        padding=PAD,
        output_padding=OUTPUT_PAD,
        groups=groups,
        dilation=DILATION,
    )

    c = (0.5 + c)//128

    c = torch.flatten(torch.as_tensor(c, dtype=torch.float))
    
    l = torch.nn.functional.linear(
        torch.as_tensor(c, dtype=torch.float),
        torch.as_tensor(wl.swapaxes(0, 1), dtype=torch.float),
    ).int().squeeze().numpy()

    l = np.flip(l)
    print("Pytorch :")
    l = clamp(np.floor(0.5 + l) / 128).astype(np.int64)
    print(l.shape, l)

    test_input = data
    input_layer = tf.keras.Input(shape=(3, 5))
    reshape = tf.keras.layers.Reshape(target_shape=(3, 5, 1))(input_layer)

    conv1 = tf.keras.layers.Conv2DTranspose(
        filters=1, 
        kernel_size=3, 
        strides=STRIDE, 
        padding="same", 
        output_padding=OUTPUT_PAD,
        dilation_rate=DILATION, 
        use_bias=False,
        kernel_initializer=tf.keras.initializers.constant(w1)
    )(reshape)
    conv1 = (0.5 + conv1)//128

    flat = tf.keras.layers.Flatten()(conv1)

    output_layer = tf.keras.layers.Dense(3, 
                                         use_bias=False,
                                         kernel_initializer=tf.keras.initializers.constant(wl)
                   )(flat)

    model = tf.keras.Model(inputs=[input_layer], outputs=[conv1, flat, output_layer])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    conv_out, flat_out, output = model.predict(test_input)
    output = np.squeeze(output,0)
 
    
    # Model output
    print('TF/Keras :')
    output = clamp(np.floor(0.5 + output) / 128).astype(np.int64)
    print(output.shape,output)
    
    test_input = data
    test_input = clamp(np.floor(test_input + 0.5))/128.0
    w2 =  clamp(np.floor(0.5 + w1))/128.0
    d1 = wl

    input_layer = tf.keras.Input(shape=(3, 5))
    reshape = tf.keras.layers.Reshape(target_shape=(3, 5, 1))(input_layer)
    conv1 = ai8xTF.FusedConv2DTranspose(
        filters=1,
        kernel_size=3,
        strides=2,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.constant(w2)
    )(reshape)

    flat = tf.keras.layers.Flatten()(conv1)

    output_layer = ai8xTF.FusedDense(3,
                                     wide=True,
                                     kernel_initializer=tf.keras.initializers.constant(d1))(flat)

    model = tf.keras.Model(inputs=[input_layer], outputs=[conv1,flat,output_layer])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    conv1_out, flat1_out, output1 = model.predict(test_input)

    output1 = np.squeeze(output1,0)
    output1 = clamp(np.floor(0.5 + output)).astype(np.int64)
    print('ai8xTF :')
    print(output1.shape)
    print((output1).astype(np.int64))


    if groups > 1:
        weight = weight.transpose(1, 0, 2, 3)

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
        [[[[[-13]],
           [[ 13]],
           [[-26]]],
          [[[ 26]],
           [[-38]],
           [[ 38]]],
          [[[-51]],
           [[ 51]],
           [[-64]]]]],
        dtype=np.int64,
    )
    w0 = np.squeeze(w1,0)
    w0 = np.transpose(w0,(2,3,0,1)) #v(d)

    wl = np.array(
            [[[-115, -114, -113 ],
              [-111, -110, -109 ],
              [-107, -106, -105 ],
              [-104, -102, -101 ],
              [-100,  -98,  -97 ],
              [ -96,  -95,  -93 ],
              [ -92,  -91,  -89 ],
              [ -88,  -87,  -86 ],
              [ -84,  -83,  -82 ],
              [ -80,  -79,  -78 ],
              [ -77,  -75,  -74 ],
              [ -73,  -71,  -70 ],
              [ -69,  -68,  -66 ],
              [ -65,  -64,  -62 ],
              [ -61,  -60,  -59 ],
              [ -57,  -56,  -55 ],
              [ -53,  -52,  -51 ],
              [ -50,  -48,  -47 ],
              [ -46,  -44,  -43 ],
              [ -42,  -41,  -39 ],
              [ -38,  -37,  -35 ],
              [ -34,  -33,  -32 ],
              [ -30,  -29,  -28 ],
              [ -26,  -25,  -24 ],
              [ -23,  -21,  -20 ],
              [ -19,  -17,  -16 ],
              [ -15,  -14,  -12 ],
              [ -11,  -10,   -8 ],
              [  -7,   -6,   -5 ],
              [  -3,   -2,   -1 ],
              [   1,    2,    3 ],
              [   5,    6,    7 ],
              [   8,   10,   11 ],
              [  12,   14,   15 ],
              [  16,   17,   19 ],
              [  20,   21,   23 ],
              [  24,   25,   26 ],
              [  28,   29,   30 ],
              [  32,   33,   34 ],
              [  35,   37,   38 ],
              [  39,   41,   42 ],
              [  43,   44,   46 ],
              [  47,   48,   50 ],
              [  51,   52,   53 ],
              [  55,   56,   57 ],
              [  59,   60,   61 ],
              [  62,   64,   65 ],
              [  66,   68,   69 ],
              [  70,   71,   73 ],
              [  74,   75,   77 ],
              [  78,   79,   80 ],
              [  82,   83,   84 ],
              [  86,   87,   88 ],
              [  89,   91,   92 ],
              [  93,   95,   96 ],
              [  97,   98,  100 ],
              [ 101,  102,  104 ],
              [ 105,  106,  107 ],
              [ 109,  110,  111 ],
              [ 113,  114,  115 ]]],
        dtype=np.int64,
    )
    wl = np.squeeze(wl,0)

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

    deconvolve(1, d0, w0, w1, wl, e0)


if __name__ == '__main__':
    test_convtranspose2d()
