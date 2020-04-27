/*
 * Copyright (C) 2010-2018 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_pool_nonsquare_q7_HWC_nonsquare.c
 * Description:  Non-square pooling function implementations
 *               on non-square images
 *
 * $Date:        17. January 2018
 * $Revision:    V.1.0.0
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_math.h"
#include "arm_nnfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup Pooling
 * @{
 */

  /**
   * @brief Q7 max pooling function
   * @param[in, out]  Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_x     convolution stride y
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   Im_out       pointer to output tensor
   * @return none.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size:  0
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void
arm_maxpool_nonsquare_q7_HWC_nonsquare(q7_t * Im_in,
                                       const uint16_t dim_im_in_x,
                                       const uint16_t dim_im_in_y,
                                       const uint16_t ch_im_in,
                                       const uint16_t dim_kernel_x,
                                       const uint16_t dim_kernel_y,
                                       const uint16_t padding_x,
                                       const uint16_t padding_y,
                                       const uint16_t stride_x,
                                       const uint16_t stride_y,
                                       const uint16_t dim_im_out_x,
                                       const uint16_t dim_im_out_y,
                                       q7_t * bufferA, q7_t * Im_out)
{
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int       max = -129;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}

  /**
   * @brief Q7 average pooling function
   * @param[in, out]  Im_in        pointer to input tensor
   * @param[in]       dim_im_in_x  input tensor dimension x
   * @param[in]       dim_im_in_y  input tensor dimension y
   * @param[in]       ch_im_in     number of input tensor channels
   * @param[in]       dim_kernel_x filter kernel size x
   * @param[in]       dim_kernel_y filter kernel size y
   * @param[in]       padding_x    padding size x
   * @param[in]       padding_y    padding size y
   * @param[in]       stride_x     convolution stride x
   * @param[in]       stride_x     convolution stride y
   * @param[in]       dim_im_out_x output tensor dimension x
   * @param[in]       dim_im_out_y output tensor dimension y
   * @param[in,out]   bufferA      pointer to buffer space for input
   * @param[in,out]   Im_out       pointer to output tensor
   * @return none.
   *
   * @details
   *
   * <b>Buffer size:</b>
   *
   * bufferA size:  2*dim_im_out*ch_im_in
   *
   * This pooling function is input-destructive. Input data is undefined
   * after calling this function.
   *
   */

void
arm_avepool_nonsquare_q7_HWC_nonsquare(q7_t * Im_in,
                                       const uint16_t dim_im_in_x,
                                       const uint16_t dim_im_in_y,
                                       const uint16_t ch_im_in,
                                       const uint16_t dim_kernel_x,
                                       const uint16_t dim_kernel_y,
                                       const uint16_t padding_x,
                                       const uint16_t padding_y,
                                       const uint16_t stride_x,
                                       const uint16_t stride_y,
                                       const uint16_t dim_im_out_x,
                                       const uint16_t dim_im_out_y,
                                       q7_t * bufferA, q7_t * Im_out)
{
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */

    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int       sum = 0;
                int       count = 0;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            count++;
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum / count;
            }
        }
    }
}
