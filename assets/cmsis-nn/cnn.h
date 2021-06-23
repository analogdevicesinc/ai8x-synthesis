/*
 * On-device execution
 */

#include <arm_math.h>
#include <arm_nnfunctions.h>

#include "weights.h"


arm_status arm_fully_connected_q7_q31(const q7_t *pV,
                                      const q7_t *pM,
                                      const uint16_t dim_vec,
                                      const uint16_t num_of_rows,
                                      const uint16_t bias_shift,
                                      const uint16_t out_shift,
                                      const q7_t *bias,
                                      q31_t *pOut,
                                      q15_t *vec_buffer);


void arm_softmax_q8p7_q15(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

void arm_softmax_q8p7_q15_frac(const q15_t * vec_in, const uint16_t dim_vec, q15_t * p_out);

void arm_maxpool_q7_HWC_nonsquare(q7_t * Im_in,
                                  const uint16_t dim_im_in_x,
                                  const uint16_t dim_im_in_y,
                                  const uint16_t ch_im_in,
                                  const uint16_t dim_kernel,
                                  const uint16_t padding,
                                  const uint16_t stride,
                                  const uint16_t dim_im_out_x,
                                  const uint16_t dim_im_out_y,
                                  q7_t * bufferA, q7_t * Im_out);

void arm_avepool_q7_HWC_nonsquare(q7_t * Im_in,
                                  const uint16_t dim_im_in_x,
                                  const uint16_t dim_im_in_y,
                                  const uint16_t ch_im_in,
                                  const uint16_t dim_kernel,
                                  const uint16_t padding,
                                  const uint16_t stride,
                                  const uint16_t dim_im_out_x,
                                  const uint16_t dim_im_out_y,
                                  q7_t * bufferA, q7_t * Im_out);

void arm_maxpool_nonsquare_q7_HWC_nonsquare(q7_t * Im_in,
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
                                            q7_t * bufferA, q7_t * Im_out);

void arm_avepool_nonsquare_q7_HWC_nonsquare(q7_t * Im_in,
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
                                            q7_t * bufferA, q7_t * Im_out);


void arm_relu32_q7(q7_t * data, uint32_t size);
