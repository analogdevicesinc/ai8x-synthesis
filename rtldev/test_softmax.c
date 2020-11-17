/**
 * @file    test_softmax.c
 * @brief   Test software Softmax
 * @details This test evaluates the Softmax functions.
 */

/*******************************************************************************
 * Copyright (C) Maxim Integrated Products, Inc., All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL MAXIM INTEGRATED BE LIABLE FOR ANY CLAIM, DAMAGES
 * OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Except as contained in this notice, the name of Maxim Integrated
 * Products, Inc. shall not be used except as stated in the Maxim Integrated
 * Products, Inc. Branding Policy.
 *
 * The mere transfer of this software does not imply any licenses
 * of trade secrets, proprietary technology, copyrights, patents,
 * trademarks, maskwork rights, or any other form of intellectual
 * property whatsoever. Maxim Integrated Products, Inc. retains all
 * ownership rights.
 *
 ******************************************************************************/

/***** Includes *****/
#include <stdio.h>
#include <stdint.h>

#include "arm_math.h"
#include "arm_nnfunctions.h"
#include "tornadocnnh.h"

/***** Definitions *****/
void test_q17p14_q15_sigmoid();
void test_q17p14_q15_vector();
void test_q17p14_q15_Sat();

void test_q8p7_q15_sigmoid();
void test_q8p7_q15_fract_sigmoid();
void test_q15_q15_sigmoid();

/***** Globals *****/

/***** Functions *****/

int main(void)
{
    printf("Hello Softmax!\n");


    test_q17p14_q15_vector();
    test_q17p14_q15_Sat();
    test_q17p14_q15_sigmoid();

   // test_q8p7_q15_sigmoid();
   // test_q8p7_q15_fract_sigmoid();
   // test_q15_q15_sigmoid();

    while(1);

}


/**
 * @brief  creates data for softmax sigmoide shape
 * @param[in]       none
 * @param[out]      none
 * @return none.
 *
 * @details
 *
 * Two element vector is fed to arm_softmax_xxx, one
 * is always zero and the other sweeps from -250k to 250k.
 * Prints outputs in integer and float format
 */
void test_q8p7_q15_sigmoid()
{
    q15_t vec_in[2];
    q15_t p_out[2];

    for (int i = -2000 ; i<2000; i+=1)
	{
		vec_in[0] = i;
		vec_in[1] = 0;

		arm_softmax_q8p7_q15(vec_in, 2, p_out);

		printf("%d,%f,%d,%f,%d,%f\r\n", vec_in[0],(double)(vec_in[0]/128.0),  p_out[0], (double)(p_out[0]/32768.0), p_out[1], (double)(p_out[1]/32768.0));

	}
}


void test_q8p7_q15_fract_sigmoid()
{
    q15_t vec_in[2];
    q15_t p_out[2];

    for (int i = -2000 ; i<2000; i+=1)
	{
		vec_in[0] = i;
		vec_in[1] = 0;

		arm_softmax_q8p7_q15_frac(vec_in, 2, p_out);

		printf("%d,%f,%d,%f,%d,%f\r\n", vec_in[0],(double)(vec_in[0]/128.0),  p_out[0], (double)(p_out[0]/32768.0), p_out[1], (double)(p_out[1]/32768.0));

	}
}

void test_q15_q15_sigmoid()
{
    q15_t vec_in[2];
    q15_t p_out[2];

    for (int i = -2000 ; i<2000; i+=1)
	{
		vec_in[0] = i;
		vec_in[1] = 0;

		arm_softmax_q15(vec_in, 2, p_out);

		printf("%d,%f,%d,%f,%d,%f\r\n", vec_in[0],(double)(vec_in[0]/32768.0),  p_out[0], (double)(p_out[0]/32768.0), p_out[1], (double)(p_out[1]/32768.0));

	}
}

void test_q17p14_q15_sigmoid()
{
    q31_t vec_in_q31[2];
    q15_t p_out[2];

	for (long int i = -250000 ; i<250000; i+=50)
	{
		vec_in_q31[0] = i;
		vec_in_q31[1] = 0;

		softmax_q17p14_q15(vec_in_q31, 2, p_out);

		printf("%d,%f,%d,%f,%d,%f\r\n", vec_in_q31[0],(double)(vec_in_q31[0]/16384.0),  p_out[0], (double)(p_out[0]/32768.0), p_out[1], (double)(p_out[1]/32768.0));

	}

}

void test_q17p14_q15_vector()
{
	q31_t vec_in_q31[10] =
	{
			0xffff7869, // 0
			0x00009278, // 1
			0x0000585e, // 2
			0x00004042, // 3
			0x000008e8, // 4
			0xffff40ba, // 5
			0xfffeb2f3, // 6
			0x00021351, // 7
			0xffff6a0a, // 8
			0xffffdc42  // 9
	};

    q15_t p_out[10];

	softmax_q17p14_q15(vec_in_q31, 10, p_out);

	printf("\r\n +++ Test Vector Input:Output \r\n");
	for (int i=0;i<10;i++)
	{
		printf("%d (%f): %d (%f)\r\n", vec_in_q31[i],(double)(vec_in_q31[i]/16384.0),  p_out[i], (double)(p_out[i]/32768.0));
	}
}


void test_q17p14_q15_Sat()
{
	q31_t vec_in_q31[10];
    q15_t p_out[10];

#define MAX_INPUT  0x00FFFFFF

	for (int i=0;i<10;i++)
	{
		vec_in_q31[i] = MAX_INPUT;
	}

	printf("\r\n +++  Max 24bit Test Vector Input:Output \r\n");
	softmax_q17p14_q15(vec_in_q31, 10, p_out);

	for (int i=0;i<10;i++)
	{
		printf("%d (%f): %d (%f)\r\n", vec_in_q31[i],(double)(vec_in_q31[i]/16384.0),  p_out[i], (double)(p_out[i]/32768.0));
	}

	for (int i=0;i<10;i++)
	{
		vec_in_q31[i] = -MAX_INPUT;
	}

	printf("\r\n +++  Min 24bit Test Vector Input:Output \r\n");
	softmax_q17p14_q15(vec_in_q31, 10, p_out);

	for (int i=0;i<10;i++)
	{
		printf("%d (%f): %d (%f)\r\n", vec_in_q31[i],(double)(vec_in_q31[i]/16384.0),  p_out[i], (double)(p_out[i]/32768.0));
	}

}

