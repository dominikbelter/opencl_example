/** @file nnets_cl.h
 *
 * OpenCL example - leakyReLU
 *
 * @author Dominik Belter
 */

#ifndef H_NNETS
#define H_NNETS

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

void initializeOpenCL(int vector_size);
void leakyReLU_c(float leak, float *input, float *output, int index);
void computeoutput(float leak, float *input, float *output, int vector_size);
void releaseOpenCL();

#ifdef __cplusplus
}
#endif

#endif /* H_NNETS */
