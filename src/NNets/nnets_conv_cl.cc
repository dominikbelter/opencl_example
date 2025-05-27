#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include "NNets/nnets_conv_cl.h"
#include <stdio.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define MASK_SIZE 3

cl_kernel kernelConv;
cl_mem inputConvClmem;
cl_mem outputConvClmem;
cl_mem maskConvClmem;
cl_command_queue commandQueueConv;
cl_program programConv;
cl_context contextConv;

/// Define platform and queues
cl_platform_id * platformsConv = NULL;
cl_uint     numplatformsConv;

/// devices
cl_device_id* deviceListConv = NULL;
cl_uint numDevicesConv;

//OpenCL kernel which is run for every work item created.
const char *convKernel =
        "__kernel                                                       \n"
        "void convKernel(__global unsigned char* input,                \n"
        "                  __global char* mask,                         \n"
        "                  __global unsigned char* output,              \n"
        "                  int rows,                                    \n"
        "                  int cols)                                    \n"
        "{                                                              \n"
        "    //Get the index                                            \n"
        "    int index = get_global_id(0);                              \n"
        "    int rowNo = index/cols;                                    \n"
        "    int colNo = index%cols;                                    \n"
        "    if (rowNo>0&&rowNo<rows-1&&colNo>0&&colNo<cols){           \n"
        "       int row,col;                                            \n"
        "       int sum=0;                                              \n"
        "       for (row=-1;row<2;row++){                               \n"
        "           for (col=-1;col<2;col++){                           \n"
        "               int maskIndex = (row+1)*3+(col+1);              \n"
        "               int imageIndex = (rowNo+row)*cols+(colNo+col);  \n"
        "               sum+= input[imageIndex]*mask[maskIndex];        \n"
        "           }                                                   \n"
        "       }                                                       \n"
        "       output[index] = (unsigned char)abs(sum);                \n"
        "    }                                                          \n"
        "}                                                              \n";

void conv_c(unsigned char *input, char *mask, unsigned char *output, int rows, int cols, int index)
{
    int rowNo = index/cols;
    int colNo = index%cols;
    if (rowNo>0&&rowNo<rows-1&&colNo>0&&colNo<cols){
        int row,col;
        int sum=0;
        for (row=-1;row<2;row++){
            for (col=-1;col<2;col++){
                int maskIndex = (row+1)*3+(col+1);
                int imageIndex = (rowNo+row)*cols+(colNo+col);
                sum+= input[imageIndex]*mask[maskIndex];
            }
        }
        output[index] = (unsigned char)abs(sum);
    }
}

void initializeOpenCLConv(int rows, int cols)
{
    int vector_size = rows*cols;
    
    //Set up the Platform
    cl_int clStatus = clGetPlatformIDs(0, NULL, &numplatformsConv);
    platformsConv = (cl_platform_id *)malloc(sizeof(cl_platform_id)*numplatformsConv);
    clStatus = clGetPlatformIDs(numplatformsConv, platformsConv, NULL);
    
    /// get num of devices
    clStatus = clGetDeviceIDs(platformsConv[0], CL_DEVICE_TYPE_GPU, 0,NULL, &numDevicesConv);
    /// allocate memory
    deviceListConv = (cl_device_id *)malloc(sizeof(cl_device_id)*numDevicesConv);
    clStatus = clGetDeviceIDs(platformsConv[0], CL_DEVICE_TYPE_GPU, numDevicesConv, deviceListConv, NULL);
    
    /// Create context
    contextConv = clCreateContext(NULL, numDevicesConv, deviceListConv, NULL, NULL, &clStatus);
    
    /// Create command queue
    commandQueueConv = clCreateCommandQueue(contextConv, deviceListConv[0], 0, &clStatus);
    
    /// Define memory objects
    inputConvClmem = clCreateBuffer(contextConv, CL_MEM_READ_ONLY, vector_size * sizeof(unsigned char), NULL, &clStatus);
    maskConvClmem = clCreateBuffer(contextConv, CL_MEM_READ_ONLY, MASK_SIZE*MASK_SIZE * sizeof(char), NULL, &clStatus);
    outputConvClmem = clCreateBuffer(contextConv, CL_MEM_WRITE_ONLY, vector_size * sizeof(unsigned char), NULL, &clStatus);

    /// Create the program
    programConv = clCreateProgramWithSource(contextConv, 1,(const char **)&convKernel, NULL, &clStatus);
    
    /// Build program
    clStatus = clBuildProgram(programConv, 1, deviceListConv, NULL, NULL, NULL);
    if (clStatus!=0){
        printf("Error building program\n");
    }
    
    /// Create the kernel
    kernelConv = clCreateKernel(programConv, "convKernel", &clStatus);
}

void computeoutputConv(unsigned char *input, char *mask, unsigned char *output, int rows, int cols){
    /// copy arguments to the device
    cl_int clStatus = clEnqueueWriteBuffer(commandQueueConv, inputConvClmem, CL_TRUE, 0, rows*cols * sizeof(unsigned char),
                                           input, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueueConv, maskConvClmem, CL_TRUE, 0, MASK_SIZE*MASK_SIZE * sizeof(char),
                                           mask, 0, NULL, NULL);
    clStatus = clEnqueueWriteBuffer(commandQueueConv, outputConvClmem, CL_TRUE, 0, rows*cols * sizeof(unsigned char),
                                               output, 0, NULL, NULL);
    if (clStatus!=0)
    {
        printf("Error %d\n",clStatus);
        if (clStatus == CL_OUT_OF_HOST_MEMORY){
            printf("CL_OUT_OF_HOST_MEMORY\n");
        }
        else if (clStatus == CL_MEM_OBJECT_ALLOCATION_FAILURE){
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
        }
        else if (clStatus == CL_INVALID_MEM_OBJECT){
            printf("CL_INVALID_MEM_OBJECT\n");
        }
        return;
    }
    
    /// Define arguments
    clStatus = clSetKernelArg(kernelConv, 0, sizeof(cl_mem), (void *)&inputConvClmem);
    clStatus = clSetKernelArg(kernelConv, 1, sizeof(cl_mem), (void *)&maskConvClmem);
    clStatus = clSetKernelArg(kernelConv, 2, sizeof(cl_mem), (void *)&outputConvClmem);
    clStatus = clSetKernelArg(kernelConv, 3, sizeof(int), (void *)&rows);
    clStatus = clSetKernelArg(kernelConv, 4, sizeof(int), (void *)&cols);
    
    /// Execute the kernel
    size_t global_size = cols*rows; // Process the entire lists
    size_t local_size = 64;           // Process one item at a time
    clStatus = clEnqueueNDRangeKernel(commandQueueConv, kernelConv, 1, NULL, &global_size, NULL,
                                      0, NULL, NULL);
    
    /// Download results
    clStatus = clEnqueueReadBuffer(commandQueueConv, outputConvClmem, CL_TRUE, 0, cols*rows * sizeof(unsigned char),
                                   output, 0, NULL, NULL);

    /// finish
    clStatus = clFlush(commandQueueConv);
    clStatus = clFinish(commandQueueConv);
}

void releaseOpenCL(){
    /// clean and release memory
    cl_int clStatus = clReleaseKernel(kernelConv);
    if (clStatus!=0)
        printf("Error release\n");
    clStatus = clReleaseProgram(programConv);
    clStatus = clReleaseMemObject(inputConvClmem);
    clStatus = clReleaseMemObject(outputConvClmem);
    clStatus = clReleaseMemObject(maskConvClmem);
    clStatus = clReleaseCommandQueue(commandQueueConv);
    clStatus = clReleaseContext(contextConv);

    free(platformsConv);
    free(deviceListConv);
}
