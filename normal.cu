#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define h_Ain(i, j) h_Ain[(i)*nx +(j)]
#define h_Aout(i, j) h_Aout[(i)*nx +(j)]


// CUDA kernel. Each thread takes care of one element of c
__global__ void stencil(float *d_Ain, float *d_Aout, int nx, int ny)
{
    #define d_Ain(i, j) d_Ain[(i)*nx +(j)]
    #define d_Aout(i, j) d_Aout[(i)*nx +(j)]

    // Get our global thread ID
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    float north, south, east, west;

    if ( (i > 0 && i < nx - 1) && (j > 0 && j < ny - 1) ) {
        float current =  d_Ain(i, j);
        north = d_Ain(i-1,j);
        south = d_Ain(i+1,j);
        east = d_Ain(i,j+1);
        west = d_Ain(i,j-1);

        d_Aout(i,j) = -4 * current + north + south + east + west;
    }
}

int main( int argc, char* argv[] )
{
    // Getting arguments
    int grid_size = atoi(argv[1]);
    int enable_pinned_memory = atoi(argv[2]);

    // Size of matrix
    int nx = grid_size + 2;
    int ny = grid_size + 2;
 
    // Host input vectors
    float *h_Ain;
    
    //Host output vector
    float *h_Aout;
 
    // Device input vectors
    float *d_Ain;
 
    //Device output vector
    float *d_Aout;

    // Size, in bytes, of each vector
    size_t bytes = nx*ny*sizeof(float);


    // Allocate memory for each vector on host
    if (enable_pinned_memory) {
        cudaMallocHost((void**)&h_Ain, bytes);
        cudaMallocHost((void**)&h_Aout, bytes);
        // memset(h_Aout, 0, bytes);
    } else {
        h_Ain = (float*)malloc(bytes);
        h_Aout = (float*)malloc(bytes);
    }
    
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_Ain, bytes);
    cudaMalloc(&d_Aout, bytes);

    int i,j;
    // Initialize vectors on host
    for( i = 0; i < ny; i++ ) {
        for(j = 0; j < nx; j++) {
            h_Ain(i,j) = rand();
        }
    }

    float ms; // elapsed time in milliseconds
    float total_time = 0;
    
    // create events and streams
    cudaEvent_t startEvent, stopEvent, dummyEvent;
    

    dim3 DimGrid(ceil(nx/16.0),ceil(ny/16.0));
    dim3 DimBlock(16,16);

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaEventCreate(&dummyEvent);

    clock_t total_time_begin = clock();

    // Copy host vectors to device
    cudaEventRecord(startEvent,0);
  
    cudaMemcpy(d_Ain, h_Ain, bytes, cudaMemcpyHostToDevice);

    //printf("Launching kernel stencil....... \n");

    stencil<<<DimGrid,DimBlock>>>(d_Ain, d_Aout, nx, ny);

    // Copy array back to host
    cudaMemcpy(h_Aout, d_Aout, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);
    cudaEventElapsedTime(&ms, startEvent, stopEvent);

    clock_t total_time_end = clock();
    total_time = ((double)(total_time_end - total_time_begin) / (CLOCKS_PER_SEC / 1000)); // output in milli seocnd

    printf("Total time (ms): %f\n", total_time);
    printf("Time for sequential transfer and execute (ms): %f\n", ms);

    // Release device memory
    cudaFree(d_Ain);
    cudaFree(d_Aout);

    // Release host memory
    if (enable_pinned_memory) {
        cudaFreeHost(h_Ain);
        cudaFreeHost(h_Aout);
    } else {
        free(h_Ain);
        free(h_Aout);
    }
 
    return 0;
}
