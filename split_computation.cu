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

    if ( (i > 0 && i < (nx/2)) && (j > 0 && j < ny - 1) ) {
        float current =  d_Ain(i, j);
        north = d_Ain(i-1,j);
        south = d_Ain(i+1,j);
        east = d_Ain(i,j+1);
        west = d_Ain(i,j-1);

        d_Aout(i,j) = -4 * current + north + south + east + west;
    }
}

void cpu_side_stencil(float *h_Ain, float *h_Aout, int nx, int ny) {
    float north, south, east, west;
    int i, j;
    for (i = nx/2, j = 0; (i < nx - 1) && (j < ny - 1); i++, j++) {
        float current =  h_Ain(i, j);
        north = h_Ain(i-1,j);
        south = h_Ain(i+1,j);
        east = h_Ain(i,j+1);
        west = h_Ain(i,j-1);

        h_Aout(i,j) = -4 * current + north + south + east + west;
    }
}

int main( int argc, char* argv[] )
{
    int grid_size = atoi(argv[1]);
    int enable_pinned_memory = atoi(argv[2]);

    // Size of matrix
    int nx = grid_size + 2;
    int ny = grid_size + 2;
 
    // Host input vectors
    float *h_Ain;
    
    // Host output vector
    float *h_Aout;
 
    // Device input vectors
    float *d_Ain;
 
    // Device output vector
    float *d_Aout;

    // Size, in bytes, of each vector
    size_t bytes = nx*ny*sizeof(float);

    // time measurement variables
    float memcpy_htod_time = 0;
    float memcpy_dtoh_time = 0;
    float gpu_compute_time = 0;
    float cpu_compute_time = 0;
    float total_time = 0;
    
    // create events
    cudaEvent_t memcpy_htod_start_event, memcpy_htod_stop_event;
    cudaEvent_t memcpy_dtoh_start_event, memcpy_dtoh_stop_event;
    cudaEvent_t gpu_compute_start_event, gpu_compute_stop_event;
    cudaEventCreate(&memcpy_htod_start_event);
    cudaEventCreate(&memcpy_htod_stop_event);
    cudaEventCreate(&memcpy_dtoh_start_event);
    cudaEventCreate(&memcpy_dtoh_stop_event);
    cudaEventCreate(&gpu_compute_start_event);
    cudaEventCreate(&gpu_compute_stop_event);


    // Allocate memory for each vector on host
    if (enable_pinned_memory) {
        cudaMallocHost((void**)&h_Ain, bytes);
        cudaMallocHost((void**)&h_Aout, bytes);
    } else {
        h_Ain = (float*)malloc(bytes);
        h_Aout = (float*)malloc(bytes);
    }
    
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_Ain, bytes/2);
    cudaMalloc(&d_Aout, bytes/2);

    int i,j;
    // Initialize vectors on host
    for( i = 0; i < ny; i++ ) {
        for(j = 0; j < nx; j++) {
            h_Ain(i,j) = rand();
        }
    }

    dim3 DimGrid(ceil(nx/16.0),ceil((ny/2)/16.0));
    dim3 DimBlock(16,16);

    clock_t total_time_begin = clock();

    // Copy host vectors to device
    cudaEventRecord(memcpy_htod_start_event, 0);

    cudaMemcpy(d_Ain, h_Ain, bytes/2, cudaMemcpyHostToDevice);

    cudaEventRecord(memcpy_htod_stop_event, 0);
    cudaEventSynchronize(memcpy_htod_stop_event);
    cudaEventElapsedTime(&memcpy_htod_time, memcpy_htod_start_event, memcpy_htod_stop_event);


    // start kernel computaion
    cudaEventRecord(gpu_compute_start_event, 0);

    stencil<<<DimGrid,DimBlock>>>(d_Ain, d_Aout, nx, ny/2);

    cudaEventRecord(gpu_compute_stop_event, 0);
    cudaEventSynchronize(gpu_compute_stop_event);
    cudaEventElapsedTime(&gpu_compute_time, gpu_compute_start_event, gpu_compute_stop_event);

    //Computation on Host

    clock_t cpu_begin = clock();

    cpu_side_stencil(h_Ain, h_Aout, nx, ny);

    clock_t cpu_end = clock();
    cpu_compute_time = ((double)(cpu_end - cpu_begin) / (CLOCKS_PER_SEC / 1000)); // output in milli seocnd
    

    // Copy array back to host
    cudaEventRecord(memcpy_dtoh_start_event, 0);

    cudaMemcpy(h_Aout, d_Aout, bytes/2, cudaMemcpyDeviceToHost);

    cudaEventRecord(memcpy_dtoh_stop_event, 0);
    cudaEventSynchronize(memcpy_dtoh_stop_event);
    cudaEventElapsedTime(&memcpy_dtoh_time, memcpy_dtoh_start_event, memcpy_dtoh_stop_event);


    clock_t total_time_end = clock();
    total_time = ((double)(total_time_end - total_time_begin) / (CLOCKS_PER_SEC / 1000)); // output in milli seocnd

    // print results
    printf("Total time (ms): %f\n", total_time);
    printf("Time for executution (GPU) (ms): %f\n", gpu_compute_time);
    printf("Time for executution (CPU) (ms): %f\n", cpu_compute_time);
    printf("Time for memory copy (HtoD) (ms): %f\n", memcpy_htod_time);
    printf("Time for memory copy (DtoH) (ms): %f\n", memcpy_dtoh_time);

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
