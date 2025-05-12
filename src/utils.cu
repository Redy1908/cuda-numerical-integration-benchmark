#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#include "utils.h"

void get_integration_parameters(double *a_ptr, double *b_ptr, int *n_ptr) {
  printf("\n--- Define Integration Interval ---\n");
  do {
    printf("Enter the lower limit of integration (a): ");
    scanf("%lf", a_ptr);
    printf("Enter the upper limit of integration (b): ");
    scanf("%lf", b_ptr);

    if (*b_ptr <= *a_ptr) {
      printf("\nError: The upper limit (b) must be strictly greater than the lower limit (a).\n");
      printf("Please try again.\n\n");
    }
  } while (*b_ptr <= *a_ptr);

  printf("\n--- Define Number of Trapezoids ---\n");
  do {
    printf("Enter the number of trapezoids (n): ");
    scanf("%d", n_ptr);

    if (*n_ptr <= 0) {
      printf("\nError: The number of trapezoids (n) must be a positive integer.\n");
      printf("Please try again.\n\n");
    }
  } while (*n_ptr <= 0);
  printf("\n");
}

void select_integration_function() {
    int choice_input;
    FunctionChoice selected_function_choice;

    printf("\n--- Select Function to Integrate ---\n");
    printf("1: x * e^(-x) * cos(2x)\n");
    printf("2: sin(x)\n");
    printf("3: x^2\n");

    do {
        printf("Enter your choice (1-%d): ", NUM_FUNCTION_CHOICES); 
        if (scanf("%d", &choice_input) != 1) {
            while (getchar() != '\n');
            choice_input = 0;
        }

        if (choice_input >= 1 && choice_input <= NUM_FUNCTION_CHOICES) {
            selected_function_choice = (FunctionChoice)(choice_input - 1);
            break;
        } else {
            printf("\nError: Invalid choice. Please enter a number between 1 and %d.\n\n", NUM_FUNCTION_CHOICES);
        }
    } while (1);

    set_active_function(selected_function_choice);
    
    printf("\n");
}

double get_cur_time() {
    struct timespec ts;
    
    clock_gettime(CLOCK_MONOTONIC, &ts);
    
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

cudaDeviceProp check_GPU_and_architecture() {
    int deviceCount;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
  
    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d -> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Can't retrive the number of GPUs. Exiting...\n");
        exit(0);
    }
  
    if (deviceCount == 0) {
        printf("No CUDA GPU found. Exiting...\n");
        exit(0);
    }
  
    int device = 0;
    cudaSetDevice(device);
  
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
  
    FILE* fp_gpu_property = fopen("../results/gpu_property.csv", "w");
    fprintf(fp_gpu_property, "GPUName,SMs,MaxThreadsPerBlock,MaxThreadsPerSM,WarpSize\n");
  
    fprintf(fp_gpu_property, "%s,%d,%d,%d,%d\n",
      deviceProp.name,
      deviceProp.multiProcessorCount,
      deviceProp.maxThreadsPerBlock,
      deviceProp.maxThreadsPerMultiProcessor,
      deviceProp.warpSize
    );
  
    fclose(fp_gpu_property);
  
    return deviceProp;
  }
  