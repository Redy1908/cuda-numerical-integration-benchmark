#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double a;
    double b;
    int n;
    double* h_result;
    double* d_result;
    double* u_result;
    double* h_fx_results;
    double* d_fx_results;
    double* u_fx_results;
    int num_blocks;
    int threads_per_block;
} ProblemsParams;

typedef enum {
    FUNC_DEFAULT,      // x * e^(-x) * cos(2x)
    FUNC_SIN_X,        // sin(x)
    FUNC_X_SQUARED,    // x^2
    NUM_FUNCTION_CHOICES
} FunctionChoice;

extern FunctionChoice host_selected_function;

typedef enum {
    ATOMIC_ADD_GLOBAL_MEMORY,
    ATOMIC_ADD_UNIFIED_MEMORY,
    REDUCTION_GLOBAL_MEMORY,
    REDUCTION_UNIFIED_MEMORY,
    REDUCTION_SHARED_MEMORY,
    REDUCTION_WARP_SHUFFLE
}TrapImplementation;

typedef struct {
    TrapImplementation* implementations;
    int num_implementations;
} ImplementationsParams;

typedef struct{
    FILE* fp_cpu_results;
    FILE* fp_gpu_results;
} OutputFiles;

__device__ __host__ double f(double x);

double iterative_trap(double a, double b, int n, double* h_result);

__global__ void trap_kernel_atomic_add(double a, double h, int n, double* d_result);
double trap_host_atomic_add_global_memory(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block);
double trap_host_atomic_add_unified_memory(double a, double b, int n, double* u_result, int num_blocks, int threads_per_block);

__global__ void trap_kernel_host_reduction(double a, double h, int n, double* d_fx_results);
double trap_host_reduction_global_memory(double a, double b, int n, double* h_result, double* h_fx_results, double* d_fx_results, int num_blocks, int threads_per_block);
double trap_host_reduction_unified_memory(double a, double b, int n, double* h_result, double* u_fx_results, int num_blocks, int threads_per_block);

__global__ void trap_kernel_shared_memory_reduction(double a, double h, int n, double* d_result);
double trap_host_reduction_shared_memory(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block);

__global__ void trap_kernel_warp_shuffle_reduction(double a, double h, int n, double* d_result);
double trap_host_warp_shuffle_reduction(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block);

void set_active_function(FunctionChoice choice); 
const char* get_method_name(TrapImplementation method);

#ifdef __cplusplus
}
#endif

#endif 