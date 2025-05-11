#include <cmath>
#include <math.h>
#include <sys/time.h>

#include "functions.h"

FunctionChoice host_selected_function = FUNC_DEFAULT;
__constant__ FunctionChoice device_selected_function_const;

void set_active_function(FunctionChoice choice) {
  host_selected_function = choice;
  cudaMemcpyToSymbol(device_selected_function_const, &host_selected_function, sizeof(FunctionChoice));
}

__device__ __host__ double f(double x) {
    FunctionChoice current_choice;

    #ifdef __CUDA_ARCH__
        current_choice = device_selected_function_const;
    #else
        current_choice = host_selected_function;
    #endif

    switch (current_choice) {
        case FUNC_SIN_X:
            return sin(x);
        case FUNC_X_SQUARED:
            return x * x;
        case FUNC_DEFAULT:
        default:
            return x * pow(M_E, -x) * cos(2.0 * x);
    }
}

double iterative_trap(double a, double b, int n, double *h_result) {

  double h = (b - a) / n;
  *h_result = f(a) + f(b);

  for (int i = 1; i <= n - 1; i++) {
    double x_i = a + i * h;
    *h_result += 2.0 * f(x_i);
  }

  return (h / 2.0) * (*h_result);
}

__global__ void trap_kernel_atomic_add(double a, double h, int n, double *d_result) {

  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  double local_sum = 0.0;

  for(int i = id + 1; i < n; i += total_threads){
    double x = a + i * h;
    local_sum += f(x);
  }

  atomicAdd(d_result, local_sum);
}

double trap_host_atomic_add_global_memory(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block){
  
  double h = (b - a) / n;
  *h_result = 0.5 * (f(a) + f(b));

  cudaMemset(d_result, 0, sizeof(double));

  trap_kernel_atomic_add<<<num_blocks, threads_per_block>>>(a, h, n, d_result);
  cudaDeviceSynchronize();

  double fx_sum;
  cudaMemcpy(&fx_sum, d_result, sizeof(double), cudaMemcpyDeviceToHost);

  *h_result += fx_sum;
  *h_result = h * (*h_result);

  return *h_result;
}

double trap_host_atomic_add_unified_memory(double a, double b, int n, double* u_result, int num_blocks, int threads_per_block){
  double h = (b - a) / n;
  *u_result = 0.5 * (f(a) + f(b));

  trap_kernel_atomic_add<<<num_blocks, threads_per_block>>>(a, h, n, u_result);
  cudaDeviceSynchronize();

  *u_result = h * (*u_result);

  return *u_result;
}

__global__ void trap_kernel_host_reduction(double a, double h, int n, double* d_fx_results){
  int id = blockDim.x * blockIdx.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;

  for (int i = id + 1; i < n; i += total_threads) {
    double x_i = a + i * h;
    d_fx_results[i] = f(x_i);
  }
}

double trap_host_reduction_global_memory(double a, double b, int n, double* h_result, double* h_fx_results, double* d_fx_results, int num_blocks, int threads_per_block){

  double h = (b - a) / n;
  *h_result = 0.5 * (f(a) + f(b));

  trap_kernel_host_reduction<<<num_blocks, threads_per_block>>>(a, h, n, d_fx_results);
  cudaDeviceSynchronize();

  cudaMemcpy(h_fx_results, d_fx_results, n * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 1; i < n; i++) {
    *h_result += h_fx_results[i];
  }

  *h_result = h * (*h_result);

  return *h_result;

}

double trap_host_reduction_unified_memory(double a, double b, int n, double* h_result, double* u_fx_results, int num_blocks, int threads_per_block){
  double h = (b - a) / n;
  *h_result = 0.5 * (f(a) + f(b));

  trap_kernel_host_reduction<<<num_blocks, threads_per_block>>>(a, h, n, u_fx_results);
  cudaDeviceSynchronize();

  for (int i = 1; i < n; i++) {
    *h_result += u_fx_results[i];
  }

  *h_result = h * (*h_result);

  return *h_result;
}

__global__ void trap_kernel_shared_memory_reduction(double a, double h, int n, double* d_result) {
  extern __shared__ double sdata[];

  int tid_in_block = threadIdx.x;
  int threads_per_block_dim = blockDim.x;

  int id_global = threads_per_block_dim * blockIdx.x + tid_in_block;
  int total_threads_global = gridDim.x * threads_per_block_dim;
  double thread_local_sum = 0.0;

  for (int i = id_global + 1; i < n; i += total_threads_global) {
    double x = a + i * h;
    thread_local_sum += f(x);
  }

  sdata[tid_in_block] = thread_local_sum;
  __syncthreads();

  for (int stride = threads_per_block_dim / 2; stride > 0; stride /= 2) {
      if (tid_in_block < stride) {
        sdata[tid_in_block] += sdata[tid_in_block + stride];
      }
      __syncthreads();
  }

  if (tid_in_block == 0) {
      atomicAdd(d_result, sdata[0]);
  }
}

double trap_host_reduction_shared_memory(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block) {

  double h = (b - a) / n;
  *h_result = 0.5 * (f(a) + f(b)); 

  cudaMemset(d_result, 0, sizeof(double));

  size_t shared_mem_size = threads_per_block * sizeof(double); 
  trap_kernel_shared_memory_reduction<<<num_blocks, threads_per_block, shared_mem_size>>>(a, h, n, d_result);
  cudaDeviceSynchronize();

  double kernel_sum_result;
  cudaMemcpy(&kernel_sum_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

  *h_result += kernel_sum_result;
  *h_result = h * (*h_result);
  
  return *h_result;
}

__global__ void trap_kernel_warp_shuffle_reduction(double a, double h, int n, double* d_result) {
    int id_global = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads_global = gridDim.x * blockDim.x;
    double thread_local_sum = 0.0;

    for (int i = id_global + 1; i < n; i += total_threads_global) {
      double x = a + i * h;
      thread_local_sum += f(x);
    }

    unsigned mask = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      thread_local_sum += __shfl_down_sync(mask, thread_local_sum, offset);
    }

    if ((threadIdx.x % warpSize) == 0) {
      atomicAdd(d_result, thread_local_sum);
    }
}

double trap_host_warp_shuffle_reduction(double a, double b, int n, double* h_result, double* d_result, int num_blocks, int threads_per_block) {
    double h = (b - a) / n;
    *h_result = 0.5 * (f(a) + f(b));

    cudaMemset(d_result, 0, sizeof(double));

    trap_kernel_warp_shuffle_reduction<<<num_blocks, threads_per_block>>>(a, h, n, d_result);
    cudaDeviceSynchronize();

    double kernel_sum_result;
    cudaMemcpy(&kernel_sum_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);

    *h_result += kernel_sum_result;
    *h_result = h * (*h_result);

    return *h_result;
}

const char* get_method_name(TrapImplementation method) {
  switch (method) {
  case ATOMIC_ADD_GLOBAL_MEMORY:
    return "AtomicAdd(Global Memory)";
  case ATOMIC_ADD_UNIFIED_MEMORY:
    return "AtomicAdd(Unified Memory)";
  case REDUCTION_GLOBAL_MEMORY:
    return "Reduction(Host Array Global Memory)";
  case REDUCTION_UNIFIED_MEMORY:
    return "Reduction(Host Array Unified Memory)";
  case REDUCTION_SHARED_MEMORY:
    return "Reduction(Shared Memory - Tree Structured Sum)";
  case REDUCTION_WARP_SHUFFLE:
    return "Reduction(Warp Shuffle - Tree Structured Sum)";
  default:
    return "UnknownMethod";
  }
}