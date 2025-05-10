#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <ctype.h>

#include "functions.h"
#include "utils.h"

void execute_benchmarks(ProblemsParams* prob_params, ImplementationsParams* impl_params, 
  double cpu_execution_time, double* baseline_times, OutputFiles* output_files) {
  
  double a = prob_params->a;
  double b = prob_params->b;
  int n = prob_params->n;
  double* h_result = prob_params->h_result;
  double* d_result = prob_params->d_result;
  double* u_result = prob_params->u_result;

  double* h_fx_results = prob_params->h_fx_results;
  double* d_fx_results = prob_params->d_fx_results;
  double* u_fx_results = prob_params->u_fx_results;

  int num_blocks = prob_params->num_blocks;
  int threads_per_block = prob_params->threads_per_block;

  double speedup_gpu, efficiency_gpu;
  double speedup_vs_cpu, efficiency_vs_cpu;
  double start_time, end_time, gpu_execution_time;


  for (int i = 0; i < impl_params->num_implementations; i++) {
    TrapImplementation curr_impl = impl_params->implementations[i];
    const char* method_name = get_method_name(curr_impl);

    start_time = get_cur_time();

    switch (curr_impl) {
      case ATOMIC_ADD_GLOBAL_MEMORY:
        trap_host_atomic_add_global_memory(a, b, n, h_result, d_result, num_blocks, threads_per_block);
        break;
      case ATOMIC_ADD_UNIFIED_MEMORY:
        trap_host_atomic_add_unified_memory(a, b, n, u_result, num_blocks, threads_per_block);
        break;
      case REDUCTION_GLOBAL_MEMORY:
        trap_host_reduction_global_memory(a, b, n, h_result, h_fx_results, d_fx_results, num_blocks, threads_per_block);
        break;
      case REDUCTION_UNIFIED_MEMORY:
        trap_host_reduction_unified_memory(a, b, n, h_result, u_fx_results, num_blocks, threads_per_block);
        break;
      case REDUCTION_SHARED_MEMORY:
        trap_host_reduction_shared_memory(a, b, n, h_result, d_result, num_blocks, threads_per_block);
        break;
      case REDUCTION_SHARED_MEMORY_WARP_SHUFFLE:
        trap_host_shared_memory_warp_shuffle_reduction(a, b, n, h_result, d_result, num_blocks, threads_per_block);
        break;
      default:
        break;
    }
    end_time = get_cur_time();
    gpu_execution_time = end_time - start_time;

    if(prob_params->num_blocks == 1 && prob_params->threads_per_block == 1){
      baseline_times[i] = gpu_execution_time;
      speedup_gpu = 1.0;
      efficiency_gpu = 1.0;
    }else{
      speedup_gpu = baseline_times[i] / gpu_execution_time;
      efficiency_gpu = speedup_gpu / (prob_params->threads_per_block * prob_params->num_blocks);
    }

    speedup_vs_cpu = cpu_execution_time / gpu_execution_time;
    efficiency_vs_cpu = speedup_vs_cpu / (prob_params->threads_per_block * prob_params->num_blocks);

    fprintf(output_files->fp_gpu_results, "%s,%d,%d,%f,%f,%f\n",
      method_name,
      prob_params->threads_per_block,
      prob_params->num_blocks,
      gpu_execution_time,
      speedup_gpu,
      efficiency_gpu);

    fprintf(output_files->fp_cpu_results, "%s,%d,%d,%f,%f,%f\n",
      method_name,
      prob_params->threads_per_block,
      prob_params->num_blocks,
      gpu_execution_time,
      speedup_vs_cpu,
      efficiency_vs_cpu);
  }
}

int main() {

  // -----------------------------------------------------------------------------------------
  // Check GPU and GPU architecture 
  // -----------------------------------------------------------------------------------------
  cudaDeviceProp gpu_prop = check_GPU_and_architecture();

  // -----------------------------------------------------------------------------------------
  // Problem definition
  // -----------------------------------------------------------------------------------------
  double a = 0.0;
  double b = 2.0 * M_PI;
  int n = 1000000;
  get_integration_parameters(&a, &b, &n);

  FunctionChoice selected_function = FUNC_DEFAULT;
  get_function_choice(&selected_function);
  set_active_function(selected_function);

  // -----------------------------------------------------------------------------------------
  // Implementation params definition
  // -----------------------------------------------------------------------------------------
  ImplementationsParams impl_params;

  TrapImplementation implementations_to_test[] = {
    ATOMIC_ADD_GLOBAL_MEMORY,
    ATOMIC_ADD_UNIFIED_MEMORY,
    REDUCTION_GLOBAL_MEMORY,
    REDUCTION_UNIFIED_MEMORY,
    REDUCTION_SHARED_MEMORY,
    REDUCTION_SHARED_MEMORY_WARP_SHUFFLE,
  };
  int num_implementations = sizeof(implementations_to_test) / sizeof(implementations_to_test[0]);

  impl_params.implementations = implementations_to_test;
  impl_params.num_implementations = num_implementations;

  // -----------------------------------------------------------------------------------------
  // Problem params definition
  // -----------------------------------------------------------------------------------------
  ProblemsParams prob_params;
  
  double* h_result;
  h_result = (double*)malloc(sizeof(double));

  double* d_result;
  cudaMalloc((void**)&d_result, sizeof(double));

  double* u_result;
  cudaMallocManaged(&u_result, sizeof(double));

  double* h_fx_results;
  h_fx_results = (double*)malloc(n * sizeof(double));

  double* d_fx_results;
  cudaMalloc((void**)&d_fx_results, n * sizeof(double));

  double* u_fx_results;
  cudaMallocManaged((void**)&u_fx_results, n * sizeof(double));
  
  prob_params.a = a;
  prob_params.b = b;
  prob_params.n = n;
  prob_params.h_result = h_result;
  prob_params.d_result = d_result;
  prob_params.u_result = u_result;
  prob_params.h_fx_results = h_fx_results;
  prob_params.d_fx_results = d_fx_results;
  prob_params.u_fx_results = u_fx_results;

  // -----------------------------------------------------------------------------------------
  // Output files definition
  // -----------------------------------------------------------------------------------------
  OutputFiles out_files;

  FILE* fp_cpu_results = fopen("../results/cpu_results.csv", "w");
  FILE* fp_gpu_results = fopen("../results/gpu_results.csv", "w");

  fprintf(fp_cpu_results, "ImplementationMethod,ThreadsPerBlock,NumBlocks,ExecutionTimeSeconds,Speedup,Efficiency\n");
  fprintf(fp_gpu_results, "ImplementationMethod,ThreadsPerBlock,NumBlocks,ExecutionTimeSeconds,Speedup,Efficiency\n");

  out_files.fp_cpu_results = fp_cpu_results;
  out_files.fp_gpu_results = fp_gpu_results;

  // -----------------------------------------------------------------------------------------
  // Benckmark cpu
  // -----------------------------------------------------------------------------------------
  printf("Executing CPU benchmark with the iterative trapezoidal rule...\n");
  double cpu_start_time = get_cur_time();
  iterative_trap(a, b, n, h_result);
  double cpu_end_time = get_cur_time();
  double cpu_execution_time = cpu_end_time - cpu_start_time;
  printf("CPU benchmark completed. Results saved in cpu_results.csv\n");

  // -----------------------------------------------------------------------------------------
  // GPU Benchmarks
  // -----------------------------------------------------------------------------------------
  double baseline_times[num_implementations];

  // Baseline with 1 thread
  printf("\nGenerating GPU baseline with 1 thread...\n");
  prob_params.num_blocks = 1;
  prob_params.threads_per_block = 1;
  execute_benchmarks(&prob_params, &impl_params, cpu_execution_time, baseline_times, &out_files);
  printf("GPU baseline generated. Results saved in gpu_results.csv\n");

  int configs[] = {
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024
  };
  int num_configs = sizeof(configs) / sizeof(configs[0]);

  // Benchmarks with different configurations num_blocks and threads_per_block grows as a power of 2
  printf("\nGenerating GPU benchmarks with different configurations...\n");
  for (int i = 0; i < num_configs; i++) {
    prob_params.num_blocks = configs[i];
    prob_params.threads_per_block = configs[i];
    printf("Testing configuration: %d blocks and %d threads per block\n", prob_params.num_blocks, prob_params.threads_per_block);
    execute_benchmarks(&prob_params, &impl_params, cpu_execution_time, baseline_times, &out_files);
  }
  printf("GPU benchmarks generated. Results saved in gpu_results.csv and cpu_results.csv\n");

  // -----------------------------------------------------------------------------------------
  // Cleaning
  // -----------------------------------------------------------------------------------------
  cudaFree(u_result);
  cudaFree(d_result);
  free(h_result);

  cudaFree(u_fx_results);
  cudaFree(d_fx_results);
  free(h_fx_results);

  fclose(fp_cpu_results);
  fclose(fp_gpu_results);
}