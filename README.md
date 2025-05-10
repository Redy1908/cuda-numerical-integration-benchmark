# Parallel Computing Project: Numerical Integration with CUDA

This project explores the implementation and benchmarking of various parallel reduction strategies using CUDA C++ to calculate definite integrals with the trapezoidal rule. It compares the performance of different GPU techniques against each other and against a sequential CPU implementation.

## Description

The software calculates the definite integral of a function `f(x)` over an interval `[a, b]` by dividing the area under the curve into `n` trapezoids. The user can select the function to integrate, the integration limits, and the number of trapezoids. Benchmarks are then performed on several GPU implementations and one CPU implementation, measuring execution time, speedup, and efficiency.

## Key Features

*   **Numerical Integration:** Calculation of definite integrals using the trapezoidal rule.
*   **Dynamic Function Selection:** The user can choose from several predefined `f(x)` functions:
    *   `x * e^(-x) * cos(2x)` (Default)
    *   `sin(x)`
    *   `x^2`
    *   (Easily extensible to add more functions)
*   **Configurable Parameters:** The user can specify integration limits `a` and `b`, and the number of subdivisions `n`.
*   **Implemented GPU Reduction Strategies:**
    1.  `AtomicAdd (Global Memory)`: Atomic summation of partial results in global memory.
    2.  `AtomicAdd (Unified Memory)`: Same as above, but using unified memory.
    3.  `Reduction (Host Array Global Memory)`: GPU threads compute `f(x_i)` and write them to an array in global memory; the final sum occurs on the host.
    4.  `Reduction (Host Array Unified Memory)`: Similar to the previous, but with unified memory.
    5.  `Reduction (Shared Memory - Tree Structured Sum)`: Classic tree-structured reduction using shared memory within each block.
    6.  `Reduction (Shared Memory - Warp Shuffle - Tree Structured Sum)`: Warp-level reduction using `__shfl_down_sync` instructions, with partial sums from warps combined atomically.
*   **Detailed Benchmarking:**
    *   Comparison of GPU implementations' performance against a GPU baseline (1 thread, 1 block).
    *   Comparison of GPU implementations' performance against a sequential CPU implementation.
    *   Tests with various GPU grid and block configurations.
*   **Results Output:**
    *   Benchmark results are saved to CSV files (`results/gpu_results.csv`, `results/cpu_results.csv`).
    *   Properties of the GPU used are saved to `results/gpu_property.csv`.
*   **Graphical Visualization:** A Python script (`scripts/plot.py`) generates plots for execution time, speedup, and efficiency, saving them to the `plots/` directory.
*   **Support Scripts:**
    *   `compile.sh`: Compiles the source code.
    *   `run.sh`: Executes the compiled application and the plotting script.
    *   `compile_and_run.sh`: Compiles and runs the application and the plotting script.

## Project Structure

```
PHPC-Consegna1/
├── .gitignore
├── bin/                # Compiled executables
│   └── .gitkeep
├── plots/              # Generated plots
│   └── .gitkeep
├── results/            # CSV files with benchmark results
│   └── .gitkeep
├── scripts/            # Utility scripts
│   ├── compile_and_run.sh
│   ├── compile.sh
│   ├── plot.py
│   └── run.sh
├── src/                # Source code
│   ├── functions.cu
│   ├── functions.h
│   ├── main.cu
│   ├── utils.cu
│   └── utils.h
├── requirements.txt    # Python dependencies (Provided)
└── README.md           # This file
```

## Prerequisites

*   **CUDA Toolkit:** Version 9.0 or later (the compile script checks the version).
*   **C++ Compiler:** Compatible with C++11 standard (usually provided with the CUDA Toolkit).
*   **NVIDIA GPU:** Required for executing CUDA code.
*   **Python 3:** For the plot generation script.

### Python Virtual Environment

It is highly recommended to use a Python virtual environment to manage dependencies for the plotting script. Using the `requirements.txt` file present in the project root.

1.  **Create and activate the virtual environment:**
    Navigate to the root directory of the `PHPC-Consegna1` project in your terminal and run the following commands:
    ```bash
    # Create the virtual environment (e.g., named .venv)
    python3 -m venv .venv

    # Activate the virtual environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (Git Bash or WSL):
    source .venv/Scripts/activate
    #On Windows (Command Prompt):
    .venv\Scripts\activate.bat
    #On Windows (PowerShell):
    .venv\Scripts\Activate.ps1
    ```
    You should see the name of the virtual environment (e.g., `(.venv)`) in your terminal prompt.

2.  **Install dependencies:**
    Once the virtual environment is activated, install the required Python libraries from the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Deactivating the virtual environment:**
    When you're done working, you can deactivate the environment by simply typing:
    ```bash
    deactivate
    ```
    Remember to activate the environment again (`source .venv/bin/activate`) whenever you want to run the Python plotting script.

## Compilation

To compile the project, navigate to the `root` directory of `PHPC-Consegna1` and run:

```bash
./scripts/compile.sh
```

This script will check the CUDA Toolkit version, detect (if possible) the GPU's Compute Capability, and compile the source code, placing the `main_app` executable in the `bin/` directory.

## Execution

To compile and run the application (including plot generation) navigate to the `root` directory of `PHPC-Consegna1` and run :

```bash
./scripts/compile_and_run.sh
```

Alternatively, if the code is already compiled, to run only the application and plot generation:

```bash
./scripts/run.sh
```

The application will prompt the user to enter integration parameters and choose the `f(x)` function to use.

## Output

*   **Console:** Information about benchmark progress.
*   **CSV Files:**
    *   `results/gpu_property.csv`: Details about the GPU used.
    *   `results/gpu_results.csv`: Execution times, speedup, and efficiency for GPU implementations compared to the GPU baseline.
    *   `results/cpu_results.csv`: Execution times, speedup, and efficiency for GPU implementations compared to the CPU implementation.
*   **PNG Plots:**
    *   `plots/gpu_execution_time_vs_total_threads.png`
    *   `plots/gpu_speedup_vs_total_threads_from_csv.png`
    *   `plots/gpu_efficiency_vs_total_threads_from_csv.png`
    *   `plots/gpu_speedup_vs_cpu_sequential.png`
