# rcc_rocm_stress_test

This code was written to stress test local MI300x and more generally, MI series AMD GPUs - but to be portable and free of major library or system OS level dependencies.

# Dependencies
* ROCm 6.3.2 and above.
* hipcc, libpthreads, pthreads
* ROCm GEMM libs.
  
# Includes
* Includes necessary headers for HIP runtime (hip_runtime.h), hipBLAS (hipblas.h), and half/bfloat16 types (hip_fp16.h, hip_bfloat16.h).
* Multi-GPU concurrent testing
* A HBM check to determine the maximum amount of memory allocable
* Error Checking Macros: HIP_CHECK and HIPBLAS_CHECK are simple macros for checking the return status of HIP and hipBLAS API calls and exiting if an error occurs.
* time_gpu_event Function: A helper to calculate the elapsed time between two HIP events in seconds.
* copy_kernel: A very basic HIP kernel for device-to-device memory copy. While hipMemcpy is often optimized, a kernel can sometimes be used for specific access patterns or to keep the copy on the GPU explicitly. For this stress test, hipMemcpy for H2D/D2H and D2D (if peer access is enabled) is sufficient and often represents peak achievable bandwidth.

# main Function:
* Gets the number of available HIP devices using hipGetDeviceCount.
* Checks memory availabilty for maximum extents
* Iterates through each device using a for loop.
* hipSetDevice(dev) sets the current GPU for subsequent operations.
* hipGetDeviceProperties retrieves device information (like name).
* Memory Bandwidth Tests:
* Allocates host and device memory.
* Uses hipMemcpy with hipMemcpyHostToDevice, hipMemcpyDeviceToHost, and hipMemcpyDeviceToDevice (if peer access is possible) to perform data transfers.
* Uses hipEvent_t to record start and stop times on the GPU timeline for accurate measurement.
* Calculates bandwidth in GB/s.
* Includes a check and attempt to enable peer-to-peer access for D2D tests between devices.
* Frees allocated memory and destroys events.

# Floating Point Performance Tests (GEMM):
* Creates a hipBLAS handle using hipblasCreate.
* Defines matrix dimensions (m, n, k) and calculates the theoretical number of floating-point operations (FLOPs) for GEMM (C=αAB+βC). The number of FLOPs for a matrix multiplication of an m×k matrix by a k×n matrix is 2×m×n×k.
* Performs GEMM for FP64 (hipblasDgemm), FP32 (hipblasSgemm), FP16 (hipblasHgemm), and BF16 (hipblasBgemm). FP8 support is noted as potentially requiring newer library versions or custom kernels.
* Allocates device memory for input and output matrices.
* Uses hipEvent_t to time the kernel execution.
* Calculates GFLOPS.
* Frees allocated device memory and destroys events for each precision test.
* Destroys the hipBLAS handle using hipblasDestroy.
* Prints summary statistics for each device.

# How to compile

```hipcc rcc_rocm_stress_test.cpp -o rcc_rocm_stress_test -lhipblas -pthread -lpthread```

# How to run

* To run for 60 seconds on 8 GPUs:

```./rcc_rocm_stress_test 60 8```
