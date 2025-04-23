#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib> // For atof and atoi
#include <algorithm> // For std::min
#include <thread> // For std::thread
#include <atomic> // Not strictly needed but good for concurrent ops
#include <mutex> // For printing synchronization
#include <exception> // For std::exception
#include <stdexcept> // For std::runtime_error
#include <cmath> // For pow

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

// Mutex for synchronizing output from multiple threads
std::mutex print_mutex;

// Helper function for timing between HIP events
double time_gpu_event(hipEvent_t start, hipEvent_t stop) {
    float milliseconds = 0;
    hipError_t status = hipEventElapsedTime(&milliseconds, start, stop);
    if (status != hipSuccess) {
        std::lock_guard<std::mutex> lock(print_mutex); // Protect print
        std::cerr << "HIP error in time_gpu_event: " << hipGetErrorString(status) << std::endl;
        // In a thread, consider a more graceful error handling
        throw std::runtime_error("HIP error in time_gpu_event");
    }
    return static_cast<double>(milliseconds) / 1000.0; // return time in seconds
}


// Macro to check HIP errors
#define HIP_CHECK(command)                                                                         \
    {                                                                                              \
        hipError_t status = command;                                                               \
        if (status != hipSuccess) {                                                                \
            std::lock_guard<std::mutex> lock(print_mutex); /* Protect print */                     \
            std::cerr << "HIP error: " << hipGetErrorString(status) << " at line " << __LINE__    \
                      << ": " << hipGetErrorString(status) << std::endl;                           \
            /* In a thread, consider a more graceful error handling than exit() */               \
            /* For this example, we'll throw an exception */                                     \
            throw std::runtime_error("HIP error");                                                 \
        }                                                                                          \
    }

// Macro to check hipBLAS errors
#define HIPBLAS_CHECK(command)                                                                     \
    {                                                                                              \
        hipblasStatus_t status = command;                                                          \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                                    \
            std::lock_guard<std::mutex> lock(print_mutex); /* Protect print */                     \
            std::cerr << "hipBLAS error: " << status << " at line " << __LINE__ << std::endl;      \
            /* Throw exception for hipBLAS errors as well */                                     \
            throw std::runtime_error("hipBLAS error");                                             \
        }                                                                                          \
    }


// Enum to specify precision
enum class TestPrecision { FP64, FP32 };

// Function to run the stress test on a single GPU for a specific precision
void run_gpu_stress(int device_id, double test_duration_sec, int m, int n, int k, TestPrecision precision) {
    hipblasHandle_t handle = nullptr; // Initialize handle to nullptr
    void *d_A = nullptr, *d_B = nullptr, *d_C = nullptr; // Initialize pointers to nullptr
    hipEvent_t start_event = nullptr, stop_event = nullptr; // Initialize events to nullptr
    std::string precision_str; // Declare precision_str outside the try block
    void* d_memory_stress = nullptr; // Pointer for the large memory stress allocation

    try {
        // Set the device for this thread
        HIP_CHECK(hipSetDevice(device_id));

        hipDeviceProp_t devProp;
        HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

        precision_str = (precision == TestPrecision::FP64) ? "FP64" : "FP32"; // Assign value here

        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "\nThread for Device " << device_id << " (" << precision_str << "): " << devProp.name << std::endl;
        }

        // --- Memory Stress: Allocate a large portion of HBM3 ---
        size_t free_mem, total_mem;
        HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));

        // Target 95% of free memory for allocation. Be cautious with the percentage.
        size_t memory_stress_size = static_cast<size_t>(free_mem * 0.95);
        size_t allocated_memory_stress_size = 0;


        if (memory_stress_size > 0) {
            hipError_t mem_alloc_status = hipMalloc(&d_memory_stress, memory_stress_size);
            if (mem_alloc_status == hipSuccess) {
                allocated_memory_stress_size = memory_stress_size;
                 {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "  Device " << device_id << " (" << precision_str << "): Allocated "
                              << allocated_memory_stress_size / (1024.0 * 1024.0 * 1024.0) << " GB for memory stress." << std::endl;
                }
                 // Optionally, write some data to the allocated memory to ensure it's resident
                 // HIP_CHECK(hipMemset(d_memory_stress, 0, allocated_memory_stress_size));
            } else {
                 std::lock_guard<std::mutex> lock(print_mutex);
                 std::cerr << "  Device " << device_id << " (" << precision_str << "): Warning: Could not allocate " << memory_stress_size
                           << " bytes (" << memory_stress_size / (1024.0 * 1024.0 * 1024.0) << " GB) for memory stress. "
                           << "HIP error: " << hipGetErrorString(mem_alloc_status) << ". Continuing without full memory stress." << std::endl;
                 d_memory_stress = nullptr; // Ensure pointer is null if allocation failed
            }
        } else {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cout << "  Device " << device_id << " (" << precision_str << "): No significant free memory to allocate for stress." << std::endl;
        }
        // --- End Memory Stress Allocation ---


        HIPBLAS_CHECK(hipblasCreate(&handle));

        size_t element_size;
        long long ops_per_single_gemm = 2LL * m * n * k; // Operations are the same regardless of type for GEMM structure

        if (precision == TestPrecision::FP64) {
            element_size = sizeof(double);
        } else { // FP32
            element_size = sizeof(float);
        }
        size_t matrix_size_bytes = (size_t)m * k * element_size;

        // Allocate device memory for GEMM matrices
        HIP_CHECK(hipMalloc(&d_A, matrix_size_bytes));
        HIP_CHECK(hipMalloc(&d_B, matrix_size_bytes));
        HIP_CHECK(hipMalloc(&d_C, matrix_size_bytes));

        // Initialize device memory (optional for pure stress)
        HIP_CHECK(hipMemset(d_A, 0, matrix_size_bytes));
        HIP_CHECK(hipMemset(d_B, 0, matrix_size_bytes));
        HIP_CHECK(hipMemset(d_C, 0, matrix_size_bytes));

        HIP_CHECK(hipEventCreate(&start_event));
        HIP_CHECK(hipEventCreate(&stop_event));

        int gemm_count = 0;
        double accumulated_device_time = 0.0;

        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "  Device " << device_id << " (" << precision_str << "): Starting timed GEMM execution..." << std::endl;
        }

        // Run GEMM repeatedly, accumulating device time
        while (accumulated_device_time < test_duration_sec) {
            // Record start event before GEMM
            HIP_CHECK(hipEventRecord(start_event, 0));

            if (precision == TestPrecision::FP64) {
                double alpha = 1.0, beta = 0.0;
                HIPBLAS_CHECK(hipblasDgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, (const double*)d_A, m, (const double*)d_B, k, &beta, (double*)d_C, m));
            } else { // FP32
                float alpha = 1.0f, beta = 0.0f;
                HIPBLAS_CHECK(hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, (const float*)d_A, m, (const float*)d_B, k, &beta, (float*)d_C, m));
            }

            // Record stop event after GEMM
            HIP_CHECK(hipEventRecord(stop_event, 0));

            // Synchronize on the stop event and measure the time for this single GEMM
            HIP_CHECK(hipEventSynchronize(stop_event));
            double single_gemm_time = time_gpu_event(start_event, stop_event);

            accumulated_device_time += single_gemm_time; // Accumulate device time
            gemm_count++;

            // Optional: Small sleep to yield CPU time if needed
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        // Calculate total operations and GFLOPS using accumulated_device_time
        long long total_ops = ops_per_single_gemm * gemm_count;
        double performance = (double)total_ops / (accumulated_device_time * 1e9); // GFLOPS

        {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cout << "  Device " << device_id << " (" << precision_str << "): Completed " << gemm_count << " GEMM operations." << std::endl;
            std::cout << "  Device " << device_id << " (" << precision_str << "): Accumulated Device Time: " << accumulated_device_time << " seconds" << std::endl;
            std::cout << "  Device " << device_id << " (" << precision_str << "): Performance: " << performance << " GFLOPS" << std::endl;
        }

    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "Error in thread for Device " << device_id << " (" << precision_str << "): " << e.what() << std::endl;
    } catch (...) {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cerr << "Unknown error in thread for Device " << device_id << " (" << precision_str << ")" << std::endl;
    }

    // Clean up resources in a way that handles potential exceptions
    if (start_event) {
        hipError_t err = hipEventDestroy(start_event);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error destroying start_event for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }
    if (stop_event) {
         hipError_t err = hipEventDestroy(stop_event);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error destroying stop_event for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }
    if (d_A) {
        hipError_t err = hipFree(d_A);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error freeing d_A for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }
    if (d_B) {
         hipError_t err = hipFree(d_B);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error freeing d_B for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }
    if (d_C) {
         hipError_t err = hipFree(d_C);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error freeing d_C for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }
    if (handle) {
        hipblasStatus_t status = hipblasDestroy(handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            std::lock_guard<std::mutex> lock(print_mutex);
            std::cerr << "Error destroying hipBLAS handle for Device " << device_id << " (" << precision_str << "): " << status << std::endl;
        }
    }

    // --- Memory Stress Cleanup: Free the large allocated memory ---
    if (d_memory_stress) {
         hipError_t err = hipFree(d_memory_stress);
        if (err != hipSuccess) {
             std::lock_guard<std::mutex> lock(print_mutex);
             std::cerr << "Error freeing memory_stress_buffer for Device " << device_id << " (" << precision_str << "): " << hipGetErrorString(err) << std::endl;
        }
    }


    {
        std::lock_guard<std::mutex> lock(print_mutex);
        std::cout << "Thread for Device " << device_id << " (" << precision_str << ") finished." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    double test_duration_sec = 10.0; // Default duration per GPU per precision cycle
    int num_gpus_to_target = -1;    // Default to all available GPUs

    if (argc > 1) {
        test_duration_sec = std::atof(argv[1]);
        if (test_duration_sec <= 0) {
            std::cerr << "Error: Invalid test duration. Please provide a positive value in seconds." << std::endl;
            return -1;
        }
    }

    if (argc > 2) {
        num_gpus_to_target = std::atoi(argv[2]);
        if (num_gpus_to_target < -1 || num_gpus_to_target == 0) {
            std::cerr << "Error: Invalid number of GPUs to target. Use -1 for all or a positive integer." << std::endl;
            return -1;
        }
    }

    std::cout << "Welcome to UQ RCC's RoCM HIP AMD MI Stress Tester. Please use at your own risk" << std::endl;


    int deviceCount = 0; // Initialize deviceCount
    hipError_t hip_get_device_count_status = hipGetDeviceCount(&deviceCount);
    if (hip_get_device_count_status != hipSuccess) {
         std::cerr << "Error getting HIP device count: " << hipGetErrorString(hip_get_device_count_status) << std::endl;
         return -1;
    }


    if (deviceCount < 1) {
        std::cerr << "No HIP-enabled devices found!" << std::endl;
        return -1;
    }

    int actual_gpus_to_test = deviceCount;
    if (num_gpus_to_target > 0) {
        actual_gpus_to_test = std::min(deviceCount, num_gpus_to_target);
        std::cout << "Targeting " << actual_gpus_to_test << " GPU(s) for the test." << std::endl;
    } else {
        std::cout << "Targeting all " << actual_gpus_to_test << " GPU(s) for the test." << std::endl;
    }

    if (actual_gpus_to_test == 0) {
         std::cerr << "No GPUs selected for testing." << std::endl;
         return -1;
    }

    // Parameters for GEMM tests
    int matrix_dim = 16384; // Using 16384x16384 matrices
    int m = matrix_dim;
    int n = matrix_dim;
    int k = matrix_dim;

    // Number of floating-point operations for one GEMM call (same for FP64 and FP32)
    long long ops_per_single_gemm = 2LL * m * n * k;

    // --- FP64 Testing Cycle ---
    std::cout << "\n--- Starting FP64 Testing Cycle (GEMM " << m << "x" << k << " * " << k << "x" << n << ") ---" << std::endl;
    std::cout << "Total operations per single GEMM: " << ops_per_single_gemm << std::endl;
    std::cout << "Target test duration per GPU: " << test_duration_sec << " seconds" << std::endl;

    std::vector<std::thread> fp64_threads;
    for (int dev = 0; dev < actual_gpus_to_test; ++dev) {
        fp64_threads.push_back(std::thread(run_gpu_stress, dev, test_duration_sec, m, n, k, TestPrecision::FP64));
    }

    // Join threads to wait for all FP64 tests to complete
    for (std::thread& thread : fp64_threads) {
        if (thread.joinable()) { // Check if thread is joinable before joining
            thread.join();
        }
    }
    std::cout << "\n--- FP64 Testing Cycle Finished ---" << std::endl;

    // --- FP32 Testing Cycle ---
    std::cout << "\n--- Starting FP32 Testing Cycle (GEMM " << m << "x" << k << " * " << k << "x" << n << ") ---" << std::endl;
    std::cout << "Total operations per single GEMM: " << ops_per_single_gemm << std::endl;
    std::cout << "Target test duration per GPU: " << test_duration_sec << " seconds" << std::endl;


    std::vector<std::thread> fp32_threads;
    for (int dev = 0; dev < actual_gpus_to_test; ++dev) {
        fp32_threads.push_back(std::thread(run_gpu_stress, dev, test_duration_sec, m, n, k, TestPrecision::FP32));
    }

    // Join threads to wait for all FP32 tests to complete
    for (std::thread& thread : fp32_threads) {
         if (thread.joinable()) { // Check if thread is joinable before joining
            thread.join();
        }
    }
    std::cout << "\n--- FP32 Testing Cycle Finished ---" << std::endl;


    std::cout << "\n--- Concurrent Stress Test Finished ---" << std::endl;

    return 0;
}
