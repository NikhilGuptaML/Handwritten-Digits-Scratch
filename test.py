# test.py

import numpy as np
import dpnp as dp
import time

def generate_data_cpu():
    A = np.random.rand(60000, 728).astype(np.float32)
    B = np.random.rand(728, 128).astype(np.float32)
    return A, B

def generate_data_gpu():
    A = dp.random.rand(60000, 728).astype(dp.float32)
    B = dp.random.rand(728, 128).astype(dp.float32)
    return A, B

def cpu_matrix_multiplication(A, B):
    return np.dot(A, B)

def gpu_matrix_multiplication(A, B):
    return dp.dot(A, B)

def measure_time(func, A, B, warmup=False, runs=1000):
    if warmup:
        func(A, B)  # Warm-up run (not timed)

    times = []
    for _ in range(runs):
        start = time.time()
        func(A, B)
        end = time.time()
        times.append(end - start)

    total_time = sum(times) 
    return total_time

def validate_results(cpu_result, gpu_result):
    cpu_np = np.array(cpu_result)
    gpu_np = np.array(dp.asnumpy(gpu_result))
    return np.allclose(cpu_np, gpu_np, atol=1e-5)

def main():
    print("Generating CPU data...")
    A_cpu, B_cpu = generate_data_cpu()

    print("Generating GPU data...")
    A_gpu, B_gpu = generate_data_gpu()

    print("Benchmarking CPU...")
    cpu_time = measure_time(cpu_matrix_multiplication, A_cpu, B_cpu, runs=1000)

    print("Benchmarking GPU (with warm-up)...")
    gpu_time = measure_time(gpu_matrix_multiplication, A_gpu, B_gpu, warmup=True, runs=1000)

    print(f"\n✅ Average CPU time over 1000 runs: {cpu_time:.4f} seconds")
    print(f"🚀 Average GPU time over 1000 runs: {gpu_time:.4f} seconds")

    print("\nValidating results...")
    cpu_result = cpu_matrix_multiplication(A_cpu, B_cpu)
    gpu_result = gpu_matrix_multiplication(A_gpu, B_gpu)
    if validate_results(cpu_result, gpu_result):
        print("✅ CPU and GPU results match!")
    else:
        print("⚠️ Results differ between CPU and GPU!")

if __name__ == "__main__":
    main()