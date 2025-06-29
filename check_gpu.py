#!/usr/bin/env python3
"""
GPU Check Script for AC-GAN Project
Checks GPU availability and TensorFlow configuration
"""

import tensorflow as tf
import numpy as np

def check_gpu_status():
    """Comprehensive GPU status check"""
    print("=" * 50)
    print("🔍 GPU STATUS CHECK")
    print("=" * 50)
    
    # TensorFlow version
    print(f"📱 TensorFlow Version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"🎮 Physical GPUs Available: {len(gpus)}")
    
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            
        # Check GPU memory
        try:
            gpu_details = tf.config.experimental.get_device_details(gpus[0])
            print(f"   GPU Details: {gpu_details}")
        except:
            print("   GPU details not available")
    
    # Check logical devices
    logical_devices = tf.config.list_logical_devices()
    print(f"🔧 Logical Devices: {len(logical_devices)}")
    for device in logical_devices:
        print(f"   {device.device_type}: {device.name}")
    
    # CUDA availability
    print(f"🚀 CUDA Available: {tf.test.is_built_with_cuda()}")
    print(f"⚡ GPU Support: {tf.test.is_gpu_available()}")
    
    # Test GPU computation
    if gpus:
        print("\n🧪 Testing GPU Computation...")
        try:
            with tf.device('/GPU:0'):
                # Simple matrix multiplication test
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                print("✅ GPU computation test successful!")
                print(f"   Result shape: {c.shape}")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    
    # Memory growth test
    if gpus:
        print("\n💾 Setting up memory growth...")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory growth enabled successfully!")
        except Exception as e:
            print(f"❌ Memory growth setup failed: {e}")
    
    # Mixed precision check
    print("\n🎯 Mixed Precision Support...")
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        print(f"✅ Mixed precision policy available: {policy.name}")
    except Exception as e:
        print(f"❌ Mixed precision not available: {e}")
    
    print("\n" + "=" * 50)
    
    # Recommendations
    if not gpus:
        print("💡 RECOMMENDATIONS:")
        print("   - Install CUDA toolkit if you have NVIDIA GPU")
        print("   - Install cuDNN library")
        print("   - Install tensorflow-gpu or ensure TensorFlow has GPU support")
        print("   - Check NVIDIA driver compatibility")
    else:
        print("🎉 GPU SETUP LOOKS GOOD!")
        print("💡 TIPS:")
        print("   - Use larger batch sizes for better GPU utilization")
        print("   - Enable mixed precision for faster training")
        print("   - Monitor GPU memory usage during training")
    
    print("=" * 50)

def benchmark_gpu_vs_cpu():
    """Simple benchmark comparing GPU vs CPU performance"""
    print("\n🏁 PERFORMANCE BENCHMARK")
    print("=" * 30)
    
    # Test parameters
    matrix_size = 2000
    iterations = 5
    
    # CPU benchmark
    print("🖥️  CPU Benchmark...")
    with tf.device('/CPU:0'):
        start_time = tf.timestamp()
        for i in range(iterations):
            a = tf.random.normal([matrix_size, matrix_size])
            b = tf.random.normal([matrix_size, matrix_size])
            c = tf.matmul(a, b)
        cpu_time = tf.timestamp() - start_time
        print(f"   CPU Time: {cpu_time:.2f} seconds")
    
    # GPU benchmark (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("🎮 GPU Benchmark...")
        with tf.device('/GPU:0'):
            start_time = tf.timestamp()
            for i in range(iterations):
                a = tf.random.normal([matrix_size, matrix_size])
                b = tf.random.normal([matrix_size, matrix_size])
                c = tf.matmul(a, b)
            gpu_time = tf.timestamp() - start_time
            print(f"   GPU Time: {gpu_time:.2f} seconds")
            
            speedup = cpu_time / gpu_time
            print(f"🚀 GPU Speedup: {speedup:.2f}x faster than CPU")
    else:
        print("⚠️  No GPU available for benchmark")

if __name__ == "__main__":
    check_gpu_status()
    benchmark_gpu_vs_cpu()
