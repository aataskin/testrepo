# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

#from tensorflow.keras.mixed_precision import experimental as mixed_precision

#policy = mixed_precision.Policy('float16')
#mixed_precision.set_policy(policy)

def get_times(maximum_time):
    device_times = {
        "float16":[],
        "float32":[],
        "numpy":[]
    }

    matrix_sizes = [10, 100, 200, 500, 1000, 2000, 4000, 7000, 10000, 15000]

    for size in matrix_sizes:
        print(size)
        shape = (size, size)
        r1np = np.random.uniform(low=0, high=1, size=shape)
        r2np = np.random.uniform(low=0, high=1, size=shape)

        
        print("####### Calculating with numpy #######")
        start_time = time.time()
        dot_res = np.matmul(r1np, r2np)
        time_taken = time.time() - start_time
        print(time_taken)
        device_times["numpy"].append(time_taken)
        
        with tf.device("/gpu:0"):
            r1 = tf.convert_to_tensor(r1np, dtype=tf.float32)
            r2 = tf.convert_to_tensor(r2np, dtype=tf.float32)
        
            print("####### float32 #######")
            start_time = time.time()
            dot_operation = tf.matmul(r2, r1)

            time_taken = time.time() - start_time
            print(time_taken)
            device_times["float32"].append(time_taken)
        
        with tf.device("/gpu:0"):
            r1 = tf.convert_to_tensor(r1np, dtype=tf.float16)
            r2 = tf.convert_to_tensor(r2np, dtype=tf.float16)
        
            print("####### float16 #######")
            start_time = time.time()
            dot_operation = tf.matmul(r2, r1)

            time_taken = time.time() - start_time
            print(time_taken)
            device_times["float16"].append(time_taken)

    return device_times, matrix_sizes


def main():   
    device_times, matrix_sizes = get_times(60)
    print(device_times)
    float16_times=device_times["float16"][1:]
    float32_times = device_times["float32"][1:]
    numpy_times = device_times["numpy"][1:]

    plt.plot(matrix_sizes[1:], float16_times, 's-.')
    plt.plot(matrix_sizes[1:], float32_times, 'r-')
    plt.plot(matrix_sizes[1:], numpy_times, 'o-')
    plt.ylabel('Time, sec')
    plt.xlabel('Matrix size')
    plt.show()

    return

