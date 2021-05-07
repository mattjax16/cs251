import concurrent.futures
import time
import numpy as np
import tensorflow as tf
import cupy as cp
import time
# start = time.perf_counter()
#
#
# def do_something(seconds):
#     print(f'Sleeping {seconds} second(s)...')
#     time.sleep(seconds)
#     return f'Done Sleeping...{seconds}'
#
#
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     secs = np.arange(30)+1
#     results = executor.map(do_something, secs)
#
#     for result in results:
#         print(result)
#
# finish = time.perf_counter()
#
# print(f'Finished in {round(finish-start, 2)} second(s)')

# gpu_devices = tf.config.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# tf.compat.v1.InteractiveSession(config=config)
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ### Numpy and CPU
# s = time.time()
# x_cpu = np.ones((1920,1080,3))
# e = time.time()
# print(e - s)
#
# # ### CuPy and GPU
# s = time.time()
# x_gpu = cp.ones((1920,1080,3))
# e = time.time()
# print(e - s)
