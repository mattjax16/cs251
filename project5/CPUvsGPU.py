
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import kmeans
import kmeansGPU
import cupy as cp
import cv2 as cv
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})
np.set_printoptions(suppress=True, precision=5)
import pandas as pd
import time



def flatten(img):
    '''Flattens `img` to N 1D vectors.
    For example, for an RGB image, `(num_rows, num_cols, rgb)` -> `(num_rows*num_cols, rgb)`.

    Parameters:
    -----------
    img: ndarray. shape=(num_rows, num_cols, rgb)

    Returns:
    -----------
    Flattened `img`. ndarray. shape=(num_rows*num_cols, rgb)
    '''
    flatten_img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    return flatten_img


intense_m29_img = plt.imread('https://p.vitalmtb.com/photos/products/28620/photos/54903/s1600_photo_734056.jpg?1578283962', 'jpg')
flatten_intense_m29 = flatten(intense_m29_img)
m29_cluster_gpu = kmeansGPU.KMeansGPU(flatten_intense_m29)
m29_cluster = kmeansGPU.KMeansGPU(flatten_intense_m29, use_gpu=False)

#L2 GPU m29 bike
# m29_cluster_gpu.cluster_batch(25,n_iter=4,init_method='points')
m29_cluster_gpu.cluster_batch(25,n_iter=4,init_method='++')
#L2 CPU cpu m29 bike
m29_cluster.cluster_batch(25,n_iter=4)
