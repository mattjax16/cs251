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
import kmeansCupyEx

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
flatten_intense_m29_f32 = np.float32(flatten_intense_m29)

blue_bird_img = plt.imread('data/baby_bird.jpeg')
blue_bird_flattened = flatten(blue_bird_img)
blue_bird_flattened_32 = np.float32(blue_bird_flattened)

## CV
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
retCv, labelCV, centerCV = cv2.kmeans(blue_bird_flattened_32,6, None, criteria, 10,cv2.KMEANS_RANDOM_CENTERS)



m29_cluster = kmeans.KMeans(flatten_intense_m29)


# s = time.time()
# m29_cluster.cluster_batch(k = 5, n_iter=1, max_iter=5)
# e = time.time()
# print(e - s)
#
#
m29_cluster_gpu = kmeansGPU.KMeansGPU(flatten_intense_m29)
# s = time.time()
# m29_cluster_gpu.cluster_batch(k = 5, n_iter=1, max_iter=5)
# e = time.time()
# print(e - s)


m29_gpu_data = m29_cluster_gpu.get_data()
m29_gpu_centroids = m29_cluster_gpu.initialize(2, init_method='++')
# m29_gpu_labels = m29_cluster_gpu.update_labels(m29_gpu_centroids)
# m29_gpu_centroids, m29_gpu_centroid_dif = m29_cluster_gpu.update_centroids(2,m29_gpu_labels,m29_gpu_centroids)


five_blobs = pd.read_csv('data/five_blobs.csv')
five_blobs_cluster_gpu = kmeansGPU.KMeansGPU(five_blobs.values)
five_blobs_data = five_blobs_cluster_gpu.get_data()
five_blobs_centroids = five_blobs_cluster_gpu.initialize(5, init_method='++')

t_cp_centers, t_cp_pred = kmeansCupyEx.fit_custom(five_blobs_data,5,40)

'''Kernal Testing'''
l2norm_kernel = cp.ReductionKernel(
'T x',  # input params
'T y',  # output params
'x * x',  # map
'a + b',  # reduce
'y = sqrt(a)',  # post-reduction map
'0',  # identity value
'l2norm'  # kernel name
)


var_kernel = cp.ElementwiseKernel(
    'T x0, T x1, T c0, T c1', 'T out',
    'out = (x0 - c0) * (x0 - c0) + (x1 - c1) * (x1 - c1)',
    'var_kernel'
)
sum_kernel = cp.ReductionKernel(
    'T x, S mask', 'T out',
    'mask ? x : 0',
    'a + b', 'out = a', '0',
    'sum_kernel'
)
count_kernel = cp.ReductionKernel(
    'T mask', 'float32 out',
    'mask ? 1.0 : 0.0',
    'a + b', 'out = a', '0.0',
    'count_kernel'
)

fit_calc_distances = cp.ElementwiseKernel(
    'S data , raw S centers, int32 n_clusters, int32 dim', 'raw S dist',
    '''
    for (int j = 0; j < n_clusters; j++){
        int cent_ind[] = {j , i % dim};
        int dist_ind[] = {i/dim,j};
        double diff = centers[cent_ind] - data;
        atomicAdd(&dist[dist_ind],diff * diff);
    }
    ''',
    'calc_distances'
)

x = cp.arange(10, dtype='f').reshape(5, 2)

zzz_dist = cp.zeros((five_blobs_data.shape[0], 5), dtype ='int32')
l2_kernal_res = fit_calc_distances(cp.asarray(cp.int32(five_blobs_data)),cp.asarray(cp.int32(five_blobs_centroids))
                                   ,cp.int32(5),cp.int32(2), zzz_dist)






zzz_dist2 = cp.zeros((m29_gpu_data.shape[0], 5), dtype ='int32')
l2_kernal_res3 = fit_calc_distances(cp.asarray(cp.int32(m29_gpu_data)),cp.asarray(cp.int32(m29_gpu_centroids))
                                   ,cp.int32(5),cp.int32(3), zzz_dist2)




zzz_dist3 = cp.zeros((m29_gpu_data.shape[0], 5), dtype ='float64')

test_x_1 = m29_gpu_data[:,None,:]
test_x_2 = m29_gpu_centroids[None,:,:]

final_x_2 = test_x_1 - test_x_2
final_x_2 = cp.asarray(final_x_2)
zzz_dist3 = l2norm_kernel(final_x_2,axis = 2)


print('Done Testing Cupy Kernals')




# blue_bird_cluster = kmeans.KMeans(blue_bird_flattened)
#
# s = time.time()
# blue_bird_cluster.cluster_batch(k = 15, n_iter=2)
# e = time.time()
# print(e - s)
#
#
# blue_bird_cluster_gpu = kmeansGPU.KMeansGPU(np.array(blue_bird_flattened))
# s = time.time()
# blue_bird_cluster_gpu.cluster_batch(k = 15, n_iter=2)
# e = time.time()
# print(e - s)
#
# print('done')


# a = np.array([120,5,200])
# b = np.array([23,178,32])
# a = a.reshape(1,a.size)
# b = b.reshape(1,b.size)
# euc_dist_a_b = np.linalg.norm(a-b)
# euc_dist_b_a = np.linalg.norm(b-a)
#
# manhat_dist_a_b = distance.cdist(a,b,metric='cityblock')
#
# test_manhat_dist_0 = a-b
# test_manhat_dist_1 = np.sum(np.abs(test_manhat_dist_0))
#
# import cv2
#
# trek_session_img_cv = cv2.imread('data/trekSession.jpg',3)
# #cv im read reads in as bgr not rgb so fixing that
# b,g,r = cv2.split(trek_session_img_cv)
# trek_session_img_cv = cv2.merge([r,g,b])
# trek_session_rescaled = cv2.resize(trek_session_img_cv,(int(np.round(trek_session_img_cv.shape[1]/4)),int(np.round(trek_session_img_cv.shape[0]/4))))
# trek_session_rescaled_flat = flatten(trek_session_rescaled)
# trek_session_kmeans = kmeans.KMeans(trek_session_rescaled_flat)
# trek_session_kmeans.cluster(k=2,max_iter=1,distance_calc_method = 'L2')
#
# bike_img_data = trek_session_kmeans.get_data()
# bike_img_centroids = trek_session_kmeans.get_centroids()
#
# print('done')





# super_simple = pd.read_csv('data/super_simple.csv')
# super_simple = super_simple.values
#
# cluster = kmeans.KMeans(super_simple)
#
# # test_sk = sckm(super_simple,3,1000)
#
#
# a = np.array([1, 2, 3, 4])
# b = 4*a
# print(f'Your pt-to-pt distance is {cluster.dist_pt_to_pt(a, b)}')
# print(f'Correct pt-to-pt distance is {np.linalg.norm(a-b)}')
#
#
# test_pt = np.array([[1, 2]])
# test_centroids = np.array([[9, 9], [11, 11], [0, 0]])
# print(f'Your pt-to-centroids distance is {cluster.dist_pt_to_centroids(test_pt.flatten(), test_centroids)}')
# print(f'Correct pt-to-centroids distance is {distance.cdist(test_pt, test_centroids)[0]}')
#
#
# test_k = 3
# init_centroids = cluster.initialize(test_k)
# print(f'Initial cluster centroids shape is:\n{init_centroids.shape} and should be (3, 2)')
#
#
# # Consistently set initial centroids for test
# init_centroids = np.array([[ 0.338, 4.4672], [-1.8401, 3.1123], [1.7931, 0.5427]])
#
# new_labels = cluster.update_labels(init_centroids)
# print(f'After the first update data label step, 1st 10 of your cluster assignments are:\n{new_labels[:10]}')
# print('Your 1st 10 cluster assignments should be:\n[0 1 1 1 2 0 2 1 2 1]')
#
# new_centroids, diff_from_prev_centroids = cluster.update_centroids(test_k, new_labels, init_centroids)
# print(f'After the first centroid update, your cluster assignments are:\n{new_centroids}')
# print(f'Your difference from previous centroids:\n{diff_from_prev_centroids}')
# # np.random.seed(0)
# # cluster.cluster(k = 3)
# # cluster.plot_clusters()
# # plt.show()
# #
# #
# #
# # cluster.elbow_plot(30,title = f'Number of Ks for Simple Data\nEffect on Inertia')
# # plt.show()
# #
# #
# #
# # cluster.elbow_plot(8,title = f'Number of Ks for Simple Data\nEffect on Inertia')
# # plt.show()