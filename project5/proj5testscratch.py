import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.vq import kmeans as sckm
import kmeans

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})
np.set_printoptions(suppress=True, precision=5)
import pandas as pd


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

a = np.array([120,5,200])
b = np.array([23,178,32])
a = a.reshape(1,a.size)
b = b.reshape(1,b.size)
euc_dist_a_b = np.linalg.norm(a-b)
euc_dist_b_a = np.linalg.norm(b-a)

manhat_dist_a_b = distance.cdist(a,b,metric='cityblock')

test_manhat_dist_0 = a-b
test_manhat_dist_1 = np.sum(np.abs(test_manhat_dist_0))

import cv2

trek_session_img_cv = cv2.imread('data/trekSession.jpg',3)
#cv im read reads in as bgr not rgb so fixing that
b,g,r = cv2.split(trek_session_img_cv)
trek_session_img_cv = cv2.merge([r,g,b])
trek_session_rescaled = cv2.resize(trek_session_img_cv,(int(np.round(trek_session_img_cv.shape[1]/4)),int(np.round(trek_session_img_cv.shape[0]/4))))
trek_session_rescaled_flat = flatten(trek_session_rescaled)
trek_session_kmeans = kmeans.KMeans(trek_session_rescaled_flat)
trek_session_kmeans.cluster(k=2,max_iter=1,distance_calc_method = 'L2')

bike_img_data = trek_session_kmeans.get_data()
bike_img_centroids = trek_session_kmeans.get_centroids()

print('done')





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