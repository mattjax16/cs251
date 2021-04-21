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

super_simple = pd.read_csv('data/super_simple.csv')
super_simple = super_simple.values

cluster = kmeans.KMeans(super_simple)

test_sk = sckm(super_simple,3,1000)


# a = np.array([1, 2, 3, 4])
# b = 4*a
# print(f'Your pt-to-pt distance is {cluster.dist_pt_to_pt(a, b)}')
# print(f'Correct pt-to-pt distance is {np.linalg.norm(a-b)}')


# test_pt = np.array([[1, 2]])
# test_centroids = np.array([[9, 9], [11, 11], [0, 0]])
# print(f'Your pt-to-centroids distance is {cluster.dist_pt_to_centroids(test_pt.flatten(), test_centroids)}')
# print(f'Correct pt-to-centroids distance is {distance.cdist(test_pt, test_centroids)[0]}')


test_k = 3
init_centroids = cluster.initialize(test_k)
print(f'Initial cluster centroids shape is:\n{init_centroids.shape} and should be (3, 2)')


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
np.random.seed(0)
cluster.cluster(k = 3)
cluster.plot_clusters()
plt.show()



cluster.elbow_plot(30,title = f'Number of Ks for Simple Data\nEffect on Inertia')
plt.show()



cluster.elbow_plot(8,title = f'Number of Ks for Simple Data\nEffect on Inertia')
plt.show()