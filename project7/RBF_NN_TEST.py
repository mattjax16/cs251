from rbf_net import RBF_Net
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import palettable
import concurrent.futures
import pandas as pd
import cupy as cp
from kmeansGPU import KMeansGPU as KMeans

np.random.seed(1)
np.set_printoptions(suppress=True, precision=5)

# N = 3 samples, M = 5 features
test_data = np.random.normal(size=(3, 5))
# 4 hidden units / prototypes, each has 5 features
test_centroids = np.random.normal(size=(4, 5))
# Each sample assigned to one of 4 hidden unit prototypes
test_sigmas = np.random.uniform(size=(4,))
test_wts = 2*np.random.uniform(size=(4+1, 3)) - 1

test_net = RBF_Net(4, 3, use_gpu=False)
test_net.prototypes = test_centroids
test_net.sigmas = test_sigmas
test_net.wts = test_wts
test_h_act = test_net.hidden_act(test_data)
print(f'Your hidden layer activation is\n{test_h_act.astype("float16")}\n\nand should be')
print('[[0.      0.      0.00009 0.00033]\n [0.00013 0.      0.00004 0.00014]\n [0.      0.      0.      0.00001]]')


test_net_gpu = RBF_Net(4, 3)
test_net_gpu.prototypes = test_centroids
test_net_gpu.sigmas = test_sigmas
test_net_gpu.wts = test_wts
test_h_act_gpu = test_net_gpu.hidden_act(test_data)
print(f'\n\nYour hidden layer activation is\n{test_h_act_gpu}\n\nand should be')
print('[[0.      0.      0.00009 0.00033]\n [0.00013 0.      0.00004 0.00014]\n [0.      0.      0.      0.00001]]')

test_out_act = test_net.output_act(test_h_act)
print(f'\nYour output layer activation is\n{test_out_act}\n\nand should be')
print('[[-0.72136  0.61505 -0.20481]\n [-0.72151  0.61487 -0.20466]\n [-0.72144  0.61479 -0.20465]]')





df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
iris = df[['sepal_length', 'petal_width']].to_numpy()

num_hidden_units = 1
num_classes = 4
net = RBF_Net(num_classes=num_classes, num_hidden_units=num_hidden_units, use_gpu=False)
iris_x = np.reshape(iris[:, 0], [len(iris), 1])
iris_y = np.reshape(iris[:, 1], [len(iris), 1])
iris_c = net.linear_regression(iris_x, iris_y)

line_x = np.linspace(iris_x.min(), iris_x.max())
line_y = line_x * iris_c[0] + iris_c[1]


plt.scatter(iris_x, iris_y)
plt.plot(line_x, line_y)
plt.title('Iris — Linear Regression test')
plt.xlabel('sepal_length')
plt.ylabel('petal_width')
plt.show()



# np.random.seed(0)
#
# # N = 10 samples, M = 5 features
# test_data = np.random.normal(size=(10, 5))
# # 4 hidden units / prototypes, each has 5 features
# test_centroids = np.random.normal(size=(4, 5))
# # Each sample assigned to one of 4 hidden unit prototypes
# test_assignments = np.random.randint(low=0, high=4, size=(10,))
# kmeansObj = KMeans(use_gpu=True)
#
# test_net = RBF_Net(4, 3)
# print(f'Number of hidden units in your net is {test_net.get_num_hidden_units()} and should be 4')
# print(f'Number of output units in your net is {test_net.get_num_output_units()} and should be 3')
# test_clust_mean_dists = test_net.avg_cluster_dist(test_data, test_centroids, test_assignments, kmeansObj)
#
# print(f'Your avg within cluster distances are\n{test_clust_mean_dists} and should be\n[2.23811 3.94891 3.12267 3.4321]')
#
# test_net.initialize(test_data)
#
# print(f'Your prototypes have shape {test_net.get_prototypes().shape} and the shape should be (4, 5).')
# print(f'Your hidden unit sigmas have shape {test_net.sigmas.shape} and the shape should be (4,).')
