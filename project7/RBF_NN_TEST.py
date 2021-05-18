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
from rbf_net import RBF_Reg_Net
# # N = 3 samples, M = 5 features
# test_data = np.random.normal(size=(3, 5))
# # 4 hidden units / prototypes, each has 5 features
# test_centroids = np.random.normal(size=(4, 5))
# # Each sample assigned to one of 4 hidden unit prototypes
# test_sigmas = np.random.uniform(size=(4,))
# test_wts = 2*np.random.uniform(size=(4+1, 3)) - 1
#
# test_net = RBF_Net(4, 3, use_gpu=False)
# test_net.prototypes = test_centroids
# test_net.sigmas = test_sigmas
# test_net.wts = test_wts
# test_h_act = test_net.hidden_act(test_data)
# print(f'Your hidden layer activation is\n{test_h_act.astype("float16")}\n\nand should be')
# print('[[0.      0.      0.00009 0.00033]\n [0.00013 0.      0.00004 0.00014]\n [0.      0.      0.      0.00001]]')
#
#
# test_net_gpu = RBF_Net(4, 3)
# test_net_gpu.prototypes = test_centroids
# test_net_gpu.sigmas = test_sigmas
# test_net_gpu.wts = test_wts
# test_h_act_gpu = test_net_gpu.hidden_act(test_data)
# print(f'\n\nYour hidden layer activation is\n{test_h_act_gpu}\n\nand should be')
# print('[[0.      0.      0.00009 0.00033]\n [0.00013 0.      0.00004 0.00014]\n [0.      0.      0.      0.00001]]')
#
# test_out_act = test_net.output_act(test_h_act)
# print(f'\nYour output layer activation is\n{test_out_act}\n\nand should be')
# print('[[-0.72136  0.61505 -0.20481]\n [-0.72151  0.61487 -0.20466]\n [-0.72144  0.61479 -0.20465]]')

def load_dev_ds(filename):
    df = pd.read_csv(filename)
    x = df[['X', 'Y']].to_numpy()
    y = df['class'].to_numpy()
    return x, y


def normalize_features(x, mins, maxs):
    return (x - mins) / (maxs - mins)


rbf_regression_x = np.load('data/rbf_regression_x.npy')
rbf_regression_y = np.load('data/rbf_regression_y.npy')
# rbf_regression_y = rbf_regression_y.reshape(rbf_regression_y.size)




num_hidden_units = 50
num_classes = 1


rbf_regression_net = RBF_Reg_Net(num_hidden_units=num_hidden_units,num_classes=num_classes)
rbf_regression_net.train(rbf_regression_x,rbf_regression_y)



# rbf_dev_train, y_train = load_dev_ds('data/rbf_dev_train.csv')
# rbf_dev_test, y_test = load_dev_ds('data/rbf_dev_test.csv')
#
#
#
# train_min = np.min(rbf_dev_train, axis=0)
# train_max = np.max(rbf_dev_train, axis=0)
# rbf_dev_train = normalize_features(rbf_dev_train, train_min, train_max)
# rbf_dev_test = normalize_features(rbf_dev_test, train_min, train_max)
# # Load data here
# x_train = np.load('data/mnist_train_data.npy')
# y_train = np.load('data/mnist_train_labels.npy')
# x_test = np.load('data/mnist_test_data.npy')
# y_test = np.load('data/mnist_test_labels.npy')
# x_train = np.reshape(x_train, [len(x_train), -1])
# x_test = np.reshape(x_test, [len(x_test), -1])
#
# x_train = x_train/255
# x_test = x_test/255
#
#
# num_hidden_units = 20
# num_classes = 10
# n = 50
#
# mnist_net_gpu = RBF_Net(num_classes=num_classes, num_hidden_units=num_hidden_units,use_gpu=True)
# mnist_net_gpu.train(x_train[:n], y_train[:n])
#
# # test acc
# mnist_test_y_pred_gpu = mnist_net_gpu.predict(x_test)
# accuracy_test_gpu = mnist_net_gpu.accuracy(mnist_test_y_pred_gpu, y_test)
#
# print(accuracy_test_gpu)
#
#
# # Load data here
# x_train = np.load('data/mnist_train_data.npy')
# y_train = np.load('data/mnist_train_labels.npy')
# x_test = np.load('data/mnist_test_data.npy')
# y_test = np.load('data/mnist_test_labels.npy')
# x_train = np.reshape(x_train, [len(x_train), -1])
# x_test = np.reshape(x_test, [len(x_test), -1])
#
# x_train = x_train/255
# x_test = x_test/255
#
#
# num_hidden_units = 20
# num_classes = 10
# n = 50
#
#
#
# num_hidden_units = 150
# num_classes = 10
# n = 500
#
# mnist_net = RBF_Net(num_classes=num_classes, num_hidden_units=num_hidden_units)
# mnist_net.train(x_train[:n], y_train[:n])
#
# # test acc
# mnist_test_y_pred = mnist_net.predict(x_test)
# accuracy_test_cpu = mnist_net.accuracy(mnist_test_y_pred, y_test)
# print(accuracy_test_cpu)

# num_hidden_units = 10
# num_classes = 2
#
# net = RBF_Net(num_classes=num_classes, num_hidden_units=num_hidden_units)
# net.train(rbf_dev_train, y_train)
#
# # Training set accuracy
# y_pred = net.predict(rbf_dev_train)
# acc1 = net.accuracy(y_train, y_pred)
#
#
# # Test set accuracy
# y_pred_test = net.predict(rbf_dev_test)
# acc2 = net.accuracy(y_test, y_pred_test)
#
#
#
# rbf_dev_train.shape
#
#
# df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
# iris = df[['sepal_length', 'petal_width']].to_numpy()
#
# num_hidden_units = 1
# num_classes = 4
# net = RBF_Net(num_classes=num_classes, num_hidden_units=num_hidden_units, use_gpu=False)
# iris_x = np.reshape(iris[:, 0], [len(iris), 1])
# iris_y = np.reshape(iris[:, 1], [len(iris), 1])
# iris_c = net.linear_regression(iris_x, iris_y)
#
# line_x = np.linspace(iris_x.min(), iris_x.max())
# line_y = line_x * iris_c[0] + iris_c[1]
#
#
# plt.scatter(iris_x, iris_y)
# plt.plot(line_x, line_y)
# plt.title('Iris — Linear Regression test')
# plt.xlabel('sepal_length')
# plt.ylabel('petal_width')
# plt.show()



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
