import os
import random
import numpy as np
import matplotlib.pyplot as plt
import data
import scipy
import linear_regression

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)


day_bike_data = data.Data('data/updates_bikes_day.csv')
bike_headers = day_bike_data.get_headers()
print(bike_headers)

numeric_headers = bike_headers[-7:]
categorical_headers = bike_headers[:8]

day_linear_reg = linear_regression.LinearRegression(day_bike_data)
day_linear_reg.pair_plot((numeric_headers[:-3]+[numeric_headers[-1]]))
day_linear_reg.show()

import transformation
import copy

# normalized_day_bike_data = copy.copy(day_bike_data)
# day_bike_transform = transformation.Transformation(normalized_day_bike_data)
#
# normalized_day_bike_array = day_bike_transform.normalize_separately()
#
#
# def step_wise_linear_regression(dataObj, ind_vars=None, dep_var=None, method='normal',
#                                 R2_list=[], R2_adjusted_list=[], ind_vars_used_ordered=[]):
#
#
#     # print(f'\nInd_Vars: {ind_vars}')
#
#     # could jsut pass in the linear regression object would be more efficiet
#     # but am making a new one for each recursion (run) of this function instead
#     lin_reg = linear_regression.LinearRegression(dataObj)
#
#     if isinstance(ind_vars, type(None)) or isinstance(dep_var, type(None)):
#         print(f'Error: there must be atleast 1 ind_var and dep_var\nRight now they are {ind_vars} and  {dep_var}')
#         sys.exit()
#
#     print(f'Length ind_var {len(ind_vars)}')
#
#     # if there are not ind_vars left return the R2_list, R2_adjusted_list, and ind_vars_used_ordered
#     if len(ind_vars) < 1:
#         return R2_list, R2_adjusted_list, ind_vars_used_ordered
#
#     headers_array = np.array(dataObj.get_headers())
#
#     if dep_var not in headers_array:
#         print(f'Error: dep_var: {dep_var} needs to be in {headers_array}')
#         sys.exit()
#     for ind_var in ind_vars:
#         if ind_var not in headers_array:
#             print(f'Error: ind_var: {ind_var} needs to be in {headers_array}')
#             sys.exit()
#
#     # if it is the first time of the function running for the data
#     max_R2_var = tuple()
#     if len(R2_list) == 0:
#         max_R2_var = ('holder', float('-inf'))
#     else:
#         max_R2_var = (ind_vars_used_ordered[-1], R2_list[-1])
#
#     # loop through all the variables to find the one with the best
#
#     for ind_var in ind_vars:
#         # print(f'\nindVar current {ind_var}')
#
#         # run linear regression bassed off of the method chosen
#         if method == 'scipy':
#             lin_reg.linear_regression(ind_vars_used_ordered+[ind_var], dep_var, 'scipy')
#         elif method == 'normal':
#             lin_reg.linear_regression(ind_vars_used_ordered+[ind_var], dep_var, 'normal')
#         elif method == 'qr':
#             lin_reg.linear_regression(ind_vars_used_ordered+[ind_var], dep_var, 'qr')
#
#         # see if R2 is greater
#         if lin_reg.R2 > max_R2_var[1]:
#             max_R2_var = (ind_var, lin_reg.R2)
#
#     # print(f'\nVar chosen is {max_R2_var}')
#
#     ind_vars.remove(max_R2_var[0])
#     ind_vars_used_ordered.append(max_R2_var[0])
#     R2_list.append(max_R2_var[1])
#     step_wise_linear_regression(dataObj, ind_vars, dep_var, method,R2_list, R2_adjusted_list, ind_vars_used_ordered)
#
#     return R2_list, R2_adjusted_list, ind_vars_used_ordered
#
# print(step_wise_linear_regression(normalized_day_bike_data, bike_headers[:-3], 'cnt'))
# R2_list, R2_adjusted_list, ind_vars_used_ordered = step_wise_linear_regression(normalized_day_bike_data, bike_headers[:-3], 'cnt')
#
# print(f'\n{R2_list}\n{ind_vars_used_ordered}')
# def make_train_test(data, train_percentage=0.50):
#     data_matrix = data.get_all_data()
#     num_samps = data_matrix.shape[0]
#     train_num = round(num_samps * train_percentage)
#     test_num = num_samps - train_num
#     print(train_num)
#     train_samps_idx = np.random.randint(num_samps, size=train_num)
#     print(train_samps_idx.shape)
#     test_samps_idx = []
#
#     for samp_row in range(0, num_samps):
#         if samp_row not in list(train_samps_idx):
#             test_samps_idx.append(samp_row)
#
#     test_samps_idx = np.array(test_samps_idx)
#     print(test_samps_idx.shape)
#     print(num_samps)
#     print(test_samps_idx.shape[0] + train_samps_idx.shape[0])
#     sorted_train = sorted(train_samps_idx)
#     sorted_test = sorted(test_samps_idx)
#
#     print(1)
#
# make_train_test(normalized_day_bike_data)




#import CSV module to read in csv files so that we can add data types to the data and manmipulate
# the dteday var to a utc time integer
# import csv
#
# with open('data/bike-sharing-datasets/day.csv', 'r', newline = '') as day_csv:
#     day_reader = csv.reader(day_csv, delimiter = ',')
#
#     print(1)

brain_filename = 'data/brain.csv'
brain_data = data.Data(brain_filename)

brain_headers = brain_data.get_headers()
print(f'Brain Data Headers:\n{brain_headers}\n\n{len(brain_headers)} Variables')
lin_reg_brain_scipy = linear_regression.LinearRegression(brain_data)
print(f'The time it took for Scipy method is:')
lin_reg_brain_scipy.linear_regression(brain_headers[:61], brain_headers[-1], 'scipy')
print(f'\nThe MSSE for Scipy method is {lin_reg_brain_scipy.mean_sse()}')

lin_reg_brain_norm = linear_regression.LinearRegression(brain_data)
print(f'The time it took for Normal method is:')
lin_reg_brain_norm.linear_regression(brain_headers[:61], brain_headers[-1], 'normal')
print(f'\nThe MSSE for Normal method is {lin_reg_brain_norm.mean_sse()}')


lin_reg_brain_qr = linear_regression.LinearRegression(brain_data)
print(f'The time it took for QR method is:')
lin_reg_brain_qr.linear_regression(brain_headers[:61], brain_headers[-1], 'qr')
print(f'\nThe MSSE for QR method is {lin_reg_brain_qr.mean_sse()}')

iris_filename = 'data/iris.csv'
iris_data = data.Data(iris_filename)

A = iris_data.select_data(['sepal_length', 'petal_width'])
A1 = np.hstack([A, np.ones([len(A), 1])])

lin_reg_qr = linear_regression.LinearRegression(iris_data)
myQ, myR = lin_reg_qr.qr_decomposition(A1)

Q, R = np.linalg.qr(A1)
brain_headers_list = []
with open('data/brain_var_names.txt','r') as brain_headers_txt:
    brain_headers_list = brain_headers_txt.read().split(',')
brain_filename = 'data/brain.csv'
brain_data = data.Data(brain_filename, headers = brain_headers_list)
print(brain_data)
#
#
#
# import csv
#
#
# with open('data/brain.csv','r') as brain_csv:
#     with open('data/brain_var_names.txt','r') as brain_headers_txt:
#         with open('data/brainCorrect.csv','w') as correct_csv:
#             brain_headers_list = brain_headers_txt.read().split(',')
#             csv_writer = csv.writer(correct_csv, delimiter = ',')
#             csv_writer.writerow(brain_headers_list)
#             brain_data_array = np.array(brain_csv.read().split(','))
#             print(len(brain_headers_list))
#             print(len(brain_data_array))
#
#             print(1)



# TODO see what is quicker factorization Final is slower but need to explain why Actually there is a difference when
# running



# def qr_factorization(A):
#     m, n = A.shape
#     Q = np.zeros((m, n))
#     R = np.zeros((n, n))
#
#     for j in range(n):
#         v = A[:, j]
#
#         for i in range(j):
#             q = Q[:, i]
#             R[i, j] = q.dot(v)
#             v = v - R[i, j] * q
#
#         norm = np.linalg.norm(v)
#         Q[:, j] = v / norm
#         R[j, j] = norm
#     return Q, R
#
#
# def qr_factorization_final(A):
#     m, n = A.shape
#     Q = np.zeros((m, n))
#     for j in range(n):
#         u = A[:, j]
#         for i in range(j):
#             u = u - (Q[:, i]@u) * Q[:, i]
#         Q[:, j] = u/np.linalg.norm(u)
#
#     R = Q.T@A
#     return Q, R
# def gram_schmidt_process(A):
#     """Perform QR decomposition of matrix A using Gram-Schmidt process."""
#     (num_rows, num_cols) = np.shape(A)
#
#     # Initialize empty orthogonal matrix Q.
#     Q = np.empty([num_rows, num_rows])
#     cnt = 0
#
#     # Compute orthogonal matrix Q.
#     for a in A.T:
#         u = np.copy(a)
#         for i in range(0, cnt):
#             proj = np.dot(np.dot(Q[:, i].T, a), Q[:, i])
#             u = u - proj
#
#         e = u / np.linalg.norm(u)
#         Q[:, cnt] = e
#
#         cnt += 1  # Increase columns counter.
#
#     # Compute upper triangular matrix R.
#     R = np.dot(Q.T, A)
#
#     return (Q, R)
#

# import time
#
#
#
#
#
# iris_filename = 'data/iris.csv'
# iris_data = data.Data(iris_filename)
#
# A = iris_data.select_data(['sepal_length', 'petal_width'])
# A1 = np.hstack([A, np.ones([len(A), 1])])
#
# brain_filename = 'data/brain.csv'
# brain_data = data.Data(brain_filename)
# lin_reg_brain = linear_regression.LinearRegression(brain_data)
# B = brain_data.select_data(brain_data.get_headers()[:61])
# B1 = np.hstack([B, np.ones([len(B), 1])])
# st = time.time()
# t1 ,t2  = qr_factorization(B1)
# print("----%.12f----"%(time.time()-st))
#
#
# st = time.time()
# t3, t4 = qr_factorization_final(B1)
# print("----%.12f----"%(time.time()-st))



# lin_reg_qr = linear_regression.LinearRegression(iris_data)
# myQ, myR = lin_reg_qr.qr_decomposition(A1)
#
# Q, R = np.linalg.qr(A1)
#
# print('NOTE: It is ok if numbers match but whole columns are negated.\n')
# print(f'Your Q shape is {myQ.shape} and should be {Q.shape}')
# print(f'Your R shape is {myR.shape} and should be {R.shape}')
# print(f'1st few rows of your Q are\n{myQ[:3]} and should be\n{Q[:3]}')
# print(f'\nYour R is\n{myR[:5]} and should be\n{R[:5]}')
#
# lin_reg_qr.linear_regression(['sepal_length'], 'petal_width', 'qr')
# lin_reg_qr.scatter('sepal_length', 'petal_width', 'qr')
# lin_reg_qr.show()
#
# lin_reg_qr.linear_regression(['sepal_length'], 'petal_width', 'scipy')
# lin_reg_qr.scatter('sepal_length', 'petal_width', 'scipy')
# lin_reg_qr.show()

# testA = np.array([[2,3],[2,4],[1,1]])
# testA1 = np.hstack([testA, np.ones([len(testA), 1])])
# print(f'{testA}\n')
#
# tQ0, TR0 = qr_factorization(testA)
# print(f'{tQ0}\n{TR0}\n')
#
# tQ1, TR1 = qr_factorization_final(testA)
# print(f'{tQ1}\n{TR1}\n')
#
#
#
# Qcorrect, Rcorrect = np.linalg.qr(testA)
# print(f"Correct Q:\n{Qcorrect}\nCorrect R:\n{Rcorrect}")

