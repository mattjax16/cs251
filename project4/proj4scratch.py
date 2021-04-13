import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import pca_cov

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)

iris_data = pd.read_csv('data/iris.csv')
pca = pca_cov.PCA_COV(iris_data)

# Test pca (no normalization) here
iris_headers = list(iris_data.columns[:-1])
# pca.pca(iris_headers)
pca.pca(iris_headers, normalize=True)
# question is an array passed in not python list
# num_dims = 2
# t_list = [0,2,3]
# pcs_to_keep = np.arange(num_dims)
# # iris_proj = pca.pca_project(pcs_to_keep)
# iris_proj = pca.pca_project(t_list)
# print(iris_proj.shape)
fig, axs = plt.subplots(2,2)
keep_top_k = 1
pca.pca(iris_headers, normalize=True)
# 2x2 grid of scatter plots
for grid_y in range(0,2):
    for grid_x in range(0,2):
        reconstructed_data =  pca.pca_then_project_back(keep_top_k)

        axs[grid_y,grid_x].scatter(pca.data['sepalLength'],pca.data['sepalWidth'])
        axs[grid_y,grid_x].set_title(f'Top {keep_top_k} K(s) Kept')
        axs[grid_y,grid_x].set_xlabel('sepalLength')
        axs[grid_y,grid_x].set_ylabel('sepalWidth')

        keep_top_k += 1

fig.tight_layout()


print(f'Done Testing')