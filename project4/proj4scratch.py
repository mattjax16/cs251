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
# fig, axs = plt.subplots(2,2)
# keep_top_k = 1
# pca.pca(iris_headers, normalize=True)
# # 2x2 grid of scatter plots
# for grid_y in range(0,2):
#     for grid_x in range(0,2):
#         reconstructed_data =  pca.pca_then_project_back(keep_top_k)
#
#         axs[grid_y,grid_x].scatter(reconstructed_data[:,0],reconstructed_data[:,1])
#         axs[grid_y,grid_x].set_title(f'Top {keep_top_k} K(s) Kept')
#         axs[grid_y,grid_x].set_xlabel('sepalLength')
#         axs[grid_y,grid_x].set_ylabel('sepalWidth')
#
#         keep_top_k += 1
#
# fig.tight_layout()

def face_plot(face_imgs, face_names):
    '''Create a 5x5 grid of face images

    Parameters:
    -----------
    face_imgs: ndarray. shape=(N, img_y, img_x).
        Grayscale images to show.
    face_names: ndarray. shape=(N,).
        Names of the person in each image represented as strings.

    TODO:
    - Create a 5x5 grid of plots of a legible size
    - In each plot, show the grayscale image and make the title the person's name.
    '''
    fig, axs = plt.subplots(5,5)

    for face_img, face_name, ax in zip(face_imgs,face_names,axs.flatten()):
        ax.imshow(face_img,cmap=plt.get_cmap('gray'))
        ax.set_title(face_name)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


face_imgs = np.load('data/lfwcrop.npy')
face_names = np.loadtxt('data/lfwcrop_ids.txt', dtype=str, delimiter='\n')


face_plot(face_imgs,face_names)

print(f'Done Testing')