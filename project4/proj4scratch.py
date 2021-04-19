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


# face_plot(face_imgs,face_names)



def plot_img(ax, image_array, title = ''):

    if len(image_array.shape) == 2:
        img_plot = ax.imshow(image_array,cmap=plt.get_cmap('gray'))
        # img_plot = ax.imshow(image_array)
    elif len(image_array.shape) == 3:
        img_plot = ax.imshow(image_array)

    ax.set_title(title)
    ax.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    return ax

def distance_between_images(img_1, img_2):

    img_1 = img_1.reshape(1, img_1.size)
    img_2 = img_2.reshape(1, img_2.size)
    image_dist = np.sqrt(np.sum((img_1 - img_2) * (img_1 - img_2), axis=1))
    return image_dist[0]



def vectorize_images(images_data):

    images_data_vec = images_data.reshape(images_data.shape[0], images_data.shape[1]*images_data.shape[2])
    return images_data_vec



def match_photo(known_data, known_data_labels, querry_data, num_top_pcs = 32, threshold=0.70,normalization = False, vectorize_image_data = True):

   if vectorize_image_data:
       known_data = vectorize_images(known_data)
       querry_data = querry_data.reshape(1, querry_data.shape[0]*querry_data.shape[1])

   known_df = pd.DataFrame(known_data)
   known_pca = pca_cov.PCA_COV(known_df)
   known_pca.pca(np.arange(known_df.shape[1]), normalize=normalization)
   known_pca_projections = known_pca.pca_project(np.arange(num_top_pcs))

   # set up querry_image for projection into PCA space
   querry_data = querry_data - known_pca.get_A_mean()
   if normalization:
       querry_data = known_pca.normalize_separately(querry_data)

   P_hat = known_pca.get_eigenvectors()[:, :num_top_pcs]
   querry_img_projection = querry_data @ P_hat

   # computing distance between querry image projection and projected known images
   # Will return at the first image that meets the thresh hold
   for known_name, known_projection in zip(known_data_labels,known_pca_projections):
       image_distance = abs(distance_between_images(known_projection, querry_img_projection))
       if image_distance < threshold:
           return known_name



face_imags_df = pd.DataFrame(vectorize_images(face_imgs), index = face_names)



# show unique faces in image data base
unique_celebs = np.unique(np.array(face_names))
print(unique_celebs)

#create clebs DF
face_name_series = pd.Series(np.array(face_names))
name_count = face_name_series.value_counts()

# Gerorge W bush has most photos so going to use him

# showing image I am testing which is from the know dataset
test_know_face_img = face_imgs[3]
test_know_face_name = face_names[3]

test_know_face_plot = plt.imshow(test_know_face_img, cmap=plt.get_cmap('gray'))
test_know_face_plot.axes.set_title(f'{test_know_face_name}\nKnow Test Image')
test_know_face_plot.axes.get_xaxis().set_visible(False)
test_know_face_plot.axes.get_yaxis().set_visible(False)
plt.show()



face_imgs_vec  = face_imgs.reshape(face_imgs.shape[0],face_imgs.shape[1]*face_imgs.shape[2])

# test_match_result = match_photo(known_data=face_imgs, known_data_labels=face_names,
#             querry_data=test_know_face_img, num_top_pcs=10)
#
#
# print(f'{test_match_result}')


gw_test_im = plt.imread('data/gw_test_nocrop.png')
fig, axes = plt.subplots(1,1)
axes = plot_img(axes,gw_test_im,f'G.W Bush Tesh Im No Crop Colored')
plt.show()

# WITH OPEN CV
test_know_GWB_face_img = face_imgs[1]
test_know_GWB_face_name = face_names[1]
import cv2

# 0 in imread makes it grey scale
gw_img_cv = cv2.imread('data/gw_test_nocrop.png',0)
gw_img_cv_face_crop = gw_img_cv[104:258,130:285]

gw_resized = cv2.resize(gw_img_cv_face_crop, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)

fig, axes = plt.subplots(2,2)
axes[0,0] = plot_img(axes[0,0],gw_img_cv,f'G.W Bush No Crop Grey')
axes[1,0] = plot_img(axes[1,0],gw_img_cv_face_crop,f'Cropped Grey')
axes[0,1] = plot_img(axes[0,1],test_know_GWB_face_img, f'{test_know_GWB_face_name}\nKnow Test Image')
axes[1,1] = plot_img(axes[1,1],gw_resized, f'Resized to 64 by 64')
plt.show()

test_match_result_GWB = match_photo(known_data=face_imgs, known_data_labels=face_names,
            querry_data=gw_resized, num_top_pcs=3, threshold= 0.7)

print(f'{test_match_result_GWB}')

print(f'Done Testing')