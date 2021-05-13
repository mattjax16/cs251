import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import KNN



spiral_1_train = np.loadtxt('data/spiral_train_1.csv', skiprows=1, delimiter=',')
spiral_1_val = np.loadtxt('data/spiral_val_1.csv', skiprows=1, delimiter=',')
spiral_2_train = np.loadtxt('data/spiral_train_2.csv', skiprows=1, delimiter=',')
spiral_2_val = np.loadtxt('data/spiral_val_2.csv', skiprows=1, delimiter=',')

spiral_1_train_y = spiral_1_train[:, 2]
spiral_1_val_y = spiral_1_val[:, 2]
spiral_2_train_y = spiral_2_train[:, 2]
spiral_2_val_y = spiral_2_val[:, 2]

spiral_1_train = spiral_1_train[:, :2]
spiral_1_val = spiral_1_val[:, :2]
spiral_2_train = spiral_2_train[:, :2]
spiral_2_val = spiral_2_val[:, :2]

print(f'Spiral 1 train {spiral_1_train.shape}, classes {spiral_1_train_y.shape}')
print(f'Spiral 1 validation {spiral_1_val.shape}, classes {spiral_1_val_y.shape}')
print(f'Spiral 2 train {spiral_2_train.shape}, classes {spiral_2_train_y.shape}')
print(f'Spiral 2 validation {spiral_2_val.shape}, classes {spiral_2_val_y.shape}')


n_classes = 4
#
# classifier = KNN(num_classes=n_classes)
# classifier.train(spiral_1_train, spiral_1_train_y)
#
# k = 1
# spiral_1_y_pred = classifier.predict(spiral_1_train, k)
# acc = classifier.accuracy(y=spiral_1_train_y, y_pred=spiral_1_y_pred)
# print(f'Your accuracy with K=1 is {acc} and should be 1.0')
# classifier.plot_predictions(4,400)






classifier_gpu = KNN(num_classes=n_classes,use_gpu=True,kernal_accelerate=False)
classifier_gpu.train(spiral_1_train, spiral_1_train_y)

k = 1
spiral_1_y_pred = classifier_gpu.predict(spiral_1_train, k)
acc = classifier_gpu.accuracy(y=spiral_1_train_y, y_pred=spiral_1_y_pred)
print(f'Your accuracy with K=1 is {acc} and should be 1.0')
classifier_gpu.plot_predictions(4,100)






# classifier_gpu_kernal = KNN(num_classes=n_classes,use_gpu=True)
# classifier_gpu_kernal.train(spiral_1_train, spiral_1_train_y)
#
# k = 1
# spiral_1_y_pred = classifier_gpu_kernal.predict(spiral_1_train, k)
# acc = classifier_gpu_kernal.accuracy(y=spiral_1_train_y, y_pred=spiral_1_y_pred)
# print(f'Your accuracy with K=1 is {acc} and should be 1.0')
# classifier_gpu_kernal.plot_predictions(4,300)


















###EMAIL

import email_preprocessor as epp




# word_freq, num_emails = epp.count_words(email_path='data/enron_dev/')
#
# print(f'You found {num_emails} emails in the datset. You should have found 5.')

# hard_code_words = ['subject', 'you', 'get', 'that', 'new', 'car', 'now', 'can', 'be', 'smart', 'love', 'ecards', 'christmas', 'tree', 'farm', 'pictures', 're', 'rankings', 'thank']
# features, y = epp.make_feature_vectors(hard_code_words, email_path='data/enron_dev/')
# np.random.seed(2)
# x_train, y_train, inds_train, x_test, y_test, inds_test = epp.make_train_test_sets(features, y)
# epp.retrieve_emails(inds = [0,2,4])

word_freq, num_emails = epp.count_words()

