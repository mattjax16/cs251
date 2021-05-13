import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import KNN
import email_preprocessor as epp


from naive_bayes_multinomial import NaiveBayes




# num_test_classes = 4
# np.random.seed(0)
# data_train = np.random.randint(low=0, high=num_test_classes, size=(100, 10))
# data_test = np.random.randint(low=0, high=num_test_classes, size=(15, 10))
# y_test = np.random.randint(low=0, high=num_test_classes, size=(100,))
#
#
# nbc = NaiveBayes(num_classes=num_test_classes)
# nbc.train(data_train, y_test)
# test_y_pred = nbc.predict(data_test)
# test_confus = np.array([1 ,1 ,3 ,1 ,0 ,2 ,2 ,0 ,0 ,2 ,0 ,3 ,3 ,2 ,0])
# nbc.confusion_matrix(test_confus,test_y_pred)
# print(f'Your predicted classes are\n{test_y_pred}\nand should be\n[3 0 3 1 0 2 2 0 0 2 0 3 0 2 0]')
email_train_x = np.load('email_train_x.npy')
email_train_y = np.load('email_train_y.npy')
email_train_inds = np.load('email_train_inds.npy')
email_test_x = np.load('email_test_x.npy')
email_test_y = np.load('email_test_y.npy')
email_test_inds = np.load('email_test_inds.npy')
# email_nbc = NaiveBayes(num_classes=2)
# email_nbc.train(email_train_x, email_train_y)
# email_y_pred = email_nbc.predict(email_test_x)
# print(f'The Accuarcy is {email_nbc.accuracy(email_test_y,email_y_pred)}\n{email_nbc.accuracy(email_y_pred,email_test_y)}')
#
email_knn = KNN()
email_knn_gpu = KNN(use_gpu=True)



email_knn.train(email_train_x, email_train_y)

email_kn_test_cpu = email_test_x[:200]



cccc  =  email_knn.predict(email_kn_test_cpu,k = 7)
print(1)