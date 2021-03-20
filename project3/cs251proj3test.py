import os
import random
import numpy as np
import matplotlib.pyplot as plt

from data import Data
import linear_regression

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)
iris_filename = 'iris.csv'
# iris_data = Data(iris_filename)
# lin_reg = linear_regression.LinearRegression(iris_data)
#
# lin_reg.linear_regression(['sepal_length'], 'petal_length')
#
# print(f'Your regression fitted slope coefficients have shape\n{lin_reg.slope.shape}\nand the shape should be\n(1, 1)')
# print(f'Your regression fitted slope coefficient is\n{lin_reg.slope[0, 0]:.4f}\nand it should be\n1.8584')
# print(f'Your regression fitted intercept is\n{lin_reg.intercept:.4f}\nand it should be\n-7.1014')

iris_data_pred = Data(iris_filename)
iris_data_pred.limit_samples(100, 105)
iris_headers = iris_data_pred.get_headers()

lin_reg_pred = linear_regression.LinearRegression(iris_data_pred)
lin_reg_pred.linear_regression(iris_headers[:2], iris_headers[3])
y_pred = lin_reg_pred.predict()

print(f'Your model y predictions are\n{np.squeeze(y_pred)} and should be\n[2.48684 1.81579 2.06842 2.01316 2.11579]')
print(f'Your model y predictions shape is\n{y_pred.shape} and should be\n(5, 1)')