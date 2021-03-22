import os
import random
import numpy as np
import matplotlib.pyplot as plt

from data import Data
import linear_regression

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20})

np.set_printoptions(suppress=True, precision=5)
# iris_filename = 'iris.csv'
# iris_data = Data(iris_filename)
# lin_reg = linear_regression.LinearRegression(iris_data)
#
# lin_reg.linear_regression(['sepal_length'], 'petal_length')
#
# print(f'Your regression fitted slope coefficients have shape\n{lin_reg.slope.shape}\nand the shape should be\n(1, 1)')
# print(f'Your regression fitted slope coefficient is\n{lin_reg.slope[0, 0]:.4f}\nand it should be\n1.8584')
# print(f'Your regression fitted intercept is\n{lin_reg.intercept:.4f}\nand it should be\n-7.1014')
#
# iris_data_pred = Data(iris_filename)
# iris_data_pred.limit_samples(100, 105)
# iris_headers = iris_data_pred.get_headers()
#
# lin_reg_pred = linear_regression.LinearRegression(iris_data_pred)
# lin_reg_pred.linear_regression(iris_headers[:2], iris_headers[3])
# y_pred = lin_reg_pred.predict()
#
# print(f'Your model y predictions are\n{np.squeeze(y_pred)} and should be\n[2.48684 1.81579 2.06842 2.01316 2.11579]')
# print(f'Your model y predictions shape is\n{y_pred.shape} and should be\n(5, 1)')
#
#
#
#
#
#
#
#
# np.random.seed(0)
# # fake test data: 3 data samples, 4 dimensional.
# test_slope = np.random.normal(size=(5, 1))
# test_X = np.random.normal(size=(3, 5))
# lin_reg.slope = test_slope
# lin_reg.intercept = np.pi
# y_pred = lin_reg.predict(test_X)
# print(f'Your model y predictions are\n{np.squeeze(y_pred)} and should be\n[2.18518 5.82409 3.23376]')
# print(f'Your model y predictions shape is\n{y_pred.shape} and should be\n(3, 1)')
#
#
# lin_reg.linear_regression(['sepal_length'], 'petal_width')
#
# # test shapes of instance variables
# print(f'Shape of your A data array is\n{lin_reg.A.shape} and should be\n(150, 1)')
# print(f'Shape of your y dep var vector is\n{lin_reg.y.shape} and should be\n(150, 1)\n')
# print(f"Your independent variables are:\n{lin_reg.ind_vars}\nand should be:\n['sepal_length']")
# print(f'Your dependent variables are:\n{lin_reg.dep_var}\nand should be:\npetal_width\n')
# print(f'Shape of your slope fits are {lin_reg.slope.shape} and should be (1, 1)')
#
# # Test specific values
# print(f'Your slope is {lin_reg.slope} and should be [[0.75292]]')
# print(f'Your intercept is {lin_reg.intercept:.2f} and should be -3.20')
# print(f'Your R^2 is {lin_reg.R2:.2f} and should be 0.67')
# print(f'Your 1st few residuals are\n{lin_reg.residuals[:5].T} and should be\n[[-0.43966 -0.28908 -0.1385  -0.06321 -0.36437]]')
#


# iris_filename = 'data/iris.csv'
# iris_data = Data(iris_filename)
# lin_reg = linear_regression.LinearRegression(iris_data)
# lin_reg.linear_regression(['sepal_length'], 'petal_width')
#
# lin_reg.scatter('sepal_length', 'petal_width', 'Regression on Iris!')
# lin_reg.show()

# test52_filename = 'data/testdata52.csv'
# test52_data = Data(test52_filename)
# lin_reg = linear_regression.LinearRegression(test52_data)
# print(test52_data)
# lin_reg.pair_plot(test52_data.get_headers(), hists_on_diag=False)

# test_A = np.r_[1:10].reshape((9, 1))
# test_p = 3
#
# # Test cubic
# lin_reg = linear_regression.LinearRegression(Data())
# print(f'Your polynomial matrix:\n{lin_reg.make_polynomial_matrix(test_A, 3)}')
#
# true_mat = '''
# [[  1.   1.   1.]
#  [  2.   4.   8.]
#  [  3.   9.  27.]
#  [  4.  16.  64.]
#  [  5.  25. 125.]
#  [  6.  36. 216.]
#  [  7.  49. 343.]
#  [  8.  64. 512.]
#  [  9.  81. 729.]]
# '''
# print('It should look like:\n', true_mat)




# poly_filename = 'data/poly_data.csv'
# poly_data = Data(poly_filename)
# poly_lin_reg = linear_regression.LinearRegression(poly_data)
# poly_lin_reg.linear_regression(['X'], 'Y', p = 7)
# poly_lin_reg.scatter('X','Y')
# poly_lin_reg.show()
# print(f'MSSE: {poly_lin_reg.m_sse}')


poly_filename = 'data/poly_data.csv'
fit_set = Data(poly_filename)
fit_set.limit_samples(0,50)
validation_set = Data(poly_filename)
validation_set.limit_samples(50,100)
print(fit_set)
print(validation_set)
# print(fit_set)
# print(validation_set)
poly_lin_reg_fit = linear_regression.LinearRegression(fit_set)
poly_lin_reg_fit.linear_regression(['X'], 'Y', p = 7)

poly_lin_reg_fit.scatter('X','Y', title = f'Fit Data\nMSSE : {poly_lin_reg_fit.m_sse:.8f}')
poly_lin_reg_fit.show()



poly_lin_reg_val = linear_regression.LinearRegression(validation_set)
fit_slopes =  poly_lin_reg_fit.get_fitted_slope()
fit_intercept= poly_lin_reg_fit.get_fitted_intercept()
poly_lin_reg_val.initialize(['X'], 'Y', p = 7, slope = fit_slopes,
                       intercept = fit_intercept)
poly_lin_reg_val.scatter('X','Y', title = f'Validation Data\nMSSE : {poly_lin_reg_val.m_sse:.8f}')
poly_lin_reg_val.show()