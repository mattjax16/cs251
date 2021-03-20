from data import *
from linear_regression_old import *



iris_data = Data('iris.csv')

lin_reg = LinearRegression(iris_data)
lin_reg.linear_regression(ind_vars=['sepal_length'],dep_var='petal_width', method = 'scipy')

lin_reg_norm = LinearRegression(iris_data)
lin_reg_norm.linear_regression(['sepal_length'], 'petal_width', 'normal')
lin_reg_norm.scatter('sepal_length', 'petal_width', 'normal')
lin_reg_norm.show()



# test52_data = Data('testdata52.csv')
# print(test52_data)
# lin_reg = LinearRegression(test52_data)
# lin_reg.pair_plot(test52_data.get_headers())
# lin_reg.show()




# iris_data = Data('iris.csv')
# # print(iris_data)
# #
# lin_reg = LinearRegression(iris_data)
# lin_reg.linear_regression(ind_vars=['sepal_length'],dep_var='petal_width', method = 'scipy')
#
# lin_reg.scatter('sepal_length', 'petal_width', 'scipy')
# lin_reg.show()


# # test shapes of instance variables
# print(f'Shape of your A data array is\n{lin_reg.A.shape} and should be\n(150, 1)')
# print(f'Shape of your y dep var vector is\n{lin_reg.y.shape} and should be\n(150, 1)\n')
# print(f"Your independent variables are {lin_reg.ind_vars} and should be ['sepal_length']")
# print(f'Your dependent variables are {lin_reg.dep_var} and should be petal_width\n')
# print(f'Shape of your slope fits are {lin_reg.slope.shape} and should be (1, 1)')
#
# # Test specific values
# print(f'Your slope is {lin_reg.slope} and should be [[0.75292]]')
# print(f'Your intercept is {lin_reg.intercept:.2f} and should be -3.20')
# print(f'Your R^2 is {lin_reg.R2:.2f} and should be 0.67')
# print(f'Your 1st few residuals are\n{lin_reg.residuals[:5].T} and should be\n[[-0.43966 -0.28908 -0.1385  -0.06321 -0.36437]]')

# np.random.seed(0)
# # test data: 3 data samples, 4 dimensional.
# test_slope = np.random.normal(size=(5, 1))
# test_X = np.random.normal(size=(3, 5))
# pie = np.pi
# y_pred = lin_reg.predict(test_slope, pie, test_X)
# print(f'Your model y predictions are\n{np.squeeze(y_pred)} and should be\n[2.18518 5.82409 3.23376]')
# print(f'Your model y predictions shape is\n{y_pred.shape} and should be\n(3, 1)')