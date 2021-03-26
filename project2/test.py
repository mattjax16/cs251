import numpy as np
import matplotlib.pyplot as plt

from data import *
import transformation
import palettable
from palettable import cartocolors
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 20, 'figure.figsize': (4, 4)})

np.set_printoptions(suppress=True, precision=5)
brewer_colors = cartocolors.qualitative.Safe_4.mpl_colormap


import requests
data_url = 'https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv'
data_request = requests.get('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/WaffleDivorce.csv')
print(data_request.text)

data_string_list = data_request.text.splitlines()
data_string_list.insert(1,'string;string;numeric;numeric;numeric;numeric;'
                        'numeric;numeric;numeric;numeric;numeric;numeric;numeric')

data_parsed = [line.split(';') for line in data_string_list]

with open('WH_divorce.csv','w') as WH_divorce_csv:
    csv_writer = csv.writer(WH_divorce_csv, delimiter = ',')
    csv_writer.writerows(data_parsed)


states_list = []
for line in data_parsed[2:]:
    states_list.append(line[0])
print(states_list)


wh_data = Data('WH_divorce.csv')
Allwh_data = AllData('WH_divorce.csv')
Allwh_transform = transformation.Transformation(Allwh_data)
print(Allwh_data.get_headers())
Allwh_transform.pair_plot(Allwh_data.get_headers()[:7],title = 'Waffle House Present Data',
                         cat = Allwh_data.get_headers()[7], diag = 'hist')
#
# #showing the use of a size and color scatter plot with the use of mpg data
# plt.rcParams.update({'font.size': 20, 'figure.figsize': (12, 12)})
# auto_data = Data('auto-mpg.csv')
# auto_transform = transformation.Transformation(auto_data)
#
# auto_transform.normalize_separately_zscore()


# auto_transform.scatter_color_size('mpg', 'acceleration', 'origin', 'cylinders')
# auto_transform.scatter_color_size('mpg', 'acceleration','cylinders', 'origin')
# auto_transform.scatter_color_size('mpg', 'acceleration', 'modelyear', 'cylinders')
# auto_transform.scatter_color_size('mpg', 'acceleration', 'cylinders','modelyear')
# auto_transform.scatter_color_size('mpg', 'weight', 'modelyear','origin', size_scale = 160)
#
# auto_transform.scatter_color_size('weight', 'acceleration', 'mpg','modelyear')
# auto_transform.scatter_color_size('weight', 'acceleration','modelyear', 'mpg')
# auto_transform.show()


# iris_data = Data('iris.csv')
# print(iris_data)
# iris_transform = transformation.Transformation(iris_data)
#
# # iris_transform.scatter_color('sepal_length', 'petal_length', 'sepal_width')
#
# iris_transform.heatmap(cmap = brewer_colors )
# iris_transform.show()



# letter_data = Data('letter_data.csv')
# letter_transform = transformation.Transformation(letter_data)
# letter_3D_together = letter_transform.normalize_separately()
# letter_transform.rotate_3d('X', 271)
# letter_transform.pair_plot(letter_data.get_headers())
# letter_transform.show()
#
# def scatter3d(data, headers=['X', 'Y', 'Z'], title='Raw letter data'):
#     '''Creates a 3D scatter plot to visualize data'''
#     letter_xyz = data.select_data(headers)
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#
#     # Scatter plot of data in 3D
#     ax.scatter3D(letter_xyz[:, 0], letter_xyz[:, 1], letter_xyz[:, 2])
#
#     # style the plot
#     ax.set_xlabel(headers[0])
#     ax.set_ylabel(headers[1])
#     ax.set_zlabel(headers[2])
#     ax.set_title(title)
#
#     plt.show()
#
#
# scatter3d(letter_transform.data)
# iris_data = Data('iris.csv')
#
# iris_transform = transformation.Transformation(iris_data)
# iris_transform.project(['sepal_length', 'sepal_width', 'petal_length'])
# rot_3d = iris_transform.rotate_3d('sepal_length', 10)
# print(rot_3d[:5])
# iris_transform.project(iris_data.get_headers())
# iris_transform.normalize_together()
# iris_transform.project(['sepal_length', 'sepal_width', 'petal_length'])
# C = iris_transform.scale_matrix([1,2,(1/3)])
#
# translate_part = iris_transform.translation_matrix([-0.5,0,0.5])
# C[:,3] = translate_part[:,3]


# transformed_data = iris_transform.transform(C)



# print(f'\nCompound transformation matrix:\n{C}\nTransformed data:\n{transformed_data[:5]}')



# iris_transform = transformation.Transformation(iris_data)
# iris_transform.project(['sepal_length', 'petal_length'])
# iris_transform.scatter('sepal_length','petal_length')
# iris_transform.show()
# iris_data = Data('iris.csv')
# iris_transform = transformation.Transformation(iris_data)
#
# iris_transform.project(['sepal_length', 'petal_length', 'petal_width'])
# iris_transform.pair_plot(['sepal_length', 'petal_length', 'petal_width'])
# iris_transform.show()



# iris_transform.project(['sepal_length','sepal_width', 'petal_length'])
# iris_translation = iris_transform.translation_matrix([-0.5,0,1.5])
# print(iris_translation)
#
# iris_scale = iris_transform.scale_matrix([1,2,(1/3)])
# print(iris_scale)
#
# # iris_transform.translate([1,2,(1/3)])
# iris_transform.scale([1,2,10])