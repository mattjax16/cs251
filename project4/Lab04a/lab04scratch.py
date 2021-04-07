import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import scipy as sp
from mpl_toolkits.mplot3d import Axes3D
import requests
import csv

plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'figure.figsize': [8,8]})

np.set_printoptions(suppress=True, precision=5)


bc_df = pd.read_csv('BC_Lab04A_data.csv')

print(bc_df)
#1
bc_df = pd.read_csv('https://raw.githubusercontent.com/patrickmlong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv ')
# bc_df

#2 b.
print(f'Headers of the Breast Cancer Data Frame are:')
print(" , ".join(f"{header}" for header in bc_df.columns.values))

first5_items = []
to_numpy_list = []
dot_array_list = []
for item_num, item in enumerate(bc_df.items()):
    if item_num > 4:
        break
    first5_items.append(item)
    to_numpy_list.append(item[1].to_numpy())
    dot_array_list.append(item[1].array)




for m in to_numpy_list:
    print(m[:1])
# print(f'The first 5 items from bc_df.items() (really first 5 vals\n' +
#       f'of the first 5 series) are:\n {[item[1].to_numpy for item in first5_items]}')



#2 c.
# x_items = []
# for item_num, item in enumerate(bc_df.items()):
#     x_items.append(item)
#
# x_tups = []
# for item in bc_df.itertuples():
#     x_tups.append(item)
#
#
# x_itemit = []
# for item in bc_df.iteritems():
#     x_itemit.append(item)

print("done test")