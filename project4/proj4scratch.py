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
print(1)