'''knn.py
K-Nearest Neighbors algorithm for classification
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import palettable
import pandas as pd
plt.style.use(['seaborn-colorblind', 'seaborn-darkgrid'])

#libraries for gpu acculeration
# GPU version of numpy basicly
import cupy as cp
import time


#TODO QUESTION what is the point of num_classes cant we just find that from unique data points
        #TODO why not do dot product for euclidien
class KNN:
    '''K-Nearest Neighbors supervised learning algorithm'''
    def __init__(self, num_classes = None, use_gpu = False, ret_gpu = None,  kernal_accelerate = True):
        '''KNN constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # exemplars: ndarray. shape=(num_train_samps, num_features).
        #   Memorized training examples
        self.exemplars = None
        # classes: ndarray. shape=(num_train_samps,).
        #   Classes of memorized training examples
        self.classes = None

        self.num_classes = num_classes

        #wether to gpu accelerate the calculations with cupy or not (just leave as numpy)
        self.use_gpu = use_gpu

        # wether to return data as cupy array or not
        if isinstance(ret_gpu,type(None)):
            self.ret_gpu = use_gpu
        else:
            self.ret_gpu = ret_gpu

        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

        # wether to use the custom Cuda Kernals for calculations
        self.kernal_accelerate = kernal_accelerate

        # Making Cuda Kernal Functions for increased speed on gpu
        # learned how to thanks to Cupy documentation!
        # https://readthedocs.org/projects/cupy/downloads/pdf/stable/

        # making kernal functions for different l Norms (distances)

        # L2 (euclidien distance kernal)
        self.euclidean_dist_kernel = cp.ReductionKernel(
            in_params='T x', out_params='T y', map_expr='x * x', reduce_expr='a + b',
            post_map_expr='y = sqrt(a)', identity='0', name='euclidean'
        )

        # L1 (manhattan distance kernal)
        self.manhattan_dist_kernel = cp.ReductionKernel(
            in_params='T x', out_params='T y', map_expr='abs(x)', reduce_expr='a + b',
            post_map_expr='y = a', identity='0', name='manhattan'
        )

        # these next 2 kerneals are used to get the mean of a cluster of data
        # (update the centroids)

        # gets the sum of a matrix based off of one hot encoding
        self.sum_kernal = cp.ReductionKernel(
            in_params='T x, S oneHotCode', out_params='T result',
            map_expr='oneHotCode ? x : 0.0', reduce_expr='a + b', post_map_expr='result = a', identity='0',
            name='sum_kernal'
        )

        # gets the count of a matrix from one hot encoding (by booleans)
        # TODO make a class variable to hold data type of data set
        self.count_kernal = cp.ReductionKernel(
            in_params='T oneHotCode', out_params='float32 result',
            map_expr='oneHotCode ? 1.0 : 0.0', reduce_expr='a + b', post_map_expr='result = a', identity='0',
            name='count_kernal'
        )



    def train(self, data, y):
        '''Train the KNN classifier on the data `data`, where training samples have corresponding
        class labels in `y`.

        Parameters:
        -----------
        data: ndarray. shape=(num_train_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_train_samps,). Corresponding class of each data sample.

        TODO:
        - Set the `exemplars` and `classes` instance variables such that the classifier memorizes
        the training data.
        '''


        # Check the size of the inputs
        if data.ndim != 2:
            print(f'Error Data needs to be a 2D Matrix!\nCurrently it is {data.ndim}')
            raise ValueError
        if y.ndim != 1:
            print(f'Error Data needs to be an Array!\nCurrently it is {y.ndim}')
            raise ValueError


        if self.use_gpu:
            if cp.get_array_module(data) == np:
                data = cp.array(data)
            if cp.get_array_module(y) == np:
                y = cp.array(y)
        else:
            if cp.get_array_module(data) == cp:
                data = np.array(data)
            if cp.get_array_module(y) == cp:
                y = np.array(y)


        # set the exemplar and classes instance variables
        self.exemplars = data
        self.classes = y
        self.num_classes = self.xp.unique(y).size


    def predict(self, data, k, dist_method = 'L2'):
        '''Use the trained KNN classifier to predict the class label of each test sample in `data`.
        Determine class by voting: find the closest `k` training exemplars (training samples) and
        the class is the majority vote of the classes of these training exemplars.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network.
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.

        dist_method: str . either L1 or L2 norm (default L2)

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_test_samps,). Predicted class of each test data
        sample.

        TODO:
        - Compute the distance from each test sample to all the training exemplars.
        - Among the closest `k` training exemplars to each test sample, count up how many belong
        to which class.
        - The predicted class of the test sample is the majority vote.
        '''

        # Check the size of the input
        if data.ndim != 2:
            print(f'Error Data needs to be a 2D Matrix!\nCurrently it is {data.ndim}')
            raise ValueError

        if self.use_gpu:
            if cp.get_array_module(data) == np:
                data = cp.array(data)
        else:
            if cp.get_array_module(data) == cp:
                data = np.array(data)

        test_data_matrix = data[:, None, :]
        exemplars_matrix = self.exemplars[None,:, :]

        if dist_method == 'L1':
            if self.use_gpu:

                pass
            pass
        elif dist_method == 'L2':

            distances = test_data_matrix - exemplars_matrix

            #use GPU Cuda Kernal Acceleration
            if self.use_gpu and self.kernal_accelerate:
                data_distance_from_exemplars = self.xp.zeros((data.shape[0],self.exemplars.shape[0]), dtype=self.exemplars.dtype)
                data_distance_from_exemplars = self.euclidean_dist_kernel(distances, axis=2)
            #numpy cpu /gpu  no kernal calculation
            else:

                data_distance_from_exemplars = (distances*distances)
                data_distance_from_exemplars = data_distance_from_exemplars.sum(axis=2)
                data_distance_from_exemplars =self.xp.sqrt(data_distance_from_exemplars)
        else:
            print(f'Error!!! dist_method needs to be L1 or L2!\nCurrently it is {dist_method}')

        #  https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
        # TODO maybe search
        """""
        !!!!!!!!!!!!
        
        np.unique can return counts of value
        
        !!!!!!!!
        """
        #sort the data for the closests distance
        data_nearest_neighbor_idx = self.xp.argpartition(data_distance_from_exemplars,k, axis=1)
        data_nearest_neighbor_idx = data_nearest_neighbor_idx[:,:k]

        labels_array = self.xp.arange(self.num_classes+1)
        labels_matrix = labels_array[:, None]
        predict_data_nearest_neighbors = self.xp.take(self.classes,data_nearest_neighbor_idx)

        # #one hot encode what each datas neighbors's label counts are
        neighbors_label_matrix = predict_data_nearest_neighbors[:, None, :] == labels_matrix
        neighbors_label_matrix = neighbors_label_matrix.astype('int')
        label_counts = neighbors_label_matrix.sum(axis=2)
        predicted_labels = self.xp.argmax(label_counts,axis=1)
        return predicted_labels
        ''''
         you could also take the indexes after take, do something like eye = np.eye(num_classes), then eye[idxs_matrix] 
         and sum over the first axis
        
        '''

    def accuracy(self, y, y_pred):
        '''Computes accuracy based on percent correct: Proportion of predicted class labels `y_pred`
        that match the true values `y`.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_sams,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_sams,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        float. Between 0 and 1. Proportion correct classification.

        NOTE: Can be done without any loops
        '''

        y = self.checkArrayType(y)
        y_pred = self.checkArrayType(y_pred)

        # use GPU Cuda Kernal Acceleration
        if self.use_gpu and self.kernal_accelerate:
            pass
        # numpy cpu /gpu  no kernal calculation
        else:
            accuracy_array = y == y_pred
            accuracy_array = accuracy_array.astype('int')
            accuracy = accuracy_array.sum()/accuracy_array.size
            return accuracy

    def plot_predictions(self, k, n_sample_pts = 300,x_col = 0,y_col = 1, color_map = palettable.cartocolors.qualitative.Safe_10.mpl_colors ):
        '''Paints the data space in colors corresponding to which class the classifier would
         hypothetically assign to data samples appearing in each region.

        Parameters:
        -----------
        k: int. Determines the neighborhood size of training points around each test sample used to
            make class predictions. In other words, how many training samples vote to determine the
            predicted class of a nearby test sample.
        n_sample_pts: int.
            How many points to divide up the input data space into along the x and y axes to plug
            into KNN at which we are determining the predicted class. Think of this as regularly
            spaced 2D "fake data" that we generate and plug into KNN and get predictions at.

        TODO:
        - Pick a discrete/qualitative color scheme. We suggest, like in the clustering project, to
        use a ColorBrewer color palette. List of possible ones here:
        https://github.com/CartoDB/CartoColor/wiki/CARTOColor-Scheme-Names
            - An example: cartocolors.qualitative.Safe_4.mpl_colors
            - The 4 stands for the number of colors in the palette. For simplicity, you can assume
            that we're hard coding this at 4 for 4 classes.
        - Each ColorBrewer palette is a Python list. Wrap this in a `ListedColormap` object so that
        matplotlib can parse it (already imported above).
        - Make an ndarray of length `n_sample_pts` of regularly spaced points between -40 and +40.
        - Call `np.meshgrid` on your sampling vector to get the x and y coordinates of your 2D
        "fake data" sample points in the square region from [-40, 40] to [40, 40].
            - Example: x, y = np.meshgrid(samp_vec, samp_vec)
        - Combine your `x` and `y` sample coordinates into a single ndarray and reshape it so that
        you can plug it in as your `data` in self.predict.
            - Shape of `x` should be (n_sample_pts, n_sample_pts). You want to make your input to
            self.predict of shape=(n_sample_pts*n_sample_pts, 2).
        - Reshape the predicted classes (`y_pred`) in a square grid format for plotting in 2D.
        shape=(n_sample_pts, n_sample_pts).
        - Use the `plt.pcolormesh` function to create your plot. Use the `cmap` optional parameter
        to specify your discrete ColorBrewer color palette.
        - Add a colorbar to your plot
        '''
        plot_cmap = ListedColormap(color_map, self.num_classes)


        # Get the range of the random points to be plotted on the 2 selected axis
        # of the training data
        feature_x = self.exemplars[:, x_col]
        feature_y = self.exemplars[:, y_col]

        x_min = feature_x.min() * .95
        x_max = feature_x.max() * 1.05
        y_min = feature_y.min() * .95
        y_max = feature_y.max() * 1.05

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_sample_pts),
                             np.linspace(y_min, y_max, n_sample_pts))

        test_data_reshaped = self.xp.c_[xx.ravel(), yy.ravel()]
        prediction_results = self.predict(test_data_reshaped, k = k)


        # Put the result into a color plot
        prediction_results = prediction_results.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(self.getAsNumpy(xx), self.getAsNumpy(yy), self.getAsNumpy(prediction_results), cmap=plot_cmap,shading = 'auto',alpha=0.3)


        # Plot also the training points
        plt.scatter(self.getAsNumpy(feature_x),self.getAsNumpy(feature_y), c=self.getAsNumpy(self.classes), cmap=plot_cmap, edgecolor='k', s=20)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title("2-Class classification (k = %i)" % (10))
        plt.show()
        print('done')


    #helper function to check array type of data
    def checkArrayType(self,data):
        if self.use_gpu:
            if cp.get_array_module(data) == np:
                data = cp.array(data)
        else:
            if cp.get_array_module(data) == cp:
                data = np.array(data)

        return data

    #helper function to get things as numpy
    def getAsNumpy(self,data):
        if cp.get_array_module(data) == cp:
            data = data.get()
        return data

    # helper function to get things as numpy
    def getAsCupy(self, data):
        if cp.get_array_module(data) == np:
            data = cp.array(data)
        return data


    def confusion_matrix(self, y, y_pred, debug = False):
        '''Create a confusion matrix based on the ground truth class labels (`y`) and those predicted
        by the classifier (`y_pred`).

        Recall: the rows represent the "actual" ground truth labels, the columns represent the
        predicted labels.

        Parameters:
        -----------
        y: ndarray. shape=(num_data_samps,)
            Ground-truth, known class labels for each data sample
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted class labels by the model for each data sample

        Returns:
        -----------
        ndarray. shape=(num_classes, num_classes).
            Confusion matrix
        '''
        # To get the number of classes, you can use the np.unique
        # function to identify the number of unique categories in the
        # y matrix.

        if debug:
            start_time = time.time()
        #TODO come back and optimize



        # data_categories = self.xp.unique(self.xp.concatenate((y,y_pred)))
        # data_categories= data_categories[:,None]
        #
        # y_expanded = y[:,None]
        # y_pred_expanded = y_pred[:,None]
        #
        # t1 = y_expanded == data_categories[:,None]
        # t2 = y_pred_expanded == data_categories[:,None]
        # t1 = t1.astype('int')
        # t2 = t2.astype('int')
        # t3 = t1 == t2
        # t3 = t3.astype('int')
        # confustion_matrix1 = t3.sum(axis = 1)-3
        if self.num_classes == None:
            self.num_classes= len(self.xp.unique(self.xp.concatenate((y, y_pred))))

        confusion_matrix = self.xp.zeros((self.num_classes,self.num_classes))
        for i in range(len(y)):
            confusion_matrix[y[i]][y_pred[i]] += 1

        if debug:
            print(f'It Took {start_time - time.time()} to run confusion_matrix()')
        return confusion_matrix