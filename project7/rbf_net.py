'''rbf_net.py
Radial Basis Function Neural Network
YOUR NAME HERE
CS 252: Mathematical Data Analysis Visualization, Spring 2021
'''
import numpy as np
from kmeansGPU import KMeansGPU
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import palettable
import concurrent.futures
import pandas as pd
import cupy as cp
import time
from linear_regression_gpu import LinearRegression

class RBF_Net:
    def __init__(self, num_hidden_units, num_classes, use_gpu = True):
        '''RBF network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset

        TODO:
        - Define number of hidden units as an instance variable called `k` (as in k clusters)
            (You can think of each hidden unit as being positioned at a cluster center)
        - Define number of classes (number of output units in network) as an instance variable
        '''

        #TODO maybe make a check to make sure the num_hidden_units is not smaller than
        #   the number of classes needing to be predicted
        self.k = num_hidden_units

        self.num_output_units = num_classes


        # prototypes: Hidden unit prototypes (i.e. center)
        #   shape=(num_hidden_units, num_features)
        self.prototypes = None



        # sigmas: Hidden unit sigmas: controls how active each hidden unit becomes to inputs that
        # are similar to the unit's prototype (i.e. center).
        #   shape=(num_hidden_units,)
        #   Larger sigma -> hidden unit becomes active to dissimilar inputs
        #   Smaller sigma -> hidden unit only becomes active to similar inputs
        self.sigmas = None




        # wts: Weights connecting hidden and output layer neurons.
        #   shape=(num_hidden_units+1, num_classes)
        #   The reason for the +1 is to account for the bias (a hidden unit whose activation is always
        #   set to 1).
        self.wts = None



        # holds wether gpu is being used or not
        self.use_gpu = use_gpu

        # holds whether the array in a numpy or cumpy array
        if use_gpu:
            self.xp = cp
        else:
            self.xp = np


        # GPU accelerate Cuda kernals
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


    #helper functions for gpu acceleration
    def checkArrayType(self, data):
        if self.use_gpu:
            if cp.get_array_module(data) == np:
                data = cp.array(data)
        else:
            if cp.get_array_module(data) == cp:
                data = np.array(data)

        return data

        # helper function to get things as numpy

    def getAsNumpy(self, data):
        if cp.get_array_module(data) == cp:
            data = data.get()
        return data

        # helper function to get things as numpy

    def getAsCupy(self, data):
        if cp.get_array_module(data) == np:
            data = cp.array(data)
        return data




    def get_prototypes(self):
        '''Returns the hidden layer prototypes (centers)

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, num_features).
        '''
        return self.prototypes



    def get_num_hidden_units(self):
        '''Returns the number of hidden layer prototypes (centers/"hidden units").

        Returns:
        -----------
        int. Number of hidden units.
        '''
        return self.k

    def get_num_output_units(self):
        '''Returns the number of output layer units.

        Returns:
        -----------
        int. Number of output units
        '''
        return self.num_output_units

    def avg_cluster_dist(self, data, centroids, cluster_assignments, kmeans_obj, debug = False):
        '''Compute the average distance between each cluster center and data points that are
        assigned to it.

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        centroids: ndarray. shape=(k, num_features). Centroids returned from K-means.
        cluster_assignments: ndarray. shape=(num_samps,). Data sample-to-cluster-number assignment from K-means.
        kmeans_obj: KMeans. Object created when performing K-means.

        Returns:
        -----------
        ndarray. shape=(k,). Average distance within each of the `k` clusters.

        Hint: A certain method in `kmeans_obj` could be very helpful here!
        '''
        #TODO update numpu version of kmeans numpy update centroids
        if debug:
            start_time = time.time()


        # #make sure all arrays are correct type (cupy or numpy)
        # data = self.checkArrayType(data)
        # centroids = self.checkArrayType(centroids)
        # cluster_assignments = self.checkArrayType(cluster_assignments)
        kmeans_obj.set_data(data)
        kmeans_obj.centroids = centroids
        kmeans_obj.data_centroid_labels = cluster_assignments

        avg_cluster_dist = kmeans_obj.compute_inertia(get_mean_dist_per_centroid=True)[1]
        return avg_cluster_dist

    def initialize(self, data, n_iter=10,  tol=1e-2, max_iter=100, init_method = 'points',distance_calc_method = 'L2',debug = False,):
        '''Initialize hidden unit centers using K-means clustering and initialize sigmas using the
        average distance within each cluster

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        TODO:
        - Determine `self.prototypes` (see constructor for shape). Prototypes are the centroids
        returned by K-means. It is recommended to use the 'batch' version of K-means to reduce the
        chance of getting poor initial centroids.
            - To increase the chance that you pick good centroids, set the parameter controlling the
            number of iterations > 1 (e.g. 5)
        - Determine self.sigmas as the average distance between each cluster center and data points
        that are assigned to it. Hint: You implemented a method to do this!
        '''
        if debug:
            start_time = time.time()

        #check data is right type (cupy or numpy)
        data = self.checkArrayType(data)

        #make a kmeans object
        kmeans_obj = KMeansGPU(data,use_gpu=self.use_gpu)

        #run batch clustering on data
        kmeans_obj.cluster_batch(k = self.k,n_iter=n_iter,tol=tol,max_iter=max_iter,verbose=debug,
                                 init_method=init_method,distance_calc_method=distance_calc_method)

        self.prototypes = kmeans_obj.get_centroids()
        comput_res = kmeans_obj.compute_inertia(get_mean_dist_per_centroid=True)
        self.sigmas = comput_res[1]

        if debug:
            print(f'It Took {start_time - time.time()} to run initialize()')

    def linear_regression(self, A, y, debug = False,method='qr'):
        '''Performs linear regression
        CS251: Adapt your SciPy lstsq code from the linear regression project.
        CS252: Adapt your QR-based linear regression solver

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_features).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_features+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: Remember to handle the intercept ("homogenous coordinate")
        '''
        if debug:
            start_time = time.time()

        #check array types correct
        A = self.checkArrayType(A)
        y = self.checkArrayType(y)

        #make linear regression object
        lin_reg_obj = LinearRegression(data=list(A))


        if method == 'scipy':
            c = lin_reg_obj.linear_regression_scipy(A,y)
        elif method == 'normal':
            c = lin_reg_obj.linear_regression_normal(A,y)
        elif method == 'qr':
            c = lin_reg_obj.linear_regression_qr(A,y)



        if debug:
            print(f'It Took {start_time - time.time()} to run linear_regression()')

        return c

    def hidden_act(self, data, debug = False, epsilon= 1e-8 ):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation
        '''
        if debug:
            start_time = time.time()


        #check data type input
        data = self.checkArrayType(data)
        prototypes = self.checkArrayType(self.prototypes)
        sigmas = self.checkArrayType(self.sigmas)
        weights = self.checkArrayType(self.wts)

        data_matrix = data[:,None,:]
        prototypes_matrix = prototypes[None,:,:]

        dist_matrix = data_matrix - prototypes_matrix
        dist_matrix = dist_matrix * dist_matrix
        dist_matrix = dist_matrix.sum(axis=2)

        sigmas_squared = sigmas * sigmas
        sigmas_squared = sigmas_squared*2
        sigmas_squared = sigmas_squared + epsilon

        hidden_acts = self.xp.exp(-(dist_matrix/sigmas_squared))

        if debug:
            print(f'It Took {start_time - time.time()} to run hidden_act()')
        return hidden_acts


    def output_act(self, hidden_acts, debug = False):
        '''Compute the activation of the output layer units

        Parameters:
        -----------
        hidden_acts: ndarray. shape=(num_samps, k).
            Activation of the hidden units to each of the data samples.
            Does NOT include the bias unit activation.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_units).
            Activation of each unit in the output layer to each of the data samples.

        NOTE:
        - Assumes that learning has already taken place
        - Can be done without any for loops.
        - Don't forget about the bias unit!
        '''
        if debug:
            start_time = time.time()

        #get right data types
        hidden_acts = self.checkArrayType(hidden_acts)

        wts = self.checkArrayType(self.wts)

        #add ones
        hidden_acts = self.xp.hstack((hidden_acts,self.xp.ones((hidden_acts.shape[0],1))))


        output_activation = hidden_acts @ wts

        if debug:
            print(f'It Took {start_time - time.time()} to run output_act()')
        return output_activation
    def train(self, data, y, debug = False):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the correct classes of each training sample. To solve for the weights going FROM all of the
        hidden units TO output unit c, recode the class vector `y` to 1s and 0s:
            1 if the class of a data sample in `y` is c
            0 if the class of a data sample in `y` is not c

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.
        '''
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run train()')

    def predict(self, data, debug = False):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each data sample.

        TODO:
        - Pass the data thru the network (input layer -> hidden layer -> output layer).
        - For each data sample, the assigned class is the index of the output unit that produced the
        largest activation.
        '''
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run predict()')

    def accuracy(self, y, y_pred, debug = False):
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
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run accuracy()')


class RBF_Reg_Net(RBF_Net):
    '''RBF Neural Network configured to perform regression
    '''
    def __init__(self, num_hidden_units, num_classes, h_sigma_gain=5):
        '''RBF regression network constructor

        Parameters:
        -----------
        num_hidden_units: int. Number of hidden units in network. NOTE: does NOT include bias unit
        num_classes: int. Number of output units in network. Equals number of possible classes in
            dataset
        h_sigma_gain: float. Multiplicative gain factor applied to the hidden unit variances

        TODO:
        - Create an instance variable for the hidden unit variance gain
        '''
        super().__init__(num_hidden_units, num_classes)

    def hidden_act(self, data, debug = False):
        '''Compute the activation of the hidden layer units

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.

        Returns:
        -----------
        ndarray. shape=(num_samps, k).
            Activation of each unit in the hidden layer to each of the data samples.
            Do NOT include the bias unit activation.
            See notebook for refresher on the activation equation

        TODO:
        - Copy-and-paste your classification network code here.
        - Modify your code to apply the hidden unit variance gain to each hidden unit variance.
        '''
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run Reg_Net hidden_act()')

    def train(self, data, y, debug = False):
        '''Train the radial basis function network

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        Goal: Set the weights between the hidden and output layer weights (self.wts) using
        linear regression. The regression is between the hidden layer activation (to the data) and
        the desired y output of each training sample.

        Notes:
        - Remember to initialize the network (set hidden unit prototypes and sigmas based on data).
        - Pay attention to the shape of self.wts in the constructor above. Yours needs to match.
        - The linear regression method handles the bias unit.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to perform regression on
        the actual y values instead of the y values that match a particular class. Your code should be
        simpler than before.
        - You may need to squeeze the output of your linear regression method if you get shape errors.
        '''
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run Reg_Net train()')

    def predict(self, data, debug = False):
        '''Classify each sample in `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to predict classes for.
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray. shape=(num_samps, num_output_neurons). Output layer neuronPredicted "y" value of
            each sample in `data`.

        TODO:
        - Copy-and-paste your classification network code here, modifying it to return the RAW
        output neuron activaion values. Your code should be simpler than before.
        '''
        if debug:
            start_time = time.time()

        if debug:
            print(f'It Took {start_time - time.time()} to run Reg_Net predict()')
