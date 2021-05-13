'''naive_bayes_multinomial.py
Naive Bayes classifier with Multinomial likelihood for discrete features
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import time

# For GPU processing
import cupy as cp


class NaiveBayes:
    '''Naive Bayes classifier using Multinomial likeilihoods (discrete data belonging to any
     number of classes)'''
    def __init__(self, num_classes, use_gpu = False,kernal_accelerate = True):
        '''Naive Bayes constructor

        TODO:
        - Add instance variable for `num_classes`
        '''
        # class_priors: ndarray. shape=(num_classes,).
        #   Probability that a training example belongs to each of the classes
        #   For spam filter: prob training example is spam or ham
        self.class_priors = None
        # class_likelihoods: ndarray. shape=(num_classes, num_features).
        #   Probability that each word appears within class c
        self.class_likelihoods = None

        self.num_classes = num_classes

        # Use GPU processing or not
        self.use_gpu = use_gpu

        #which library to use
        if use_gpu:
            self.xp = cp
        else:
            self.xp = np

        # wether to use the custom Cuda Kernals for calculations
        self.kernal_accelerate = kernal_accelerate

        # Making Cuda Kernal Functions for increased speed on gpu
        # learned how to thanks to Cupy documentation!
        # https://readthedocs.org/projects/cupy/downloads/pdf/stable/

        #https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/#:~:text=CUDA%20kernels%20are%20subdivided%20into,of%20threads%20(Figure%202).

        # gets the sum of a matrix based off of one hot encoding
        self.sum_kernal = cp.ReductionKernel(
            in_params='T x, S oneHotCode', out_params='T result',
            map_expr='oneHotCode ? x : 0.0', reduce_expr='a + b', post_map_expr='result = a', identity='0',
            name='sum_kernal'
        )

        # gets the count of a matrix from one hot encoding (by booleans)
        # TODO make a class variable to hold data type of data set
        self.count_kernal = cp.ReductionKernel(
            in_params='T oneHotCode', out_params='T result',
            map_expr='oneHotCode ? 1.0 : 0.0', reduce_expr='a + b', post_map_expr='result = a', identity='0',
            name='count_kernal'
        )

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



    # helper function to get the total counts of each feature
    # for all samples in each c
    def calculate_liklihoods(self,data,y,debug = False ):
        if debug:
            start_time = time.time()


        num_classes_array = self.xp.arange(self.num_classes)
        label_one_hot = y == num_classes_array[:,None]
        sum_mask = label_one_hot[:, None,:]
        if self.use_gpu and self.kernal_accelerate:
            pass
        else:

            #make booleans 0s and 1s to be multiplied
            sum_mask = sum_mask.astype('int')

            grouped_data = sum_mask*data[:,:,None].T

            total_count_of_freat_per_group = (self.xp.sum(grouped_data,axis=2)).T
            st1 = self.xp.sum(grouped_data,axis=(1,2))
            st11 =  self.xp.sum(grouped_data,axis=(2,1))
            st2 = (self.xp.sum(grouped_data,axis=(2,1)))+ data.shape[1]
            class_likelihoods_matrix = ((total_count_of_freat_per_group+1))/(self.xp.sum(grouped_data,axis=(2,1)) + data.shape[1])
            class_likelihoods_matrix = class_likelihoods_matrix.T
        if debug:
            print(f'It Took {start_time - time.time()} to run calculate_liklihoods()')

        return class_likelihoods_matrix



    def get_class_count_of_data(self,y,debug = False):
        if debug:
            start_time = time.time()

        # Get Class Priors
        num_classes_array = self.xp.arange(self.num_classes)
        label_one_hot = y == num_classes_array[:,None]
        if self.use_gpu and self.kernal_accelerate:
            pass
        else:
            label_one_hot = label_one_hot.astype('int')
            class_counts = label_one_hot.sum(axis=1)


        if debug:
            print(f'It Took {start_time - time.time()} to run get_class_count_of_data()')

        return class_counts


    def train(self, data, y, debug = False, use_log = False):
        '''Train the Naive Bayes classifier so that it records the "statistics" of the training set:
        class priors (i.e. how likely an email is in the training set to be spam or ham?) and the
        class likelihoods (the probability of a word appearing in each class — spam or ham)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features). Data to learn / train on.
        y: ndarray. shape=(num_samps,). Corresponding class of each data sample.

        TODO:
        - Compute the instance variables self.class_priors and self.class_likelihoods needed for
        Bayes Rule. See equations in notebook.
        '''
        if debug:
            start_time = time.time()

        data = self.checkArrayType(data)
        y = self.checkArrayType(y)

        #lapaice smooth data
        # data = data+1

        # set up y to be used for 1 hot encoding
        y = self.xp.expand_dims(y,-1).T
        num_samps, num_features = data.shape

        # Get Class Priors
        class_counts = self.get_class_count_of_data(y)
        self.class_priors = class_counts/y.size

        # set  the class_likelihoods
        self.class_likelihoods = self.calculate_liklihoods(data,y)
        if use_log:
            self.class_likelihoods = self.xp.log(self.class_likelihoods)

        if debug:
            print(f'It Took {start_time - time.time()} to run train()')


    def predict(self, data,debug = False):
        '''Combine the class likelihoods and priors to compute the posterior distribution. The
        predicted class for a test sample from `data` is the class that yields the highest posterior
        probability.

        Parameters:
        -----------
        data: ndarray. shape=(num_test_samps, num_features). Data to predict the class of
            Need not be the data used to train the network

        Returns:
        -----------
        ndarray of nonnegative ints. shape=(num_samps,). Predicted class of each test data sample.

        TODO:
        - Process test samples one-by-one.
        - For each, we want to compute the log of the numerator of the posterior:
        - (a) Use matrix-vector multiplication (or the dot product) with the log of the likelihoods
          and the test sample (transposed, but no logarithm taken)
        - (b) Add to it the log of the priors
        - Predict the class of each test sample according to the class that produces the largest
        log(posterior) probability (use argmax)
        '''
        if debug:
            start_time = time.time()

        data = self.checkArrayType(data)

        num_test_samps, num_features = data.shape

        #set up data for prediction
        data1 = data[:,None,:]


        if self.use_gpu and self.kernal_accelerate:
            pass
        else:
            log_priors = self.xp.log(self.class_priors)
            log_class_likelihoods = self.xp.log(self.class_likelihoods)
            p0 = log_class_likelihoods[None,:,:].T
            p1 = data1 * log_class_likelihoods[None,:,:]
            p2 = self.xp.sum(p1,axis=2)
            log_posteriers = log_priors + p2
            predicted_labels = self.xp.argmax(log_posteriers,axis = 1)
            predicted_labels = predicted_labels
        if debug:
            print(f'It Took {start_time - time.time()} to run predict()')
        return predicted_labels

    def accuracy(self, y, y_pred , debug = False):
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


        acc = self.xp.sum(y_pred == y)/y.size


        if debug:
            print(f'It Took {start_time - time.time()} to run accuracy()')

        return acc
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