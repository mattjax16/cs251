'''kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
import palettable
import concurrent.futures

class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape



        #TODO ASK MY NOT MAKE A self.dataframe object

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps = data.values.shape[0]
        self.num_features = data.values.shape[1]

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''

        return np.copy(self.data)

    def get_inertia(self):
        return  self.inertia

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        pt_1 = pt_1.reshape(1,pt_1.size)
        pt_2 = pt_2.reshape(1, pt_2.size)
        euclid_dist = np.sqrt(np.sum((pt_1-pt_2)*(pt_1-pt_2),axis=1))
        return euclid_dist[0]

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''

        centroid_dist_array = np.sqrt(np.sum((pt - centroids) * (pt - centroids), axis=1))
        return centroid_dist_array


    def initialize(self, k, init_method = 'range'):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        if init_method == 'range':

            maxs = np.max(self.data,axis = 0)
            mins = np.min(self.data,axis = 0)
            starting_centroids = np.random.uniform(mins,maxs, size = (k,mins.size))
        elif init_method == 'points':

            starting_centroid_point_indicies = np.random.choice(np.arange(self.data.shape[0]), replace = False,size = k)
            starting_centroids = self.data[starting_centroid_point_indicies,:]
        else:
            print(f'Error Method needs to be "range" or "points" currently it is {init_method}')
            raise Exception
            exit()

        return starting_centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, init_method = 'points'):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the difference between all the centroid values from the
        previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the difference
        between every previous and current centroid value is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.k = k

        # - Initialize K-means variables
        self.centroids = self.initialize(k,init_method)


        #do K-means untils distance less than thresh-hold or max ittters reached
        i = 0
        max_centroid_diff = np.inf
        while i < max_iter and max_centroid_diff > tol:
            self.data_centroid_labels = self.update_labels(self.centroids)
            self.inertia = self.compute_inertia()

            new_centroids, centroid_diff = self.update_centroids(k=k, data_centroid_labels=self.data_centroid_labels,
                                                                 prev_centroids=self.centroids)
            self.centroids = new_centroids

            max_centroid_diff = np.max(np.sum(centroid_diff,axis=1))

            # increment i
            i += 1

        return self.inertia, i

        # #TODO maybe update self.dataframe here
        #
        # return self.inertia, max_iter

    def cluster_batch(self, k=2, n_iter=1,  tol=1e-2, max_iter=1000, verbose=False, init_method = 'range'):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''

        # initialize best distance value to a large value
        best_intertia = np.inf
        for i in range(n_iter):
            intertia_kmeans, number_of_iters = self.cluster(k,tol=tol, max_iter=max_iter,verbose=verbose,init_method=init_method)
            if intertia_kmeans < best_intertia:
                best_intertia = intertia_kmeans
                best_centroids = self.centroids
                best_data_labels = self.data_centroid_labels

        self.centroids = best_centroids
        self.data_centroid_labels = best_data_labels

    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        data_distance_from_centroids = np.apply_along_axis(func1d = self.dist_pt_to_centroids,
                                                           axis = 1, arr = self.data, centroids = centroids)

        labels = np.argmin(data_distance_from_centroids, axis = 1)
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''

        new_centroids = []
        centroid_diff = []
        for centroid_label, prev_centroid in zip(np.arange(k), prev_centroids):
            data_group_indicies = np.where(data_centroid_labels == centroid_label)

            data_with_label = np.squeeze(self.data[data_group_indicies,:])

            if data_with_label.size == self.num_features:
                new_centroid = data_with_label
            else:
                new_centroid = data_with_label.mean(axis=0)
            new_centroids.append(new_centroid)
            #TODO maybe no abs for better speed since it is very computationaly intensive
            centroid_diff.append(abs(new_centroid - prev_centroid))

        new_centroids = np.array(new_centroids, dtype= np.float64 )
        centroid_diff = np.array(centroid_diff, dtype= np.float64)
        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''

        sum_dist_of_centroids = []
        for index, centroid in enumerate(self.centroids):
            centroid_sum = np.sum((self.data[np.where(self.data_centroid_labels == index),:] - centroid) *
                   (self.data[np.where(self.data_centroid_labels == index),:] - centroid), axis=1)
            sum_dist_of_centroids.append(centroid_sum)

        return np.sum(np.array(sum_dist_of_centroids))/self.num_samps

    def plot_clusters(self, cmap = palettable.colorbrewer.qualitative.Paired_12.mpl_colormap, title = '' ,x_axis = 0, y_axis = 1, fig_sz = (8,8), legend_font_size = 10):

        '''Creates a scatter plot of the data color-coded by cluster assignment.

        cmap = palettable.colorbrewer.qualitative.Paired_12.mpl_colors

        TODO: FIX THE LEGEND ALSO IF I WAS USING A DATA FRAME COULD USE
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        fig, axes = plt.subplots(1,1,figsize = fig_sz)

        #TODO maybe set up data frame based of of label

        # Set the color map (cmap) to the colorbrewer one
        scat = axes.scatter(self.data[:,x_axis], self.data[:,y_axis], c=self.data_centroid_labels, cmap=cmap)
        # # Show the colorbar
        # cbar = fig.colorbar(scat)
        #
        # # set labels
        # cbar.ax.set_ylabel(c_var, fontsize=20)

        # colors_legend_size = unique_c_vals.size
        color_legend = axes.legend(*scat.legend_elements(), title = 'Groups:', loc = 'best',fontsize = legend_font_size,
                                   title_fontsize = legend_font_size)

        # color_legend = axes.legend(*scat.legend_elements(), bbox_to_anchor=(1.2, 1),
        #                          loc="upper left")
        # frameon = True

        axes.add_artist(color_legend)

        # axes.set_color_cycle(cmap)
        # for group in np.unique(self.data_centroid_labels):
        #
        #     x_data = self.data[self.data_centroid_labels == group,x_axis]
        #     y_data = self.data[self.data_centroid_labels == group, y_axis]
        #     axes.scatter(x_data,y_data,label = f'Group {group}')
        #     axes.set_title(title)
        #     axes.legend([f'Group {i+1}' for i in np.arange(np.unique(self.data_centroid_labels).size)])
        return fig, axes

    def elbow_plot(self, max_k, title = '',fig_sz = (8,8), font_size = 10, cluster_method = 'single', batch_iters = 20):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''

        #set up plot
        fig, axes = plt.subplots(1,1,figsize =fig_sz)

        k_s = np.arange(max_k) + 1
        #do all the k-means
        cluster_results = []
        for i in k_s:
            if cluster_method == 'single':
                cluster_results.append(self.cluster(k=i))
            elif cluster_method == 'batch':
                self.cluster_batch(k = i,n_iter=batch_iters)
                cluster_results.append(self.get_inertia())
            else:
                print(f'Error! cluster_method needs to be single or batch\nCurrently it is {cluster_method}')
                raise ValueError

        cluster_results = np.array(cluster_results)
        k_means_interia = cluster_results[:,0]



        axes.plot(k_s,k_means_interia)
        axes.set_xticks(k_s)
        axes.set_xlabel('Cluster(s)',fontsize = font_size)
        axes.set_ylabel('Inertia')
        axes.set_title(title)
        return fig,axes

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        print(f'enteref replacve with cluster')
