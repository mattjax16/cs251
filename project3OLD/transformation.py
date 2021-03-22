'''transformation.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2021
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
from palettable import cartocolors
import analysis_old
from data import *


class Transformation(analysis_old.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''

        #also I need to make the stuff for all data
        if orig_dataset == None:
            print(f'Error: Transformation Class Needs Data')
        else:
            self.orig_dataset = orig_dataset
            if data != None and data.get_all_data().shape[1] < orig_dataset.get_all_data().shape[1]:
                super().__init__(data)
            else:
                super().__init__(orig_dataset)


    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        if len(headers)<1:
            print(f'Error: Headers needs to be a list of Variables in the original data')
        else:

            headers2col_dict = {}
            headers_indexs = self.orig_dataset.get_header_indices(headers)
            for index , header in enumerate(headers):
                headers2col_dict[header] = index
            self.data = Data(data = self.orig_dataset.select_data(headers= headers), headers=headers, header2col= headers2col_dict)


    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        return (np.hstack((self.data.get_all_data(), np.ones(self.data.get_all_data().shape[0]).reshape(self.data.get_all_data().shape[0],1))))

    def translation_matrix(self, magnitudes):
        ''' Make an M-dimensional homogeneous transformation matrix for translation,
        where M is the number of features in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these
            amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

        NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
        translation!
        '''
        if len(magnitudes) != self.data.get_all_data().shape[1]:
            print(f'Error: Magnitudes needs to be a list of floats length {self.data.get_all_data().shape[1]}')

        else:
            trans_matrix = np.eye(self.data.get_all_data().shape[1]+1)
            trans_matrix[0:-1,-1] = np.array(magnitudes)
            return trans_matrix



    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        # if not isinstance(list, magnitudes) or len(magnitudes) != self.data.get_all_data().shape[1] - 1:
        if len(magnitudes) != self.data.get_all_data().shape[1]:
            print(f'Error: Magnitudes needs to be a list of floats length {self.data.get_all_data().shape[1]}')

        else:
            size = (self.data.get_all_data().shape[1]+1)*(self.data.get_all_data().shape[1]+1)
            scale_matrix = np.zeros(size).reshape((self.data.get_all_data().shape[1]+1),(self.data.get_all_data().shape[1]+1))
            magnitudes.append(1)
            np.fill_diagonal(scale_matrix, magnitudes)
            return scale_matrix

    def translate(self, magnitudes):
        '''Translates the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Translate corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The translated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to translate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        if len(magnitudes) != self.data.get_all_data().shape[1]:
            print(f'Error: Magnitudes needs to be a list of floats length {self.data.get_all_data().shape[1]}')

        else:
            trans_matrix = self.translation_matrix(magnitudes)
            homogenieus_Data =self.get_data_homogeneous()
            return_matrix = (trans_matrix@homogenieus_Data.T).T[:,:-1]
            self.data.data = return_matrix
            return return_matrix

    def scale(self, magnitudes):
        '''Scales the variables `headers` in projected dataset in corresponding amounts specified
        by `magnitudes`.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The scaled data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to scale the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        if len(magnitudes) != self.data.get_all_data().shape[1]:
            print(f'Error: Magnitudes needs to be a list of floats length {self.data.get_all_data().shape[1]}')

        else:
            scale_matrix = self.scale_matrix(magnitudes)
            homogenieus_Data = self.get_data_homogeneous()
            return_matrix = (scale_matrix @ homogenieus_Data.T).T[:, :-1]
            self.data.data = return_matrix
            return return_matrix

    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        -----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''

        if sum(C.shape) != (self.data.get_all_data().shape[1]+1)*2:
            print(f'Error: c shape needs to be {self.data.get_all_data().shape[1]+1} by {self.data.get_all_data().shape[1]+1}')

        else:

            homogenieus_Data = self.get_data_homogeneous()
            return_matrix = (C @ homogenieus_Data.T).T[:, :-1]
            self.data.data = return_matrix
            return return_matrix

    def normalize_together(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        data_range = self.range(self.data.get_headers())
        min = np.min(data_range[0])
        max = np.max(data_range[1])

        translation_matrix = self.translate([-min for x in data_range[0]])
        return_matrix = self.scale([(1/(max-min)) for x in data_range[0]])

        return return_matrix




    def normalize_separately(self):
        '''Normalize each variable separately by translating its local minimum to zero and scaling
        its local range to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        data_range = self.range(self.data.get_headers())


        translation_matrix = self.translate([-min for min in data_range[0]])
        homogenieus_Data = self.get_data_homogeneous().T
        return_matrix = self.scale([(1 / (max - min)) for min,max in zip(data_range[0],data_range[1])])

        return return_matrix

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''

        headers = self.data.get_headers()
        if header in headers:

            #make in radian
            radian = np.deg2rad(degrees)

            cos_rad = np.cos(radian)
            sin_rad = np.sin(radian)
            neg_sin_rad = -np.sin(radian)

            return_array = np.eye(4)
            # X
            if headers.index(header) == 0:
                return_array[1,1] = cos_rad
                return_array[2,2] = cos_rad
                return_array[1, 2] = neg_sin_rad
                return_array[2, 1] = sin_rad
            # Y
            if headers.index(header) == 1:
                return_array[0, 0] = cos_rad
                return_array[3, 3] = cos_rad
                return_array[3, 0] = neg_sin_rad
                return_array[0, 3] = sin_rad
            # Z
            if headers.index(header) == 2:
                return_array[1, 1] = cos_rad
                return_array[0, 0] = cos_rad
                return_array[0, 1] = neg_sin_rad
                return_array[1, 0] = sin_rad

            return return_array
        else:
            print(f"Error: {header} not in headers {self.data.get_headers()}")

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''

        rotation_matrix = self.rotation_matrix_3d(header,degrees)
        return_matrix = self.transform(rotation_matrix)
        self.data.data = return_matrix
        return return_matrix

    def scatter_color(self, ind_var, dep_var, c_var, title=None, alpha = 1, colors = None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        # first I added from palettable import cartocolors


        if ind_var == dep_var or ind_var == c_var or dep_var == c_var:
            print(f"Error: ind_var, dep_car, and c_var must all be different!\nThey are {ind_var} {dep_var} and {c_var} respectivly")
        else:
            # Define ColorBrewer color map palette
            brewer_colors = []
            if colors == None:
                brewer_colors = cartocolors.qualitative.Safe_4.mpl_colormap
            else:
                brewer_colors = colors

            ind_vals = self.data.select_data([ind_var])
            dep_vals = self.data.select_data([dep_var])
            c_vals = self.data.select_data([c_var])

            # Set the color map (cmap) to the colorbrewer one
            scat = plt.scatter(ind_vals, dep_vals, c=c_vals, cmap=brewer_colors, alpha = alpha)
            # Show the colorbar
            cbar = plt.colorbar(scat)

            #set labels
            cbar.ax.set_ylabel(c_var)

            plt.ylabel(dep_var)
            plt.xlabel(ind_var)
            if title != None:
                plt.title(title)

    def heatmap(self, headers=None, title=None, cmap="gray"):
        '''Generates a heatmap of the specified variables (defaults to all). Each variable is normalized
        separately and represented as its own row. Each individual is represented as its own column.
        Normalizing each variable separately means that one color axis can be used to represent all
        variables, 0.0 to 1.0.

        Parameters:
        -----------
        headers: Python list of str (or None). (Optional) The variables to include in the heatmap.
            Defaults to all variables if no list provided.
        title: str. (Optional) The figure title. Defaults to an empty string (no title will be displayed).
        cmap: str. The colormap string to apply to the heatmap. Defaults to grayscale
            -- black (0.0) to white (1.0)

        Returns:
        -----------
        fig, ax: references to the figure and axes on which the heatmap has been plotted
        '''

        # Create a doppelganger of this Transformation object so that self.data
        # remains unmodified when heatmap is done
        data_clone = Data(headers=self.data.get_headers(),
                               data=self.data.get_all_data(),
                               header2col=self.data.get_mappings())
        dopp = Transformation(self.data, data_clone)
        dopp.normalize_separately()

        fig, ax = plt.subplots()
        if title is not None:
            ax.set_title(title)
        ax.set(xlabel="Individuals")

        # Select features to plot
        if headers is None:
            headers = dopp.data.headers
        m = dopp.data.select_data(headers)

        # Generate heatmap
        hmap = ax.imshow(m.T, aspect="auto", cmap=cmap, interpolation='None')

        # Label the features (rows) along the Y axis
        y_lbl_coords = np.arange(m.shape[1]+1) - 0.5
        ax.set_yticks(y_lbl_coords, minor=True)
        y_lbls = [""] + headers
        ax.set_yticklabels(y_lbls)
        ax.grid(linestyle='none')

        # Create and label the colorbar
        cbar = fig.colorbar(hmap)
        cbar.ax.set_ylabel("Normalized Features")

        return fig, ax


    def scatter_color_size(self, ind_var, dep_var, c_var, size_var,title=None, size_scale = 150, colors = None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        # first I added from palettable import cartocolors


        if ind_var == dep_var or ind_var == c_var or ind_var == size_var \
        or dep_var == c_var or dep_var == size_var or c_var == size_var:
            print(f"Error: ind_var, dep_car, c_var and size_var must all be different!\nThey are {ind_var} {dep_var} {c_var} and {size_var}respectivly")
        else:

            #get color map
            brewer_colors = []
            if colors == None:
                brewer_colors = cartocolors.qualitative.Safe_4.mpl_colormap
            else:
                brewer_colors = colors

            ind_vals = self.data.select_data([ind_var])
            dep_vals = self.data.select_data([dep_var])
            c_vals = self.data.select_data([c_var])

            #get unique c vals
            # unique_c_vals = np.unique(c_vals).flatten()

            #used this code to get normalization for side values (first part of code taken from heatmap function)
            # Create a doppelganger of this Transformation object so that self.data
            # remains unmodified when heatmap is done
            data_clone = Data(headers=self.data.get_headers(),
                              data=self.data.get_all_data(),
                              header2col=self.data.get_mappings())
            dopp = Transformation(self.data, data_clone)
            dopp.normalize_separately()

            #make data back to normal
            self.data = data_clone
            size_vals = (dopp.data.select_data([size_var])+1)*size_scale

            fig, ax = plt.subplots()

            # Set the color map (cmap) to the colorbrewer one
            scat = ax.scatter(ind_vals, dep_vals, c=c_vals, s = size_vals, cmap=brewer_colors, alpha = 0.5)
            # Show the colorbar
            cbar = fig.colorbar(scat)

            # set labels
            cbar.ax.set_ylabel(c_var, fontsize = 20)

            # colors_legend_size = unique_c_vals.size
            color_legend = ax.legend(*scat.legend_elements(),bbox_to_anchor = (1.2,1) ,
                                loc="upper left", title=f"{c_var} : Color ")
            # frameon = True

            ax.add_artist(color_legend)


            # make size legend

            #get unnormalized size values
            unnormalized_size_vals = self.data.select_data([size_var])
            unnormalized_invisible_scat = ax.scatter(ind_vals, dep_vals, c=c_vals, s = unnormalized_size_vals, cmap=brewer_colors, alpha = 0)

            invisi_scat_handles , invisi_scat_labels = unnormalized_invisible_scat.legend_elements(prop="sizes", alpha=0.8)


            unique_size_vals = np.unique(unnormalized_size_vals)
            scat_handles, scat_labels = scat.legend_elements(prop="sizes", alpha=0.6, num = len(unique_size_vals))

            # size_legend = ax.legend(*scat.legend_elements(prop = 'sizes', num = len(unique_size_vals), func = lambda s: s ,fmt = f'{s}' ),
            #                         loc="lower right", title=f"{size_var}")

            size_legend = ax.legend(handles = scat_handles, labels = invisi_scat_labels,bbox_to_anchor = (1.2,0.3) ,
                                loc="upper left", title=f"{size_var} : Size")

            ax.set_ylabel(dep_var, fontsize = 20)
            ax.set_xlabel(ind_var, fontsize = 20)
            if title != None:
                ax.set_title(title, fontsize = 25)
            else:
                ax.set_title(f'{dep_var} vs {ind_var}\n(categorized on {c_var} and {size_var})', fontsize = 25)







    def normalize_together_zscore(self, func_performance_type = 'broadcasting'):
        '''Normalize all variables in the projected dataset together by the z-score

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''





        #perform function based on type
        if func_performance_type == 'broadcasting':
            # question which way?

            means = self.mean(self.data.get_headers())
            mean_from_means = np.mean(means)

            data_matrix = self.data.get_all_data()
            mean = np.mean(data_matrix)

            stds_array = self.std(self.data.get_headers())
            sum_std = np.sum(stds_array)

            std_from_numpy = np.std(data_matrix)
            translation_matrix = self.translate([-mean_from_means for mean in means])
            return_matrix = self.scale([(1 / std_from_numpy) for x in means])

            return return_matrix




    def normalize_separately_zscore_vectorized(self, num):
        print(num)


    def normalize_separately_zscore(self, func_performance_type = 'broadcasting'):
        '''Normalize each variable separately by their z-scores

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''

        # perform function based on type
        if func_performance_type == 'broadcasting':
            # question which way?

            means = self.mean(self.data.get_headers())

            data_matrix = self.data.get_all_data()

            stds_array = self.std(self.data.get_headers())

            translation_matrix = self.translate([-mean for mean in means])
            return_matrix = self.scale([(1 / std) for std in stds_array])

            return return_matrix

        # elif func_performance_type == 'vectorized':
        #
        #     data_matrix = self.data.get_all_data()
        #     vectorize_norm_std = np.vectorize(self.normalize_separately_zscore_vectorized)
        #     vectorize_norm_std(data_matrix)
        #
        #
        #
        #
        #
        #
        # data_range = self.range(self.data.get_headers())
        #
        #
        # translation_matrix = self.translate([-min for min in data_range[0]])
        # homogenieus_Data = self.get_data_homogeneous().T
        # return_matrix = self.scale([(1 / (max - min)) for min,max in zip(data_range[0],data_range[1])])

        # return return_matrix
