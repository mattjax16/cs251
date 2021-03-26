'''linear_regression_old.py
Subclass of Analysis that performs linear regression on data
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

import analysis_old


class LinearRegression(analysis_old.Analysis):
    '''
    Perform and store linear regression and related analyses
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: Data object. Contains all data samples and variables in a dataset.
        '''
        super().__init__(data)

        # ind_vars: Python list of strings.
        #   1+ Independent variables (predictors) entered in the regression.
        self.ind_vars = None
        # dep_var: string. Dependent variable predicted by the regression.
        self.dep_var = None

        # A: ndarray. shape=(num_data_samps, num_ind_vars)
        #   Matrix for independent (predictor) variables in linear regression
        self.A = None

        # y: ndarray. shape=(num_data_samps, 1)
        #   Vector for dependent variable predictions from linear regression
        self.y = None

        # R2: float. R^2 statistic
        self.R2 = None

        # slope: ndarray. shape=(num_ind_vars, 1)
        #   Regression slope(s)
        self.slope = None
        # intercept: float. Regression intercept
        self.intercept = None
        # residuals: ndarray. shape=(num_data_samps, 1)
        #   Residuals from regression fit
        self.residuals = None

        # p: int. Polynomial degree of regression model (Week 2)
        self.p = 1










# Took these from Transformation class form last weeks project

    def get_data_homogeneous(self, data_array):
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
        return (np.hstack((data_array, np.ones(data_array.shape[0]).reshape(data_array.shape[0],1))))


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



    def scale_matrix(self, magnitudes, proj = 3):
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

        if proj == 2:
            # if not isinstance(list, magnitudes) or len(magnitudes) != self.data.get_all_data().shape[1] - 1:
            if len(magnitudes) != self.data.get_all_data().shape[1]:
                print(f'Error: Magnitudes needs to be a list of floats length {self.data.get_all_data().shape[1]}')

            else:
                size = (self.data.get_all_data().shape[1] + 1) * (self.data.get_all_data().shape[1] + 1)
                scale_matrix = np.zeros(size).reshape((self.data.get_all_data().shape[1] + 1),
                                                      (self.data.get_all_data().shape[1] + 1))
                magnitudes.append(1)
                np.fill_diagonal(scale_matrix, magnitudes)
                return scale_matrix

        elif proj == 3:
            mags = np.insert(magnitudes,magnitudes.size,1)
            mags = mags.reshape(mags.shape[0],1)
            size = mags.shape[0]
            scale_matrix = np.zeros(size*size).reshape(size,size)
            np.fill_diagonal(scale_matrix, mags)
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













    def linear_regression(self, ind_vars = None, dep_var = None, method='scipy'):
        '''Performs a linear regression on the independent (predictor) variable(s) `ind_vars`
        and dependent variable `dep_var` using the method `method`.

        Parameters:
        -----------
        ind_vars: Python list of strings. 1+ independent variables (predictors) entered in the regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. 1 dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - Use your data object to select the variable columns associated with the independent and
        dependent variable strings.
        - Perform linear regression using the appropriate method.
        - Compute R^2 on the fit and the residuals.
        - By the end of this method, all instance variables should be set (see constructor), except
        for self.adj_R2.

        NOTE: Use other methods in this class where ever possible (do not write the same code twice!)
        '''


        #first I am checking ind_vars and dep_vars and making sure they are entered and exist as headers

        if isinstance(ind_vars, type(None)) or isinstance(dep_var, type(None)):
            print(f'Error: there must be atleast 1 ind_var and dep_var\nRight now they are {ind_vars} and  {dep_var}')
            exit()
        if len(ind_vars) < 1:
            print(f'Error: there must be atleast 1 ind_var')
            exit()


        ind_vars_array = np.array(ind_vars)
        headers_array = np.array(self.data.get_headers())

        if dep_var not in headers_array:
            print(f'Error: dep_var: {dep_var} needs to be in {headers_array}')
            exit()
        for ind_var in ind_vars_array:
            if ind_var not in headers_array:
                print(f'Error: ind_var: {ind_var} needs to be in {headers_array}')
                exit()


        # NOW i AM "Use your data object to select the variable columns associated with the independent and dependent variable strings."

        self.ind_vars = ind_vars
        self.dep_var = dep_var
        self.A = self.data.select_data(self.ind_vars)
        self.y = self.data.select_data([self.dep_var])


        if method == 'scipy':
            c = self.linear_regression_scipy(self.A,self.y)
            self.slope = c[:-1]
            self.intercept = c[-1][0]

        if method == 'normal':
            c = self.linear_regression_normal(self.A, self.y)
            print(c)


        #predict part

        predicted_y = self.predict(self.slope,self.intercept)
        self.residuals =self.compute_residuals(predicted_y)
        self.R2 = self.r_squared(predicted_y)



    def linear_regression_scipy(self, A, y):
        '''Performs a linear regression using scipy's built-in least squares solver (scipy.linalg.lstsq).
        Solves the equation y = Ac for the coefficient vector c.

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var PLUS the intercept term
        '''
        M = self.get_data_homogeneous(A)

        c, res, rnk, s = scipy.linalg.lstsq(M, y)
        return c

    def linear_regression_normal(self, A, y):
        '''Performs a linear regression using the normal equations.
        Solves the equation y = Ac for the coefficient vector c.

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term
        '''



        A = self.get_data_homogeneous(A)

        A_first_part = (A.T@A)
        invA = np.linalg.inv(A_first_part)
        inverse_check = A_first_part@invA

        second_part = (A.T)@(self.y)

        c = invA@second_part

        return c

    def linear_regression_qr(self, A, y):
        '''Performs a linear regression using the QR decomposition

        (Week 2)

        See notebook for a refresher on the equation

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars).
            Data matrix for independent variables.
        y: ndarray. shape=(num_data_samps, 1).
            Data column for dependent variable.

        Returns
        -----------
        c: ndarray. shape=(num_ind_vars+1,)
            Linear regression slope coefficients for each independent var AND the intercept term

        NOTE: You should not compute any matrix inverses! Check out scipy.linalg.solve_triangular
        to backsubsitute to solve for the regression coefficients `c`.
        '''
        pass

    def qr_decomposition(self, A):
        '''Performs a QR decomposition on the matrix A. Make column vectors orthogonal relative
        to each other. Uses the Gram–Schmidt algorithm

        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, num_ind_vars+1).
            Data matrix for independent variables.
        
        Returns:
        -----------
        Q: ndarray. shape=(num_data_samps, num_ind_vars+1)
            Orthonormal matrix (columns are orthogonal unit vectors — i.e. length = 1)
        R: ndarray. shape=(num_ind_vars+1, num_ind_vars+1)
            Upper triangular matrix

        TODO:
        - Q is found by the Gram–Schmidt orthogonalizing algorithm.
        Summary: Step thru columns of A left-to-right. You are making each newly visited column
        orthogonal to all the previous ones. You do this by projecting the current column onto each
        of the previous ones and subtracting each projection from the current column.
            - NOTE: Very important: Make sure that you make a COPY of your current column before
            subtracting (otherwise you might modify data in A!).
        Normalize each current column after orthogonalizing.
        - R is found by equation summarized in notebook
        '''
        pass

    def predict(self, slope, intercept, X=None):
        '''Use fitted linear regression model to predict the values of data matrix `X`.
        Generates the predictions y_pred = mD + b, where (m, b) are the model fit slope and intercept,
        D is the data matrix.

        Parameters:
        -----------
        slope: ndarray. shape=(num_ind_vars, 1)
            Slope coefficients for the linear regression fits for each independent var
        intercept: float.
            Intercept for the linear regression fit
        X: ndarray. shape=(num_data_samps, num_ind_vars).
            If None, use self.A for the "x values" when making predictions.
            If not None, use X as independent var data as "x values" used in making predictions.
        
        Returns
        -----------
        y_pred: ndarray. shape=(num_data_samps,)
            Predicted y (dependent variable) values

        NOTE: You can write this method without any loops!
        '''


        slope_scale_matrix = self.scale_matrix(slope)

        if isinstance(X, type(None)):
            X = self.get_data_homogeneous(self.A).T

        else:
            X = self.get_data_homogeneous(X).T


        return_matrix = (slope_scale_matrix @ X).T[:, :-1].T
        return_matrix = np.sum(return_matrix,axis=0)
        return_matrix = return_matrix + intercept
        return_matrix = return_matrix.T
        return_matrix = return_matrix.reshape(return_matrix.shape[0],1)
        return return_matrix

    def r_squared(self, y_pred):
        '''Computes the R^2 quality of fit statistic

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps,).
            Dependent variable values predicted by the linear regression model

        Returns:
        -----------
        R2: float.
            The R^2 statistic
        '''
        y_mean = np.mean(self.y)

        r2_top = np.subtract(self.residuals, y_mean)
        r2_top = np.square(r2_top)
        r2_top = np.sum(r2_top)



        r2_bottom = self.y - y_mean
        check = np.sum(r2_bottom)
        r2_bottom = np.square(r2_bottom)
        r2_bottom = np.sum(r2_bottom)
        r2 = r2_top/r2_bottom
        r2 = np.sqrt(r2) - 1
        return r2

    def compute_residuals(self, y_pred):
        '''Determines the residual values from the linear regression model

        Parameters:
        -----------
        y_pred: ndarray. shape=(num_data_samps, 1).
            Data column for model predicted dependent variable values.

        Returns
        -----------
        residuals: ndarray. shape=(num_data_samps, 1)
            Difference between the y values and the ones predicted by the regression model at the 
            data samples
        '''
        residuals = self.y - y_pred
        return residuals

    def mean_sse(self, X=None):
        '''Computes the mean sum-of-squares error in the predicted y compared the actual y values.
        See notebook for equation.

        Parameters:
        -----------
        X: ndarray. shape=(anything, num_ind_vars)
            Data to get regression predictions on.
            If None, get predictions based on data used to fit model.

        Returns:
        -----------
        float. Mean sum-of-squares error

        Hint: Make use of self.compute_residuals
        '''
        pass

    def scatter(self, ind_var, dep_var, title = None, ind_var_index=0):
        '''Creates a scatter plot with a regression line to visualize the model fit.
        Assumes linear regression has been already run.
        
        Parameters:
        -----------
        ind_var: string. Independent variable name
        dep_var: string. Dependent variable name
        title: string. Title for the plot
        ind_var_index: int. Index of the independent variable in self.slope
            (which regression slope is the right one for the selected independent variable
            being plotted?)
            By default, assuming it is at index 0.

        TODO:
        - Use your scatter() in Analysis to handle the plotting of points. Note that it returns
        the (x, y) coordinates of the points.
        - Sample evenly spaced x values for the regression line between the min and max x data values
        - Use your regression slope, intercept, and x sample points to solve for the y values on the
        regression line.
        - Plot the line on top of the scatterplot.
        - Make sure that your plot has a title (with R^2 value in it)
        '''

        #- Use your scatter() in Analysis to handle the plotting of points.

        ### make even samples between



        if isinstance(title, type(None)):
            title = f'R^2 : {self.R2}'
        else:
            title = f'{title}\nR^2 : {self.R2}'

        x_cords, y_cords = analysis_old.Analysis.scatter(self, ind_var=ind_var, dep_var=dep_var, title=title)

        # - Use your regression slope, intercept, and
        # x sample points to solve for the y values on the regression line.
        # question why not just add self.residuals and self.y
        predicted_y = np.subtract(self.y,self.residuals)

        plt.plot(x_cords,predicted_y,label = f'linear reg line', c = 'r', alpha = 0.5)
        plt.legend(bbox_to_anchor = (1.26,0.1),loc = 'upper right')

        return x_cords





    #helper function for pair_plot to create linear regressions
    def pair_plot_linear_regs(self,ndenumerate_obj):

        ax = ndenumerate_obj[1]
        dependant_var = ndenumerate_obj[0][0]
        independant_var = ndenumerate_obj[0][1]
        headers = self.data.get_headers()
        x = headers[independant_var]
        y = headers[dependant_var]
        self.linear_regression(ind_vars = [y],dep_var=x)


        x_cords = self.data.select_data([x])


        # question why not just add self.residuals and self.y
        predicted_y = np.subtract(self.y, self.residuals)
        ax.plot(x_cords, predicted_y, label=f'linear reg line', c='r', alpha=0.5)
        ax.legend()

        return ax




    def pair_plot(self, data_vars, fig_sz=(12, 12), title = None):
        '''Makes a pair plot with regression lines in each panel.
        There should be a len(data_vars) x len(data_vars) grid of plots, show all variable pairs
        on x and y axes.

        Parameters:
        -----------
        data_vars: Python list of strings. Variable names in self.data to include in the pair plot.
        fig_sz: tuple. len(fig_sz)=2. Width and height of the whole pair plot figure.
            This is useful to change if your pair plot looks enormous or tiny in your notebook!

        TODO:
        - Use your pair_plot() in Analysis to take care of making the grid of scatter plots.
        Note that this method returns the figure and axes array that you will need to superimpose
        the regression lines on each subplot panel.
        - In each subpanel, plot a regression line of the ind and dep variable. Follow the approach
        that you used for self.scatter. Note that here you will need to fit a new regression for
        every ind and dep variable pair.
        - Make sure that each plot has a title (with R^2 value in it)
        '''

        if isinstance(title, type(None)):
            title = f'R^2 : {self.R2}'
        else:
            title = f'{title}\nR^2 : {self.R2}'

        fig, axes = analysis_old.Analysis.pair_plot(self, data_vars = data_vars, fig_sz=fig_sz, title=title)

        # here I am mappinng the regression lines onto all the scatter plots in the pair plot
        # with the function I made pair_plot_linear_regs()
        axes_list = np.array(list((map(lambda ax: self.pair_plot_linear_regs(ax),
                                        np.ndenumerate(axes)))))
        axes_array = axes_list
        axes = axes_array.reshape(len(data_vars), len(data_vars))
        return

    def make_polynomial_matrix(self, A, p):
        '''Takes an independent variable data column vector `A and transforms it into a matrix appropriate
        for a polynomial regression model of degree `p`.
        
        (Week 2)

        Parameters:
        -----------
        A: ndarray. shape=(num_data_samps, 1)
            Independent variable data column vector x
        p: int. Degree of polynomial regression model.

        Returns:
        -----------
        ndarray. shape=(num_data_samps, p)
            Independent variable data transformed for polynomial model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10.

        NOTE: There should not be a intercept term ("x^0"), the linear regression solver method
        will take care of that.
        '''
        pass

    def poly_regression(self, ind_var, dep_var, p, method='normal'):
        '''Perform polynomial regression — generalizes self.linear_regression to polynomial curves
        
        (Week 2)
        
        NOTE: For single linear regression only (one independent variable only)

        Parameters:
        -----------
        ind_var: str. Independent variable entered in the single regression.
            Variable names must match those used in the `self.data` object.
        dep_var: str. Dependent variable entered into the regression.
            Variable name must match one of those used in the `self.data` object.
        p: int. Degree of polynomial regression model.
            Example: if p=10, then the model should have terms in your regression model for
            x^1, x^2, ..., x^9, x^10
            (The method that you call for the linear regression solver will take care of the intercept)
        method: str. Method used to compute the linear regression. Here are the options:
            'scipy': Use scipy's linregress function.
            'normal': Use normal equations.
            'qr': Use QR factorization (linear algebra section only).

        TODO:
        - This method should mirror the structure of self.linear_regression (compute all the same things)
        - Differences are:
            - You create the independent variable data matrix (self.A) with columns appropriate for
            polynomial regresssion. Do this with self.make_polynomial_matrix
            - You should programatically generate independent variable name strings based on the
            polynomial degree.
                Example: ['X_p1, X_p2, X_p3'] for a cubic polynomial model
            - You set the instance variable for the polynomial regression degree (self.p)
        '''
        pass
