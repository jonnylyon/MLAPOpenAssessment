from __future__ import division

import numpy as np
import scipy as sp
from numpy import matrix, mean, sum, std, nditer, transpose, dot, ndarray, reshape, multiply
from sys import float_info
from math import exp, log
from scipy.optimize import fmin_bfgs

from random import shuffle, randint
from copy import deepcopy

import os
import csv

# This list is the feature inclusion list. It is used to specify which features are
# to be used, and to what powers.  The 8 features are specified as shown
# in the comments.  If you set the first value (last stock volume) to 3,
# then the last stock volume will be included raised to its first, second
# and third powers.  Similarly, if you set the sixth value (last change in
# stock price) to 1, then it will be included but only raised to its first
# power.  Features left as 0 will not be included at all.  The constant
# term is always included.
inclusion_list = []
inclusion_list.append(0) # last stock volume
inclusion_list.append(0) # last change in stock volume
inclusion_list.append(0) # mean of prev 10 rows stock volume
inclusion_list.append(0) # std dev of prev 10 rows stock volume
inclusion_list.append(0) # last stock price
inclusion_list.append(0) # last change in stock price
inclusion_list.append(1) # mean of prev 10 rows stock price
inclusion_list.append(1) # std dev of prev 10 rows stock price

# This is an arbitrary output file, for troubleshoot purposes
fp_out = os.path.join(os.path.dirname(__file__), 'stock_price_mod.csv')

#####################################################################################################
# This file is in 5 main sections, split by prominent comments such like this.
#       - Generic code (for all tasks)
#       - Linear regression code (task 1)
#       - Regularised linear regression code (task 3a) (extends on task 1 code)
#       - Logistic regression code (task 2)
#       - Regularised logistic regression code (task 3b) (extends on task 2 code)
#####################################################################################################

#####################################################################################################
# This block contains common code that is used by more than one task
# For example: reading the file, splitting the data into CV folds,
# Calculating features from the data and appending them
#####################################################################################################

class FeatureExpander:
    """ This class is used to expand the features which are already appended
        to the data, according to some inclusion list.  So if the inclusion
        list says to use the first feature up to power 1, and the second feature
        to power 2, it will append a vector of the values of
        [1, feature1, feature2, feature2-squared] to each row
        (1 is included as the constant term, which is always included)
    """
        
    def __init__(self, data):
        self.data = data
        
    def append_feature_expansions(self, data, inclusion_list):
        """ According to the options in the inclusion list, this method
            appends the appropriate expansions to the data given
        """
        for i in range(len(data)):
            expanded = []
            
            # Always append 1, as the constant term
            expanded.append(1)
            
            # Append the features to the powers specified
            # in the inclusion list
            for j in range(len(inclusion_list)):
                for k in range(1, inclusion_list[j] + 1):
                    expanded.append(data[i][-1][j] ** k)
            
            data[i].append(matrix(expanded))
        
        return data
    
    def expand_features(self, inclusion_list):
        """ Wrapper method for append_feature_expansions above
        """
        expanded = self.append_feature_expansions(deepcopy(self.data), inclusion_list)
        
        return expanded

def append_features(source_data):
    """ Given the source data, this method appends a list of the features to each row.
        These features are hardcoded as:
            - last stock volume
            - last change in stock volume
            - mean of previous 10 days stock volume
            - standard deviation of previous 10 days stock volume
            - last stock price
            - last change in stock price
            - mean of previous 10 days stock price
            - standard deviation of previous 10 days stock price
    """
    
    result_data=[]
    
    # loop over the data provided, starting from row 10
    # (missing rows 0 to 9, as we don't need to append
    # features to those rows)
    for i in range(10, len(source_data)):
        # add feature matrix in next column
        this_row = source_data[i]
        feature_list = []
        
        # Get lists of the previous ten days' stock volume and price
        prev_10_sv = [row[0] for row in source_data[i-10:i]]
        prev_10_sp = [row[1] for row in source_data[i-10:i]]
        
        # Make a list of the features for this row...
        # last stock volume
        feature_list.append(prev_10_sv[9])
        
        # last change in stock volume
        feature_list.append(prev_10_sv[9] - prev_10_sv[8])
        
        # mean of prev 10 rows stock volume
        feature_list.append(mean(prev_10_sv))
        
        # std dev of prev 10 rows stock volume
        feature_list.append(std(prev_10_sv))
        
        # last stock price
        feature_list.append(prev_10_sp[9])
        
        # last change in stock price
        feature_list.append(prev_10_sp[9] - prev_10_sp[8])
        
        # mean of prev 10 rows stock price
        feature_list.append(mean(prev_10_sp))
        
        # std dev of prev 10 rows stock price
        feature_list.append(std(prev_10_sp))

        # Append the feature list to the row
        this_row.append(feature_list)
        result_data.append(this_row)
    return result_data

def split_data(all_data):
    """ Splits the data into two random folds of approximately equal size
    """
    
    shuffle(all_data)
    split = int(len(all_data)/2)
    return [all_data[:split], all_data[split:]]

def split_data_3_folds(all_data):
    """ Splits the data into three random folds containing approximately
        2/5, 2/5 and 1/5 of the data
    """
    
    shuffle(all_data)
    split_1 = int(2 * len(all_data)/5)
    split_2 = 2 * split_1
    
    return [all_data[:split_1], all_data[split_1:split_2], all_data[split_2:]]
    
def normalise(data):
    """ Normalises the stock volume and stock price columns of the given data
        by setting to mean to 0 and standard deviation to 1 for each column
    """
    all_sv = [row[0] for row in data]
    all_sp = [row[1] for row in data]
    
    sv_std = std(all_sv)
    sv_mean = mean(all_sv)
    
    sp_std = std(all_sp)
    sp_mean = mean(all_sp)
    
    for row in data:
        row[0] = (row[0] - sv_mean) / sv_std
        row[1] = (row[1] - sp_mean) / sp_std
    
    return data

def sign(THETA):
    """ Given a THETA matrix, returns a matrix of the same shape representing
        whether each value in THETA is positive (1), zero (0) or negative (-1)
    """
    shape = THETA.shape
    out = [0 if theta == 0 else theta / abs(theta) for theta in nditer(THETA)]
    out = matrix(out)
    out = np.reshape(out, shape)
    return out

def write_to_file(data, filepath):
    """ Writes some data to a file, as specified, in CSV-ish format.
        Output format isn't very good here, but it's enough for troubleshooting
    """
    lines = []
    for row in data:
        cells = []
        for item in row:
            cells.append(str(item))
    
        line = ",".join(cells)
        lines.append(line + "\n")
        
    file = open(filepath, 'w')
    file.writelines(lines)
    file.close()

def load_data(filepath):
    """ Reads data from a given CSV file
    """
    return [[float(item) for item in row] for row in csv.reader(open(filepath, "rb"))]
    
#####################################################################################################
# This block contains the code that was implemented for task 1 (Linear regression)
#####################################################################################################
    
def generate_squared_losses(THETA, data):
    """ Returns a list of the squared losses for each row in the data,
        when the THETA is applied to the expanded features
    """
    squared_losses = ndarray(shape=(len(data),1))
    
    # for each row in the data, use theta to calculate the squared loss
    # and add it to the list of squared losses, that is finally returned
    for i in range(len(data)):
        squared_losses[i] = (data[i][1] - dot(data[i][3],THETA))**2
    
    return squared_losses
    
def total_squared_loss(THETA, data):
    """ The sum of the squared losses (not average)
        Used for BFGS linear regression minimisation
    """
    return sum(generate_squared_losses(THETA, data))

def linear_gradient(THETA, data):
    """ Calculates gradient of linear regression for given theta and data
    """
    GRAD = []
    
    # initialise gradient values to 0
    for i in range(len(THETA)):
        GRAD.append(0.0)
    
    # iterate over data rows
    for i in range(len(data)):
        X_i = data[i][3] # this is the expanded features list
        sp_computed = dot(X_i, THETA) # get computed stock price
        sp_actual = data[i][1] # actual stock price
        
        # calculate error
        error = (sp_computed - sp_actual)[0,0]

        # update all gradient values
        for j in range(len(THETA)):
            GRAD[j] += error * X_i[0,j]
    
    # return as a scipy array (required data type)
    return sp.array(GRAD)

def linear_regression(data):
    """ Initialise theta to a load of zeros.
        Perform BFGS minimisation on the squared loss to get the optimal THETA
    """
    length_of_expansion = data[0][3].shape[1]
    
    initial_THETA = matrix(np.zeros([length_of_expansion,1]))
    
    # Call optimisation function on squared loss function with gradient function specified
    return fmin_bfgs(total_squared_loss, initial_THETA, fprime=linear_gradient, args=[data])

def evaluate_linear(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
    """ Calculates the MSE for THETA_CV1 on data_CV2
        and THETA_CV2 on data_CV1
        Returns the average of both
    """
    mse_CV1 = calc_mse(data_CV1, THETA_CV2)
    mse_CV2 = calc_mse(data_CV2, THETA_CV1)
    return (mse_CV1 + mse_CV2) / 2
    
def calc_mse(data, THETA):
    """ Calculates the MSE using the given theta on the data
    """
    
    # get a list of the squared losses
    squared_losses = generate_squared_losses(THETA, data)
    
    # calculate the average squared loss from the list
    return sum(squared_losses) / len(squared_losses)
        
def linear(InputFileName):
    """ This method performs linear regression to find the best set of
        coefficients for the features specified in the inclusion list
        at the top of this file.
    """
    
    # load the data from the file
    raw_data = load_data(InputFileName)
    
    # append all of the features to the data (including those not used)
    training_data = append_features(raw_data)
    
    # use the feature expander and the inclusion list to expand the
    # features that are to be included to the appropriate powers.
    # The list of feature expansions will be appended to the data
    expander = FeatureExpander(training_data)
    expanded = expander.expand_features(inclusion_list)
    
    # Write the expanded data to a file for troubleshooting purposes
    write_to_file(expanded, fp_out)
    
    # Split the data into two folds
    [expanded_CV1, expanded_CV2] = split_data(expanded)
    
    # Perform linear regression to train a theta on each of the two folds
    THETA_CV1 = linear_regression(expanded_CV1)
    THETA_CV2 = linear_regression(expanded_CV2)
    
    # Now evaluate each theta on the fold on which it was not trained.
    # get the MSE
    result = evaluate_linear(expanded_CV1,expanded_CV2,THETA_CV1,THETA_CV2)
    
    # return the MSE
    return result
    
#####################################################################################################
# This block contains the code that was implemented to extend linear regression
# to *regularised* linear regression as part of task 3
#####################################################################################################

def regularised_linear_loss(THETA, data, lamb):
    """ Calculates loss of regularised linear regression
    """
    complexity = 0
    
    for theta in nditer(THETA):
        complexity += abs(theta)
    
    return (lamb * total_squared_loss(THETA, data)) + ((1 - lamb) * complexity)

def regularised_linear_gradient(THETA, data, lamb):
    """ Calculates gradient of regularised linear regression for given theta, data and lambda
    """
    # This THETA_0 business is necessary because fmin_bfgs flattens THETA into
    # a 1D array which is useless for us.  We iteratively place each element
    # into a list, which we then turn into a matrix.
    THETA_0 = []
    for elem in nditer(THETA):
        THETA_0.append(float(elem))
    THETA_0 = transpose(matrix(THETA_0))
    
    # Get X and y from the data passed in
    X_matrix = matrix([[float(elem) for elem in nditer(row_matrix)] for row_matrix in [row[-1] for row in data]])
    y_matrix = transpose(matrix([row[1] for row in data]))
    
    # Calculate the gradient
    first_part = -2 * lamb * dot(transpose(X_matrix), y_matrix - dot(X_matrix, THETA_0))
    second_part = ((1 - lamb) * sign(THETA_0))
    
    grad_matrix = first_part + second_part
    
    # return as a scipy array (required data type)
    return sp.array(grad_matrix.A1)

def regularised_linear_regression(data, lamb):
    """ Initialise theta to a load of ones.
        Perform BFGS minimisation on the regularised linear loss to get the optimal THETA
    """
    length_of_expansion = data[0][-1].shape[1]
    
    initial_THETA = matrix(np.ones([length_of_expansion,1]))
    
    # Call optimisation function on regularised linear loss function with gradient function specified
    return fmin_bfgs(regularised_linear_loss, initial_THETA, fprime=regularised_linear_gradient, args=[data,lamb])
        
def reglinear(InputFileName):
    """ This method performs regularised linear regression to find the best set of
        coefficients for the features specified in the inclusion list
        at the top of this file.
    """
    
    # load the data from the file
    raw_data = load_data(InputFileName)
    
    # append all of the features to the data (including those not used)
    training_data = append_features(raw_data)
    
    # use the feature expander and the inclusion list to expand the
    # features that are to be included to the appropriate powers.
    # The list of feature expansions will be appended to the data
    expander = FeatureExpander(training_data)
    expanded = expander.expand_features(inclusion_list)
    
    # Write the expanded data to a file for troubleshooting purposes
    write_to_file(expanded, fp_out)
    
    # Split the data into two training folds and a test fold
    # The training folds account for 2/5 of the data each
    # The test fold accounds for 1/5 of the data
    [expanded_CV1, expanded_CV2, expanded_test] = split_data_3_folds(expanded)
    
    results = []
    
    lamb_resolution = 100 # How many lambdas to try (i.e. 100 means increments of 0.01)
    # for each lambda...
    for lamb in [i/lamb_resolution for i in range(1,lamb_resolution)]:
        print "Currently working on lambda:",lamb # for info
        
        # Perform regularised linear regression to train a theta on each of the
        # two training folds with the current lambda
        THETA_CV1 = regularised_linear_regression(expanded_CV1, lamb)
        THETA_CV2 = regularised_linear_regression(expanded_CV2, lamb)
        
        # Evaluate the MSE of each THETA on the other training fold to the one
        # it has already been trained on.  Store the results in a list.        
        results.append((THETA_CV1, lamb, calc_mse(expanded_CV2, THETA_CV1)))
        results.append((THETA_CV2, lamb, calc_mse(expanded_CV1, THETA_CV2)))
    
    # When all the results for each lambda and theta have been stored in the
    # list, find the theta with the best MSE on the second CV fold
    best_result = results[0]
    for result in results:
        if result[2] < best_result[2]:
            best_result = result
    
    # Evaluate the performance of that theta on the test data
    final_score = calc_mse(expanded_test, best_result[0])
    
    # Print the lambda of the best theta for info
    print "Best lambda:", best_result[1]
    
    # Return the score evaluation
    return final_score

#####################################################################################################
# This block contains the code that was implemented for task 2 (Logistic regression)
#####################################################################################################

def append_classifications(source_data):
    """ Given the source data, append the classification of each
        day onto the end of the row.  Calculated using the classify
        function below.
    """
    result_data=[]
    
    for i in range(len(source_data)):
        this_row = source_data[i]
        last_row = source_data[i-1]
        this_row.append(classify(last_row[1], this_row[1]))
        result_data.append(this_row)
    
    return result_data
    
def unflatten_theta(THETA):
    """ This method is used to turn THETA into a 2D matrix
        (i.e. a single vector THETA for each of the 5 classes).
        Necessary because fmin_bfgs routine for some reason flattens
        it into a 1D array
    """
    if len(THETA.shape) > 1:
        return THETA
        
    return matrix(reshape(THETA, (-1, 5)))
    
def log_sum_exp(XTHETA):
    """ Log_sum_exp trick as described in the slides.
        Parameter XTHETA is the dot product of X and THETA
        where X is the vector of expanded features
    """
    max = XTHETA[0,0]
    for elem in nditer(XTHETA):
        if elem > max:
            max = elem
    
    sum = 0
    
    for elem in nditer(XTHETA):
        sum += exp(elem - max)

    return (max + log(sum))
    
def log_likelihood(THETA, data):
    """ Calculates the log likelihood for the regression optimisation
    """
    result = 0.0
    THETA = unflatten_theta(THETA)
    for i in range(len(data)):    
        X = data[i][4] # vector of expanded features
        c = data[i][2] # actual classification
        result += dot(X, THETA[:,c])[0,0]
        result -= log_sum_exp(dot(X, THETA))
    
    return result

def neg_log_likelihood(THETA, data):
    """ Negates the log likelihood, so that we can minimise it
    """
    return log_likelihood(THETA, data) * -1

def prob_a_given_X(THETA, X, a):
    """ Calculates the probability of a row being in class a,
        given the set of feature expansions (X) and the set of thetas (THETA)
    """
    dot_Xi_Ta = dot(X, THETA[:,a])[0,0]
    l_s_e = log_sum_exp(dot(X, THETA))
    
    if l_s_e > 700:
        # This is to avoid overflows.  If l_s_e > 700
        # then we just assume that the denominator - exp(l_s_e) is high
        # and return smallest value.
        return float_info.min
    else:
        numerator = exp(dot_Xi_Ta)
        denominator = exp(l_s_e)
        if denominator == 0:
            denominator = float_info.min
        return numerator / denominator

def classify(last_price, this_price):
    """ Calculates the classification of a data row, given the price of
        the row and the price of the row before it
    """
    percent_change = this_price / last_price
    
    if percent_change < 0.9:
        return 4
    if percent_change < 0.95:
        return 2
    if percent_change <= 1.05:
        return 0
    if percent_change <= 1.1:
        return 1
    
    return 3
        
def logistic_gradient(THETA, data):
    """ Calculates gradient of logistic regression for given theta and data
    """
    
    # Firstly, we have to get THETA back into a 2D form (i.e. one theta
    # vector for each class), because the fmin_bfgs routine flattens it
    # for some reason
    THETA = unflatten_theta(THETA)
    
    # Initialise the gradient matrix
    GRAD_shape = THETA.shape
    if len(GRAD_shape) == 1:
        GRAD_shape = [1, THETA.shape[0]]
    GRAD = np.zeros(GRAD_shape)
    GRAD[:,:] = float_info.min
    
    # Calculate the gradient for each classification
    for a in range(5):
        T_a = THETA[:,a]
        
        for i in range(len(data)):
            X_i = data[i][4] # These are the expanded features
            c = data[i][2] # This is the actual classification
            T_c = THETA[:,c] # This is the theta for the actual classification
            I = 1 if a == c else 0
            delta_loss = multiply(X_i, I - prob_a_given_X(THETA, X_i, a))
            delta_loss = [float(elem) for elem in nditer(delta_loss)]
            GRAD[:,a] += delta_loss
    
    # flatten the gradient matrix to make it compatible with fmin_bfgs routine
    flattened = np.reshape(sp.array(GRAD), (-1))    
    return flattened

def logistic_regression(data):
    """ Initialise theta to a load of ones.
        Perform BFGS minimisation on the negative log likelihood to
        get the optimal set of thetas
    """
    length_of_expansion = data[0][4].shape[1]
    
    initial_THETA = matrix(np.ones([length_of_expansion,5]))
    
    # Call optimisation function on neg log likelihood function with gradient function specified
    return fmin_bfgs(neg_log_likelihood, initial_THETA, fprime=logistic_gradient, args=[data])

def hard_predict(THETA, X):
    """ Predicts the category of a specific data row using the THETA
        provided and the expanded features appended to each row.
        This is a hard classification prediction
    """
    best_i = -1
    p_best = -1
    for i in range(5):
        p_i = prob_a_given_X(THETA, X, i)
        if prob_a_given_X(THETA, X, i) > p_best:
            p_best = p_i
            best_i = i
    
    return best_i

def percentage_correct_classifications(THETA, data):
    """ Using the THETAs given and the data given, gets the
        hard prediction classification for each data row.  Compares
        with the actual classification and returns an accuracy value
        (percentage of classifications that are correct)
    """
    correct = 0
    incorrect = 0
    
    for row in data:
        actual_classification = row[2]
        hard_prediction = hard_predict(THETA, row[4])
        
        if hard_prediction == actual_classification:
            correct += 1
        else:
            incorrect += 1
    
    return (100 * correct) / (correct + incorrect)
    
def evaluate_logistic(data_CV1, data_CV2, THETA_CV1, THETA_CV2):
    """ Calculates the percentage accuracy of the classifications
        given by THETA_CV1 on data_CV2 and THETA_CV2 on data_CV1
        Returns the average of both
    """
    pc_CV1 = percentage_correct_classifications(unflatten_theta(THETA_CV2), data_CV1)
    pc_CV2 = percentage_correct_classifications(unflatten_theta(THETA_CV1), data_CV2)
    return (pc_CV1 + pc_CV2) / 2
            
def logistic(InputFileName):
    """ This method performs logistic regression to find the best sets of
        coefficients for the features specified in the inclusion list
        at the top of this file in order to classify the data as
        directed by the assessment.
    """
    
    # load the data from the file
    raw_data = load_data(InputFileName)
    
    # calculate the classification of each data row and append it
    classified_data = append_classifications(raw_data)
    
    # normalise the stock volume and stock price data
    all_normalised_data = normalise(classified_data)
    
    # append all of the features to the data (including those not used)
    training_data = append_features(all_normalised_data)
    
    # use the feature expander and the inclusion list to expand the
    # features that are to be included to the appropriate powers.
    # The list of feature expansions will be appended to the data
    expander = FeatureExpander(training_data)
    expanded = expander.expand_features(inclusion_list)
    
    # Write the expanded data to a file for troubleshooting purposes
    write_to_file(expanded, fp_out)
    
    # Split the data into two folds
    [expanded_CV1, expanded_CV2] = split_data(expanded)
    
    # Perform logistic regression to train a set of thetas on each of the two folds
    THETA_CV1 = logistic_regression(expanded_CV1)
    THETA_CV2 = logistic_regression(expanded_CV2)
    
    # Now evaluate each set of thetas on the fold on which it was not trained.
    # get and return the accuracy
    return evaluate_logistic(expanded_CV1,expanded_CV2,THETA_CV1,THETA_CV2)
    
#####################################################################################################
# This block contains the code that was implemented to extend logistic regression
# to *regularised* logistic regression as part of task 3
#####################################################################################################
    
def regularised_logistic_loss(THETA, data, lamb):
    """ Calculates loss of regularised logistic regression
    """
    complexity = 0
    
    for theta in nditer(THETA): # since THETA is flat this is OK
        complexity += abs(theta)
        
    return (lamb * neg_log_likelihood(THETA, data)) - ((1 - lamb) * complexity)
    
def regularised_logistic_gradient(THETA, data, lamb):
    """ Calculates gradient of regularised logistic regression for given theta, data and lambda
    """
    
    # Firstly, we have to get THETA back into a 2D form (i.e. one theta
    # vector for each class), because the fmin_bfgs routine flattens it
    # for some reason
    THETA = unflatten_theta(THETA)
    
    # Initialise the gradient matrix
    GRAD_shape = THETA.shape
    if len(GRAD_shape) == 1:
        GRAD_shape = [1, THETA.shape[0]]
    unregularised_GRAD = np.zeros(GRAD_shape)
    unregularised_GRAD[:,:] = float_info.min
    
    # Calculate the unregularised gradient for each classification
    for a in range(5):
        T_a = THETA[:,a]
        
        for i in range(len(data)):
            X_i = data[i][4] # These are the expanded features
            c = data[i][2] # This is the actual classification
            T_c = THETA[:,c] # This is the theta for the actual classification
            I = 1 if a == c else 0
            delta_loss = multiply(X_i, I - prob_a_given_X(THETA, X_i, a))
            delta_loss = [float(elem) for elem in nditer(delta_loss)]
            unregularised_GRAD[:,a] += delta_loss
    
    # use lambda and sign function to get regularised gradient
    GRAD = (lamb * unregularised_GRAD) + ((1 - lamb) * sign(THETA))
    
    return sp.array(GRAD.A1)

def regularised_logistic_regression(data, lamb):
    """ Initialise theta to a load of ones.
        Perform BFGS minimisation on the negative log likelihood to get the optimal THETA
    """
    length_of_expansion = data[0][4].shape[1]
    
    initial_THETA = matrix(np.ones([length_of_expansion,5]))
    
    # Call optimisation function on neg log likelihood function with gradient function specified
    return fmin_bfgs(regularised_logistic_loss, initial_THETA, fprime=regularised_logistic_gradient, args=[data,lamb])
            
def reglogistic(InputFileName):
    """ This method performs regularised logistic regression to find the best sets of
        coefficients for the features specified in the inclusion list
        at the top of this file in order to classify the data as
        directed by the assessment.
    """
    
    # load the data from the file
    raw_data = load_data(InputFileName)
    
    # calculate the classification of each data row and append it
    classified_data = append_classifications(raw_data)
    
    # normalise the stock volume and stock price data
    all_normalised_data = normalise(classified_data)
    
    # append all of the features to the data (including those not used)
    training_data = append_features(all_normalised_data)

    # use the feature expander and the inclusion list to expand the
    # features that are to be included to the appropriate powers.
    # The list of feature expansions will be appended to the data
    expander = FeatureExpander(training_data)
    expanded = expander.expand_features(inclusion_list)
    
    # Write the expanded data to a file for troubleshooting purposes
    write_to_file(expanded, fp_out)
    
    # Split the data into two training folds and a test fold
    # The training folds account for 2/5 of the data each
    # The test fold accounds for 1/5 of the data
    [expanded_CV1, expanded_CV2, expanded_test] = split_data_3_folds(expanded)
    
    results = []
    
    lamb_resolution = 10 # How many lambdas to try (i.e. 100 means increments of 0.01)
    # for each lambda...
    for lamb in [i/lamb_resolution for i in range(1,lamb_resolution)]:
        print "Currently working on lambda:",lamb # for info
        
        # Perform regularised logistic regression to train a theta on each of the
        # two training folds with the current lambda
        THETA_CV1 = unflatten_theta(regularised_logistic_regression(expanded_CV1, lamb))
        THETA_CV2 = unflatten_theta(regularised_logistic_regression(expanded_CV2, lamb))
        
        # Evaluate the accuracy of each THETA on the other training fold to the one
        # it has already been trained on.  Store the results in a list.   
        results.append((THETA_CV1, lamb, percentage_correct_classifications(THETA_CV1, expanded_CV2)))
        results.append((THETA_CV2, lamb, percentage_correct_classifications(THETA_CV2, expanded_CV1)))
    
    # When all the results for each lambda and theta have been stored in the
    # list, find the theta with the best accuracy on the second CV fold
    best_result = results[0]
    for result in results:
        if result[2] > best_result[2]:
            best_result = result
            
    # Print the lambda of the best theta for info
    print "Best lambda:", best_result[1]
            
    # Evaluate the performance of that theta on the test data
    final_score = percentage_correct_classifications(best_result[0], expanded_test)
    
    # Return the score evaluation
    return final_score