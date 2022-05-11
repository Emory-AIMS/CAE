from sklearn.neighbors import NearestNeighbors
import numpy as np
from termcolor import colored

#  -------------------------------------------------------------
def mean_outlier_detector(meanx_class, x_class):

    # Calculate distance of untrusted training set to threshold to detemine if they are outliers
    x_class = np.asarray(x_class)
    diff_tr = np.linalg.norm((x_class - meanx_class), axis=1)

    return diff_tr

# ------------------------------------------------------------------------------

def KNN_outlier_detector(nbrs, x_class):

    distances, indices = nbrs.kneighbors(x_class)
    diff_tr = np.linalg.norm((distances), axis=1)

    return diff_tr

# ------------------------------------------------------------------------------
def create_outlier_model(training_data_x, training_data_y, detector_type):


    if detector_type == 'mean_outlier_detector':
        pos_class_x = [xn.tolist() for xn, yn in zip(training_data_x, training_data_y) if yn == 1]
        neg_class_x = [xn.tolist() for xn, yn in zip(training_data_x, training_data_y) if yn != 1]
        meanx_class_pos = np.average(pos_class_x, axis=0)
        meanx_class_neg = np.average(neg_class_x, axis=0)
        return (meanx_class_pos, meanx_class_neg)
    elif detector_type == 'KNN_outlier_detector':
        k = 2
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(training_data_x)
        return nbrs
    else:
        print(colored('The outlier detector model not found!','red'))
        exit(1)


