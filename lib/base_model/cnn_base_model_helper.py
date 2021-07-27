import numpy as np


def create_array(dataframes):
    """
    Converts an array of data frame into a 3D numpy array

    axis-0 = epoch
    axis-1 = features in a measurement
    axis-2 = measurements in an epoch

    """
    array = []

    for name, dataframe in dataframes.items():
        array.append(dataframe.to_numpy())

    return np.dstack(array).transpose(2, 1, 0)


#
# Main
#

class CnnBaseModelHelper:

    def run(self, train_dataframes, test_dataframes):
        # Create arrays
        train_array = create_array(train_dataframes)
        test_array = create_array(test_dataframes)
