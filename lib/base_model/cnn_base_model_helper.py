import numpy as np
import torch
from torch.utils.data import TensorDataset


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


def create_dataset(array):
    epochs_count = len(array)
    return TensorDataset(
        # 3D array with
        # axis-0 = epoch
        # axis-1 = features in a measurement (INPUT)
        # axis-2 = measurements in an epoch
        torch.tensor(data=array[:epochs_count, :-1]).float(),
        # 1D array with
        # axis-0 = TARGET of an epoch
        torch.tensor(data=array[:epochs_count, -1, :][:, 0]).long()
    )


#
# Main
#

class CnnBaseModelHelper:

    def run(self, train_dataframes, test_dataframes):
        # Create arrays
        train_array = create_array(train_dataframes)
        test_array = create_array(test_dataframes)

        # Create data sets
        train_dataset = create_dataset(train_array)
        test_dataset = create_dataset(test_array)
