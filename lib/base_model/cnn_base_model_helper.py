import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/base_model'),
]

# Import library classes
from classifier import Classifier

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def create_loader(dataset, batch_size=128, shuffle=False, num_workers=0):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
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

        # Create data loaders
        train_data_loader = create_loader(train_dataset, shuffle=True)
        test_data_loader = create_loader(test_dataset, shuffle=False)

        # Define classifier
        classifier = Classifier(
            input_channels=train_array.shape[1],
            num_classes=18
        ).to(device)
