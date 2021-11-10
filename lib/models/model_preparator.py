from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class ModelPreparator:

    def create_array(self, dataframes):
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

    def create_dataset(self, array):
        return TensorDataset(
            # 3D array with
            # axis-0 = epoch
            # axis-1 = features in a measurement (INPUT)
            # axis-2 = measurements in an epoch
            torch.tensor(data=array[:, -1:].astype("float64")).float(),
            # 1D array with
            # axis-0 = TARGET of an epoch
            torch.tensor(data=array[:, 0, :][:, 0].astype("int64")).long()
        )

    def create_loader(self, dataset, batch_size=128, shuffle=False, num_workers=0):
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

    def get_linear_channels(self, slice_width):
        if slice_width <= 250:
            return 256
        elif slice_width == 300:
            return 512
        elif slice_width >= 350:
            return 768
        else:
            return 256

    def split_data_and_labels(self, array):
        return array[:,1,:], array[:,0,0]