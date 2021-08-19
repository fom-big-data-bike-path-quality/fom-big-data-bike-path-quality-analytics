import os

import numpy as np
import torch
from classifier import Classifier
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from training_result_plotter import TrainingResultPlotter

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

class CnnBaseModel:

    def run(self, logger, train_dataframes, validation_dataframes, learning_rate, epochs, log_path):
        # Create arrays
        train_array = create_array(train_dataframes)
        validation_array = create_array(validation_dataframes)

        # Create data sets
        train_dataset = create_dataset(train_array)
        validation_dataset = create_dataset(validation_array)

        # Create data loaders
        train_data_loader = create_loader(train_dataset, shuffle=True)
        validation_data_loader = create_loader(validation_dataset, shuffle=True)

        # Define classifier
        classifier = Classifier(
            input_channels=1,  # TODO Derive this value from data
            # input_channels=train_array.shape[1],
            num_classes=18
        ).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        accuracy_max = 0
        patience, trials = 50_000, 0
        base = 1
        step = 2
        loss_history = []
        accuracy_history = []

        # Run training loop
        progress_bar = tqdm(iterable=range(1, epochs + 1), unit='epochs', desc="Train model")
        for epoch in progress_bar:

            classifier.train()
            epoch_loss = 0
            for i, batch in enumerate(train_data_loader):
                x_raw, y_batch = [t.to(device) for t in batch]
                optimizer.zero_grad()
                out = classifier(x_raw)
                loss = criterion(out, y_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            epoch_loss /= train_array.shape[0]
            loss_history.append(epoch_loss)

            classifier.eval()
            correct, total = 0, 0
            for batch in validation_data_loader:
                input, target = [t.to(device) for t in batch]
                output = classifier(input)
                y_hat = F.log_softmax(output, dim=1).argmax(dim=1)

                total += target.size(0)
                correct += (y_hat == target).sum().item()

            accuracy = correct / total
            accuracy_history.append(accuracy)

            if epoch % base == 0:
                logger.log_line("Epoch " + str(epoch) + " loss " + str(round(epoch_loss, 4)) + " accuracy " + str(round(accuracy, 2)))
                base *= step
            else:
                logger.log_line("Epoch " + str(epoch) + " loss " + str(round(epoch_loss, 4)) + " accuracy " + str(round(accuracy, 2)),
                                console=False, file=True)

            # Check if accuracy increased
            if accuracy > accuracy_max:
                trials = 0
                accuracy_max = accuracy
                torch.save(classifier.state_dict(), os.path.join(log_path, "model.pickle"))
            else:
                trials += 1
                if trials >= patience:
                    logger.log_line("No further improvement after epoch " + str(epoch))
                    break

        TrainingResultPlotter().run(
            logger=logger,
            data=loss_history,
            results_path=log_path + "/plots/training",
            file_name="loss",
            title="Validation loss history",
            description="Validation loss history",
            xlabel="Epoch",
            ylabel="Loss",
            clean=True)

        TrainingResultPlotter().run(
            logger=logger,
            data=accuracy_history,
            results_path=log_path + "/plots/training",
            file_name="accuracy",
            title="Validation accuracy history",
            description="Validation accuracy history",
            xlabel="Epoch",
            ylabel="Accuracy",
            clean=True)

        logger.log_line("CNN base model finished")


class CnnBaseModelEvaluation:

    def run(self, logger, test_dataframes, log_path):
        # Create arrays
        test_array = create_array(test_dataframes)

        # Create data sets
        test_dataset = create_dataset(test_array)

        # Create data loaders
        test_data_loader = create_loader(test_dataset, shuffle=False)

        model = Classifier(
            input_channels=1,  # TODO Derive this value from data
            # input_channels=train_array.shape[1],
            num_classes=18
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(log_path, "model.pickle")))
        model.eval()

        correct, total = 0, 0
        for batch in test_data_loader:
            input, target = [t.to(device) for t in batch]
            output = model(input)
            y_hat = F.log_softmax(output, dim=1).argmax(dim=1)

            total += target.size(0)
            correct += (y_hat == target).sum().item()

        accuracy = correct / total

        logger.log_line("Accuracy " + str(round(accuracy, 2)))
        logger.log_line("CNN base model evaluation finished")
