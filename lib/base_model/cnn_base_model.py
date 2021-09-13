import os
import statistics
from email.utils import formatdate

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from classifier import Classifier
from label_encoder import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix as cm
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from training_result_plotter import TrainingResultPlotter

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes
num_classes = LabelEncoder().num_classes()


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


def plot_confusion_matrix(results_path, confusion_matrix_dataframe):
    fig, ax = plt.subplots(figsize=(16, 14))

    heatmap = sns.heatmap(confusion_matrix_dataframe, annot=True, fmt="d", cmap='Blues', ax=ax)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)

    bottom, top = heatmap.get_ylim()
    heatmap.set_ylim(bottom + 0.5, top - 0.5)

    results_file = os.path.join(results_path, "confusion_matrix.png")

    plt.title("Confusion matrix")
    plt.xlabel("Prediction")
    plt.ylabel("Target")
    plt.savefig(fname=results_file,
                format="png",
                metadata={
                    "Title": "Confusion matrix",
                    "Author": "Florian Schwanz",
                    "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                    "Description": "Plot of training confusion matrix"
                })

    plt.close()


def get_accuracy(confusion_matrix_dataframe):
    tp = 0

    for i in confusion_matrix_dataframe.index:
        tp += get_true_positives(confusion_matrix_dataframe, i)

    total = get_total_predictions(confusion_matrix_dataframe)

    return tp / total


def get_precision(confusion_matrix_dataframe):
    precisions = []

    for i in confusion_matrix_dataframe.index:
        tp = get_true_positives(confusion_matrix_dataframe, i)
        fp = get_false_positives(confusion_matrix_dataframe, i)
        precision = 0

        if (tp + fp) > 0:
            precision = tp / (tp + fp)

        precisions.append(precision)

    return np.mean(precisions)


def get_recall(confusion_matrix_dataframe):
    recalls = []

    for i in confusion_matrix_dataframe.index:
        tp = get_true_positives(confusion_matrix_dataframe, i)
        fn = get_false_negatives(confusion_matrix_dataframe, i)
        recall = 0

        if (tp + fn) > 0:
            recall = tp / (tp + fn)

        recalls.append(recall)

    return np.mean(recalls)


def get_f1_score(confusion_matrix_dataframe):
    f1_scores = []

    for i in confusion_matrix_dataframe.index:
        tp = get_true_positives(confusion_matrix_dataframe, i)
        fp = get_false_positives(confusion_matrix_dataframe, i)
        fn = get_false_negatives(confusion_matrix_dataframe, i)
        precision = 0
        recall = 0
        f1_score = 0

        if (tp + fp) > 0:
            precision = tp / (tp + fp)

        if (tp + fn) > 0:
            recall = tp / (tp + fn)

        if (precision + recall) > 0:
            f1_score = (2 * precision * recall) / (precision + recall)

        f1_scores.append(f1_score)

    return statistics.harmonic_mean(f1_scores)


def get_cohen_kappa_score(target, prediction):
    return cohen_kappa_score(target, prediction)


def get_true_positives(confusion_matrix_dataframe, index):
    return confusion_matrix_dataframe.loc[index, index]


def get_false_positives(confusion_matrix_dataframe, index):
    fp = 0

    for i in confusion_matrix_dataframe.index:
        if i != index:
            fp += confusion_matrix_dataframe.loc[i, index]

    return fp


def get_false_negatives(confusion_matrix_dataframe, index):
    fn = 0

    for i in confusion_matrix_dataframe.index:
        if i != index:
            fn += confusion_matrix_dataframe.loc[index, i]

    return fn


def get_total_predictions(confusion_matrix_dataframe):
    total = 0
    for i in confusion_matrix_dataframe.index:
        for j in confusion_matrix_dataframe.index:
            total += confusion_matrix_dataframe.loc[i, j]

    return total


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
            num_classes=num_classes
        ).to(device)
        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        accuracy_max = 0
        patience, trials = 1_000, 0
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

        targets = []
        predictions = []

        model = Classifier(
            input_channels=1,  # TODO Derive this value from data
            # input_channels=train_array.shape[1],
            num_classes=num_classes
        ).to(device)
        model.load_state_dict(torch.load(os.path.join(log_path, "model.pickle")))
        model.eval()

        confusion_matrix = np.zeros((num_classes, num_classes))

        with torch.no_grad():
            for i, (input, target) in enumerate(test_data_loader):
                input = input.to(device)
                target = target.to(device)

                output = model(input)

                _, prediction = torch.max(output, 1)

                targets.extend(target.tolist())
                predictions.extend(prediction.tolist())

                for t, p in zip(target.view(-1), prediction.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        # Build confusion matrix, limit to classes actually used
        confusion_matrix_dataframe = pd.DataFrame(confusion_matrix, index=LabelEncoder().classes, columns=LabelEncoder().classes).astype(
            int)
        used_columns = (confusion_matrix_dataframe != 0).any(axis=0).where(lambda x: x == True).dropna().keys().tolist()
        used_rows = (confusion_matrix_dataframe != 0).any(axis=1).where(lambda x: x == True).dropna().keys().tolist()
        used_classes = list(dict.fromkeys(used_columns + used_rows))
        confusion_matrix_dataframe = confusion_matrix_dataframe.filter(items=used_classes, axis=0).filter(items=used_classes, axis=1)

        # Plot confusion matrix
        plot_confusion_matrix(log_path, confusion_matrix_dataframe)

        logger.log_line("Confusion matrix \n" + str(cm(predictions, targets)))
        logger.log_line("Accuracy " + str(round(get_accuracy(confusion_matrix_dataframe), 2)))
        logger.log_line("Precision " + str(round(get_precision(confusion_matrix_dataframe), 2)))
        logger.log_line("Recall " + str(round(get_recall(confusion_matrix_dataframe), 2)))
        logger.log_line("F1 Score " + str(round(get_f1_score(confusion_matrix_dataframe), 2)))
        logger.log_line("Cohen's Kappa Score " + str(round(get_cohen_kappa_score(targets, predictions), 2)))
        logger.log_line("CNN base model evaluation finished")
