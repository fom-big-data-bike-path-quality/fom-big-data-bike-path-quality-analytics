import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
from classifier import Classifier
from confusion_matrix_plotter import ConfusionMatrixPlotter
from label_encoder import LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from tracking_decorator import TrackingDecorator
from training_result_plotter import TrainingResultPlotter

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Control sources of randomness
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

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

    return np.mean(f1_scores)


def get_cohen_kappa_score(target, prediction):
    return cohen_kappa_score(target, prediction)


def get_matthews_corrcoef_score(target, prediction):
    return matthews_corrcoef(target, prediction)


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


def get_confusion_matrix_dataframe(classifier, data_loader):
    targets = []
    predictions = []
    confusion_matrix = np.zeros((num_classes, num_classes))

    for batch in data_loader:
        input, target = [t.to(device) for t in batch]
        output = classifier(input)
        prediction = F.log_softmax(output, dim=1).argmax(dim=1)

        targets.extend(target.tolist())
        predictions.extend(prediction.tolist())

        for t, p in zip(target.view(-1), prediction.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

    # Build confusion matrix, limit to classes actually used
    confusion_matrix_dataframe = pd.DataFrame(confusion_matrix, index=LabelEncoder().classes,
                                              columns=LabelEncoder().classes).astype("int64")
    used_columns = (confusion_matrix_dataframe != 0).any(axis=0).where(lambda x: x == True).dropna().keys().tolist()
    used_rows = (confusion_matrix_dataframe != 0).any(axis=1).where(lambda x: x == True).dropna().keys().tolist()
    used_classes = list(dict.fromkeys(used_columns + used_rows))
    confusion_matrix_dataframe = confusion_matrix_dataframe.filter(items=used_classes, axis=0).filter(items=used_classes, axis=1)

    return confusion_matrix_dataframe, targets, predictions


def validate(classifier, data_loader):
    confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(classifier, data_loader)

    accuracy = get_accuracy(confusion_matrix_dataframe)
    precision = get_precision(confusion_matrix_dataframe)
    recall = get_recall(confusion_matrix_dataframe)
    f1_score = get_f1_score(confusion_matrix_dataframe)
    cohen_kappa_score = get_cohen_kappa_score(targets, predictions)
    matthew_correlation_coefficient = get_matthews_corrcoef_score(targets, predictions)

    return accuracy, precision, recall, f1_score, cohen_kappa_score, matthew_correlation_coefficient


def get_linear_channels(slice_width):
    if slice_width <= 250:
        return 256
    elif slice_width == 300:
        return 512
    elif slice_width >= 350:
        return 768
    else:
        return 256


#
# Main
#

class CnnBaseModel:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, k_folds, epochs, learning_rate, patience, slice_width, log_path, quiet=False):

        # Make results path
        os.makedirs(log_path, exist_ok=True)

        kf = KFold(n_splits=k_folds)
        fold_index = 0

        overall_validation_accuracy_max = 0
        overall_validation_accuracy_history = []
        overall_validation_precision_history = []
        overall_validation_recall_history = []
        overall_validation_f1_score_history = []
        overall_validation_cohen_kappa_score_history = []
        overall_validation_matthew_correlation_coefficient_history = []

        ids = sorted(list(dataframes.keys()))

        for train_ids, validation_ids in kf.split(ids):

            # Increment fold index
            fold_index += 1

            logger.log_line("\n Fold # " + str(fold_index) + "\n")

            train_dataframes = {id: list(dataframes.values())[id] for id in train_ids}
            validation_dataframes = {id: list(dataframes.values())[id] for id in validation_ids}

            # Create data loader for train
            train_array = create_array(train_dataframes)
            train_dataset = create_dataset(train_array)
            train_data_loader = create_loader(train_dataset, shuffle=False)

            # Create data loader for validation
            validation_array = create_array(validation_dataframes)
            validation_dataset = create_dataset(validation_array)
            validation_data_loader = create_loader(validation_dataset, shuffle=False)

            # Determine number of linear channels based on slice width
            linear_channels = get_linear_channels(slice_width)

            # Define classifier
            classifier = Classifier(input_channels=1, num_classes=num_classes, linear_channels=linear_channels).to(device)
            criterion = nn.CrossEntropyLoss(reduction='sum')
            optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

            validation_accuracy_max = 0
            trials = 0

            train_loss_history = []
            train_accuracy_history = []
            train_precision_history = []
            train_recall_history = []
            train_f1_score_history = []
            train_cohen_kappa_score_history = []
            train_matthew_correlation_coefficient_history = []

            validation_loss_history = []
            validation_accuracy_history = []
            validation_precision_history = []
            validation_recall_history = []
            validation_f1_score_history = []
            validation_cohen_kappa_score_history = []
            validation_matthew_correlation_coefficient_history = []

            # Run training loop
            progress_bar = tqdm(iterable=range(1, epochs + 1), unit='epochs', desc="Train model")
            for epoch in progress_bar:

                # Train model
                classifier.train()
                train_epoch_loss = 0
                for i, batch in enumerate(train_data_loader):
                    input, target = [t.to(device) for t in batch]
                    optimizer.zero_grad()
                    output = classifier(input)
                    loss = criterion(output, target)
                    train_epoch_loss += loss.item()
                    loss.backward()
                    optimizer.step()

                train_epoch_loss /= train_array.shape[0]
                train_loss_history.append(train_epoch_loss)
                classifier.eval()

                with torch.no_grad():
                    validation_epoch_loss = 0
                    for i, batch in enumerate(validation_data_loader):
                        input, target = [t.to(device) for t in batch]
                        optimizer.zero_grad()
                        output = classifier(input)
                        loss = criterion(output, target)
                        validation_epoch_loss += loss.item()

                    validation_epoch_loss /= validation_array.shape[0]
                    validation_loss_history.append(validation_epoch_loss)

                # Validate with train dataloader
                train_accuracy, \
                train_precision, \
                train_recall, \
                train_f1_score, \
                train_cohen_kappa_score, \
                train_matthew_correlation_coefficient = validate(classifier, train_data_loader)

                train_accuracy_history.append(train_accuracy)
                train_precision_history.append(train_precision)
                train_recall_history.append(train_recall)
                train_f1_score_history.append(train_f1_score)
                train_cohen_kappa_score_history.append(train_cohen_kappa_score)
                train_matthew_correlation_coefficient_history.append(train_matthew_correlation_coefficient)

                # Validate with validation dataloader
                validation_accuracy, \
                validation_precision, \
                validation_recall, \
                validation_f1_score, \
                validation_cohen_kappa_score, \
                validation_matthew_correlation_coefficient = validate(classifier, validation_data_loader)

                validation_accuracy_history.append(validation_accuracy)
                validation_precision_history.append(validation_precision)
                validation_recall_history.append(validation_recall)
                validation_f1_score_history.append(validation_f1_score)
                validation_cohen_kappa_score_history.append(validation_cohen_kappa_score)
                validation_matthew_correlation_coefficient_history.append(validation_matthew_correlation_coefficient)

                if not quiet:
                    logger.log_line("Fold " + str(fold_index) +
                                    " epoch " + str(epoch) +
                                    " loss " + str(round(train_epoch_loss, 4)).ljust(4, '0') + ", " +
                                    " accuracy " + str(round(validation_accuracy, 2)) + ", " +
                                    " precision " + str(round(validation_precision, 2)) + ", " +
                                    " recall " + str(round(validation_recall, 2)) + ", " +
                                    " f1 score " + str(round(validation_f1_score, 2)) + ", " +
                                    " cohen kappa score " + str(round(validation_cohen_kappa_score, 2)) + ", " +
                                    " matthew correlation coefficient " + str(round(validation_matthew_correlation_coefficient, 2)),
                                    console=False, file=True)

                # Check if accuracy increased
                if validation_accuracy > validation_accuracy_max:
                    trials = 0
                    validation_accuracy_max = validation_accuracy
                else:
                    trials += 1
                    if trials >= patience and not quiet:
                        logger.log_line("\nNo further improvement after epoch " + str(epoch))
                        overall_validation_accuracy_history.append(validation_accuracy)
                        overall_validation_precision_history.append(validation_precision)
                        overall_validation_recall_history.append(validation_recall)
                        overall_validation_f1_score_history.append(validation_f1_score)
                        overall_validation_cohen_kappa_score_history.append(validation_cohen_kappa_score)
                        overall_validation_matthew_correlation_coefficient_history.append(validation_matthew_correlation_coefficient)
                        break

                if validation_accuracy > overall_validation_accuracy_max:
                    overall_validation_accuracy_max = validation_accuracy
                    torch.save(classifier.state_dict(), os.path.join(log_path, "model.pickle"))

                progress_bar.close()

            TrainingResultPlotter().run(
                logger=logger,
                data=[train_loss_history, validation_loss_history],
                labels=["Train", "Validation"],
                results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
                file_name="loss",
                title="Loss history",
                description="Loss history",
                xlabel="Epoch",
                ylabel="Loss",
                clean=True,
                quiet=quiet)

            TrainingResultPlotter().run(
                logger=logger,
                data=[train_accuracy_history, validation_accuracy_history],
                labels=["Train", "Validation"],
                results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
                file_name="accuracy",
                title="Accuracy history",
                description="Accuracy history",
                xlabel="Epoch",
                ylabel="Accuracy",
                clean=True,
                quiet=quiet)

            TrainingResultPlotter().run(
                logger=logger,
                data=[validation_accuracy_history, validation_precision_history, validation_recall_history, validation_f1_score_history,
                      validation_cohen_kappa_score_history, validation_matthew_correlation_coefficient_history],
                labels=["Accuracy", "Precision", "Recall", "F1 Score", "Cohen's Kappa Score", "Matthew's Correlation Coefficient"],
                results_path=os.path.join(log_path, "plots", "fold-" + str(fold_index)),
                file_name="metrics",
                title="Metrics history",
                description="Metrics history",
                xlabel="Epoch",
                ylabel="Value",
                clean=True,
                quiet=quiet)

        if not quiet:
            logger.log_line("Cross-validation metrics")
            logger.log_line(
                "Mean accuracy " + str(round(np.mean(overall_validation_accuracy_history), 2)) + ", " +
                " precision " + str(round(np.mean(overall_validation_precision_history), 2)) + ", " +
                " recall " + str(round(np.mean(overall_validation_recall_history), 2)) + ", " +
                " f1 score " + str(round(np.mean(overall_validation_f1_score_history), 2)) + ", " +
                " cohen kappa score " + str(round(np.mean(overall_validation_cohen_kappa_score_history), 2)) + ", " +
                " matthew correlation coefficient " + str(round(np.mean(overall_validation_matthew_correlation_coefficient_history), 2)))
            logger.log_line(
                "Standard deviation " + str(round(np.std(overall_validation_accuracy_history), 2)) + ", " +
                " precision " + str(round(np.std(overall_validation_precision_history), 2)) + ", " +
                " recall " + str(round(np.std(overall_validation_recall_history), 2)) + ", " +
                " f1 score " + str(round(np.std(overall_validation_f1_score_history), 2)) + ", " +
                " cohen kappa score " + str(round(np.std(overall_validation_cohen_kappa_score_history), 2)) + ", " +
                " matthew correlation coefficient " + str(round(np.std(overall_validation_matthew_correlation_coefficient_history), 2)))

    @TrackingDecorator.track_time
    def evaluate(self, logger, dataframes, slice_width, model_path, log_path, clean=False, quiet=False):

        # Make results path
        os.makedirs(log_path, exist_ok=True)

        # Create data loader
        test_array = create_array(dataframes)
        test_dataset = create_dataset(test_array)
        test_data_loader = create_loader(test_dataset, shuffle=False)

        # Determine number of linear channels based on slice width
        linear_channels = get_linear_channels(slice_width)

        # Define classifier
        classifier = Classifier(input_channels=1, num_classes=num_classes, linear_channels=linear_channels).to(device)
        classifier.load_state_dict(torch.load(os.path.join(model_path, "model.pickle")))
        classifier.eval()

        # Validate with test dataloader
        test_accuracy, \
        test_precision, \
        test_recall, \
        test_f1_score, \
        test_cohen_kappa_score, \
        test_matthew_correlation_coefficient = validate(classifier, test_data_loader)

        # Plot confusion matrix
        test_confusion_matrix_dataframe, targets, predictions = get_confusion_matrix_dataframe(classifier, test_data_loader)
        ConfusionMatrixPlotter().run(logger, os.path.join(log_path, "plots"), test_confusion_matrix_dataframe, clean=clean)

        if not quiet:
            logger.log_line("Confusion matrix \n" + str(cm(targets, predictions)))
            logger.log_line("Accuracy " + str(round(test_accuracy, 2)))
            logger.log_line("Precision " + str(round(test_precision, 2)))
            logger.log_line("Recall " + str(round(test_recall, 2)))
            logger.log_line("F1 Score " + str(round(test_f1_score, 2)))
            logger.log_line("Cohen's Kappa Score " + str(round(test_cohen_kappa_score, 2)))
            logger.log_line("Matthew's Correlation Coefficient Score " + str(round(test_matthew_correlation_coefficient, 2)))
