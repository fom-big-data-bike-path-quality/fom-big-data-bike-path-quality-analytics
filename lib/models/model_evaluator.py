import numpy as np
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import matthews_corrcoef


class ModelEvaluator:

    def get_accuracy(self, confusion_matrix_dataframe):
        tp = 0

        for i in confusion_matrix_dataframe.index:
            tp += self.get_true_positives(confusion_matrix_dataframe, i)

        total = self.get_total_predictions(confusion_matrix_dataframe)

        return tp / total

    def get_precision(self, confusion_matrix_dataframe):
        precisions = []

        for i in confusion_matrix_dataframe.index:
            tp = self.get_true_positives(confusion_matrix_dataframe, i)
            fp = self.get_false_positives(confusion_matrix_dataframe, i)
            precision = 0

            if (tp + fp) > 0:
                precision = tp / (tp + fp)

            precisions.append(precision)

        return np.mean(precisions)

    def get_recall(self, confusion_matrix_dataframe):
        recalls = []

        for i in confusion_matrix_dataframe.index:
            tp = self.get_true_positives(confusion_matrix_dataframe, i)
            fn = self.get_false_negatives(confusion_matrix_dataframe, i)
            recall = 0

            if (tp + fn) > 0:
                recall = tp / (tp + fn)

            recalls.append(recall)

        return np.mean(recalls)

    def get_f1_score(self, confusion_matrix_dataframe):
        f1_scores = []

        for i in confusion_matrix_dataframe.index:
            tp = self.get_true_positives(confusion_matrix_dataframe, i)
            fp = self.get_false_positives(confusion_matrix_dataframe, i)
            fn = self.get_false_negatives(confusion_matrix_dataframe, i)
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

    def get_cohen_kappa_score(self, target, prediction):
        return cohen_kappa_score(target, prediction)

    def get_matthews_corrcoef_score(self, target, prediction):
        return matthews_corrcoef(target, prediction)

    def get_true_positives(self, confusion_matrix_dataframe, index):
        return confusion_matrix_dataframe.loc[index, index]

    def get_false_positives(self, confusion_matrix_dataframe, index):
        fp = 0

        for i in confusion_matrix_dataframe.index:
            if i != index:
                fp += confusion_matrix_dataframe.loc[i, index]

        return fp

    def get_false_negatives(self, confusion_matrix_dataframe, index):
        fn = 0

        for i in confusion_matrix_dataframe.index:
            if i != index:
                fn += confusion_matrix_dataframe.loc[index, i]

        return fn

    def get_total_predictions(self, confusion_matrix_dataframe):
        total = 0
        for i in confusion_matrix_dataframe.index:
            for j in confusion_matrix_dataframe.index:
                total += confusion_matrix_dataframe.loc[i, j]

        return total
