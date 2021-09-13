import os
from email.utils import formatdate

from email.utils import formatdate

import matplotlib.pyplot as plt
import seaborn as sns


#
# Main
#

class ConfusionMatrixPlotter:

    def run(self, logger, results_path, confusion_matrix_dataframe):
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

        logger.log_line("Confusion Matrix plotter finished")
