import glob
import inspect
import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import seaborn as sns
from tracking_decorator import TrackingDecorator


#
# Main
#

class ConfusionMatrixPlotter:

    @TrackingDecorator.track_time
    def run(self, logger, results_path, confusion_matrix_dataframe, file_name="confusion_matrix", clean=False,
            quiet=False):

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, file_name + ".png"))
            for f in files:
                os.remove(f)

        fig, ax = plt.subplots(figsize=(16, 14))

        heatmap = sns.heatmap(confusion_matrix_dataframe, annot=True, fmt="d", cmap='Blues', ax=ax)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)

        bottom, top = heatmap.get_ylim()
        heatmap.set_ylim(bottom + 0.5, top - 0.5)

        plt.title("Confusion matrix")
        plt.xlabel("Prediction")
        plt.ylabel("Target")
        plt.savefig(fname=os.path.join(results_path, file_name + ".png"),
                    format="png",
                    metadata={
                        "Title": "Confusion matrix",
                        "Author": "Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of training confusion matrix"
                    })

        plt.close()

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(class_name + "." + function_name + " plotted confusion matrix")
