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

class DistanceMatrixPlotter:

    @TrackingDecorator.track_time
    def run(self, logger, results_path, distance_matrix_dataframe, clean=False, quiet=False):

        file_name = "distance_matrix"

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, f"{file_name}.png"))
            for f in files:
                os.remove(f)

        fig, ax = plt.subplots(figsize=(16, 14))

        heatmap = sns.heatmap(distance_matrix_dataframe, annot=True, fmt="d", cmap='Blues', ax=ax)
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=10)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=10)

        bottom, top = heatmap.get_ylim()
        heatmap.set_ylim(bottom + 0.5, top - 0.5)

        plt.title("Distance matrix")
        plt.savefig(fname=os.path.join(results_path, f"{file_name}.png"),
                    format="png",
                    metadata={
                        "Title": "Distance matrix",
                        "Author": "Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of kNN distance matrix"
                    })

        plt.close()

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(f"{class_name}.{function_name} plotted distance matrix")
