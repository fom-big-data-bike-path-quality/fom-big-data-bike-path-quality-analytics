import inspect
import os
import glob
from email.utils import formatdate

import matplotlib.pyplot as plt
import seaborn as sns
from tracking_decorator import TrackingDecorator
from mlxtend.plotting import plot_decision_regions


#
# Main
#

class DecisionRegionsPlotter:

    @TrackingDecorator.track_time
    def run(self, logger, results_path, data_array, data_labels, classifier, clean=False, quiet=False):

        file_name = "decision_regions"

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, file_name + ".png"))
            for f in files:
                os.remove(f)

        plot_decision_regions(data_array, data_labels.astype(int), clf=classifier, legend=2)

        plt.title("Decision regions")
        plt.savefig(fname=os.path.join(results_path, file_name + ".png"),
                    format="png",
                    metadata={
                        "Title": "Decision regions",
                        "Author": "Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": "Plot of kNN decision regions"
                    })

        plt.close()

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(class_name + "." + function_name + " plotted decision regions")
