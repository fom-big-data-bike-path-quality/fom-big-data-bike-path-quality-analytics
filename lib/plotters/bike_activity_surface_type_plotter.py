import glob
import inspect
import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter
from tracking_decorator import TrackingDecorator


def create_array(dataframes):
    """
    Converts an array of data frame into a 3D numpy array

    axis-0 = epoch
    axis-1 = features in a measurement
    axis-2 = measurements in an epoch

    """
    array = []

    for name, dataframe in dataframes.items():
        if dataframe.shape[0] == 500:
            array.append(dataframe.to_numpy())
        else:
            pass

    return np.dstack(array).transpose(2, 1, 0)


#
# Main
#

class BikeActivitySurfaceTypePlotter:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, target_column, results_path, file_name, title, description, xlabel, ylabel, clean=False, quiet=False):
        if len(dataframes) == 0:
            logger.log_line("✗️ Not plotting " + file_name + " because there are no dataframes to plot")
        else:
            # Make results path
            os.makedirs(results_path, exist_ok=True)

            # Clean results path
            if clean:
                files = glob.glob(os.path.join(results_path, "*.png"))
                for f in files:
                    os.remove(f)

            plt.figure(2)
            plt.clf()
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

            array = create_array(dataframes)

            # 1D array with
            # axis-0 = TARGET of an epoch
            data = array[:, target_column, 1]

            plt.hist(data, weights=np.ones(len(data)) / len(data))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

            plt.savefig(
                fname=results_path + "/" + file_name + ".png",
                format="png",
                metadata={
                    "Title": title,
                    "Author": "Florian Schwanz",
                    "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                    "Description": description
                }
            )

            plt.close()

            if not quiet:
                logger.log_line("✓️ Plotting " + file_name)

            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            if not quiet:
                logger.log_line(class_name + "." + function_name + " plotted surface types")
