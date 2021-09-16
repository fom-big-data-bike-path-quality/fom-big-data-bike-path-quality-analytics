import glob
import inspect
import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import numpy as np
from tracking_decorator import TrackingDecorator


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#
# Main
#

class TrainingResultPlotter:

    @TrackingDecorator.track_time
    def run(self, logger, data, labels, results_path, file_name, title, description, xlabel, ylabel, clean=False, quiet=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, file_name, "*.png"))
            for f in files:
                os.remove(f)

        plt.subplots(figsize=(16, 14))
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        for i in range(len(data)):
            plt.plot(smooth(data[i], 5), label=labels[i])

        plt.legend()

        plt.savefig(fname=results_path + "/" + file_name + ".png",
                    format="png",
                    metadata={
                        "Title": title,
                        "Author": "Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": description
                    })

        plt.close()

        if not quiet:
            logger.log_line("✓️ Plotting " + file_name)

            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " plotted training result")
