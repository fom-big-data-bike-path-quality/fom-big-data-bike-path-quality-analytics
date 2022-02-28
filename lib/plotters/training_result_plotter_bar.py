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

class TrainingResultPlotterBar:

    @TrackingDecorator.track_time
    def run(self, logger, data, labels, results_path, file_name, title, description, xlabel, ylabel, color="#3A6FB0",
            clean=False, quiet=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, file_name, "*.png"))
            for f in files:
                os.remove(f)

        y_pos = list(range(0, len(labels)))

        plt.bar(y_pos, data, align='center', alpha=0.5, color=color)
        plt.xticks(y_pos, labels)
        plt.ylabel(ylabel)
        plt.title(description)
        plt.savefig(
            fname=os.path.join(results_path, f"{file_name}.png"),
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
            logger.log_line(f"✓️ Plotting {file_name}", console=False, file=True)

            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(f"{class_name}.{function_name} plotted training result")
