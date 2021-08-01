import glob
import os
from email.utils import formatdate

import matplotlib.pyplot as plt
import numpy as np


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


#
# Main
#

class TrainingResultPlotter:

    def run(self, logger, data, results_path, file_name, title, description, xlabel, ylabel, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, file_name, "*.png"))
            for f in files:
                os.remove(f)

        plt.figure(2)
        plt.clf()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.plot(smooth(data, 5))

        plt.savefig(fname=results_path + "/" + file_name + ".png",
                    format="png",
                    metadata={
                        "Title": title,
                        "Author": "Florian Schwanz",
                        "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                        "Description": description
                    })

        plt.close()

        logger.log_line("✓️ Plotting " + file_name)

        logger.log_line("Training result plotter finished")
