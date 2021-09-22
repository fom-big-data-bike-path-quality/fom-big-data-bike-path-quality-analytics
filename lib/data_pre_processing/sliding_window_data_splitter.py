import csv
import glob
import inspect
import os
from pathlib import Path

import numpy as np
from tracking_decorator import TrackingDecorator


#
# Main
#

def rolling_window(array, window):
    shape = array.shape[:-1] + (array.shape[-1] - window + 1, window)
    strides = array.strides + (array.strides[-1],)
    return np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)


class SlidingWindowDataSplitter:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, slice_width, window_step, results_path, clean=False, quiet=False):

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.csv"))
            for f in files:
                os.remove(f)

        slices_count_total = 0

        for file_path in glob.iglob(data_path + "/*.csv"):
            file_name = os.path.basename(file_path)

            slices_count = 0

            with open(file_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)

                rows = []

                for row in csv_reader:
                    rows.append(row)

                try:
                    slices = rolling_window(np.array(rows), slice_width)
                except ValueError:
                    if not quiet:
                        logger.log_line("✗️ Cannot split " + file_path)
                    continue

                # Determine bike activity UID
                bike_activity_uid = os.path.splitext(file_name)[0]

                for index, slice in enumerate(slices):

                    if index % window_step == 0:

                        result_file = os.path.join(results_path, bike_activity_uid + "-" + str(index).rjust(5, '0') + ".csv")

                        if not Path(result_file).exists() or clean:

                            # Create file and append header
                            out_file = open(result_file, "w")
                            csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames)
                            csv_writer.writeheader()

                            for row in slice:

                                if row != None:
                                    # Append row
                                    csv_writer.writerow(row)

                            out_file.close()

                            slices_count = len(slices)
                            slices_count_total += slices_count

            if slices_count > 0 and not quiet:
                logger.log_line("✓️ Splitting into slices " + file_name)

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(class_name + "." + function_name + " splitted data in " + str(slices_count_total) + " slices")
