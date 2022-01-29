import csv
import glob
import inspect
import math
import os
from email.utils import formatdate
from pathlib import Path

import matplotlib.pyplot as plt
from tracking_decorator import TrackingDecorator
from tqdm import tqdm

#
# Main
#

class BikeActivitySlicePlotter:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, results_path, xlabel, ylabel, colors=["#3A6FB0", "#79ABD1"], clean=False,
            quiet=False):

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.png"))
            for f in files:
                os.remove(f)

        bike_activity_slices_plotted_count = 0

        files = list(Path(data_path).rglob("*.csv"))

        progress_bar = tqdm(iterable=files, unit="file", desc="Plot bike activities")
        for file_path in progress_bar:
            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            results_file = os.path.join(results_path, file_base_name + ".png")

            if not Path(results_file).exists() or clean:

                with open(str(file_path)) as csv_file:

                    csv_reader = csv.DictReader(csv_file)

                    data_accelerometer_x = []
                    data_accelerometer_y = []
                    data_accelerometer_z = []
                    data_accelerometer = []
                    data_speed = []

                    for row in csv_reader:
                        bike_activity_measurement_accelerometer_x = float(row["bike_activity_measurement_accelerometer_x"])
                        bike_activity_measurement_accelerometer_y = float(row["bike_activity_measurement_accelerometer_y"])
                        bike_activity_measurement_accelerometer_z = float(row["bike_activity_measurement_accelerometer_z"])
                        bike_activity_measurement_accelerometer = math.sqrt((bike_activity_measurement_accelerometer_x ** 2
                                                                             + bike_activity_measurement_accelerometer_y ** 2
                                                                             + bike_activity_measurement_accelerometer_z ** 2) / 3)
                        bike_activity_measurement_speed = float(row["bike_activity_measurement_speed"])
                        bike_activity_surface_type = row["bike_activity_surface_type"]

                        data_accelerometer_x.append(bike_activity_measurement_accelerometer_x)
                        data_accelerometer_y.append(bike_activity_measurement_accelerometer_y)
                        data_accelerometer_z.append(bike_activity_measurement_accelerometer_z)
                        data_accelerometer.append(bike_activity_measurement_accelerometer)
                        data_speed.append(bike_activity_measurement_speed * 3.6)

                    plt.figure(2)
                    plt.clf()
                    plt.title("Bike activity sample " + file_base_name + " (" + bike_activity_surface_type + ")")
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    # plt.plot(data_accelerometer_x, label="accelerometer x")
                    # plt.plot(data_accelerometer_y, label="accelerometer y")
                    # plt.plot(data_accelerometer_z, label="accelerometer z")
                    plt.plot(data_accelerometer, label="accelerometer", color=colors[0])
                    plt.plot(data_speed, label="speed", color=colors[1])
                    plt.legend()

                    plt.savefig(fname=results_file,
                                format="png",
                                metadata={
                                    "Title": "Bike activity sample " + file_base_name,
                                    "Author": "Florian Schwanz",
                                    "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                                    "Description": "Plot of bike activity sample " + file_base_name
                                })

                    plt.close()

                bike_activity_slices_plotted_count += 1

                if not quiet:
                    logger.log_line("✓️ Plotting " + file_name, console=False, file=True)

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " plotted " + str(bike_activity_slices_plotted_count) + " slices")
