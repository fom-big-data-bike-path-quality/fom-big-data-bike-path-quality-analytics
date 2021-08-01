import csv
import glob
import math
import os
from email.utils import formatdate
from pathlib import Path

import matplotlib.pyplot as plt


#
# Main
#

class BikeActivitySlicePlotter:

    def run(self, logger, data_path, results_path, xlabel, ylabel, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.png"))
            for f in files:
                os.remove(f)

        bike_activity_slices_plotted_count = 0

        for file_path in Path(data_path).rglob("*.csv"):
            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            results_file = results_path + "/" + file_base_name + ".png"

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
                    plt.plot(data_accelerometer, label="accelerometer")
                    plt.plot(data_speed, label="speed")
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
                logger.log_line("✓️ Plotting " + file_name)

        logger.log_line("Bike activity slice plotter finished with " + str(bike_activity_slices_plotted_count) + " slices plotted")
