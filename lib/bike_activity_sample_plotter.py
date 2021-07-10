import csv
import glob
import os
from email.utils import formatdate
from pathlib import Path

import matplotlib.pyplot as plt


#
# Main
#

class BikeActivitySamplePlotter:

    def run(self, data_path, results_path, xlabel, ylabel, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*"))
            for f in files:
                os.remove(f)

        for file_path in Path(data_path).rglob("*.csv"):
            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            with open(str(file_path)) as csv_file:

                csv_reader = csv.DictReader(csv_file)

                data = []

                for row in csv_reader:
                    bike_activity_measurement_accelerometer = row["bike_activity_measurement_accelerometer"]
                    data.append(float(bike_activity_measurement_accelerometer))

                plt.figure(2)
                plt.clf()
                plt.title("Bike activity sample " + file_name)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.plot(data)

                plt.savefig(fname=results_path + "/" + file_base_name + ".png",
                            format="png",
                            metadata={
                                "Title": "Bike activity sample " + file_base_name,
                                "Author": "Florian Schwanz",
                                "Creation Time": formatdate(timeval=None, localtime=False, usegmt=True),
                                "Description": "Plot of bike activity sample " + file_base_name
                            })

                plt.close()

            print("✔️ Plotting " + file_name)

        print("BikeActivitySamplePlotter finished.")
