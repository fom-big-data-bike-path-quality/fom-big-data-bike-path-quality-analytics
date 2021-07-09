import csv
import glob
import math
import os
from pathlib import Path


#
# Main
#


class DataTransformer:

    def run(self, data_path, results_path, clean=False):
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

                with open(os.path.join(results_path, file_name), "w") as out_file:
                    # Make results path
                    os.makedirs(results_path, exist_ok=True)

                    csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames + ["bike_activity_measurement_accelerometer"])
                    csv_writer.writeheader()

                    for row in csv_reader:
                        # Determine bike activity UID and bike activity sample UID
                        bike_activity_uid = row["bike_activity_uid"]
                        bike_activity_sample_uid = row["bike_activity_sample_uid"]

                        bike_activity_measurement_accelerometer_x = float(row["bike_activity_measurement_accelerometer_x"])
                        bike_activity_measurement_accelerometer_y = float(row["bike_activity_measurement_accelerometer_y"])
                        bike_activity_measurement_accelerometer_z = float(row["bike_activity_measurement_accelerometer_z"])
                        bike_activity_measurement_accelerometer = math.sqrt((bike_activity_measurement_accelerometer_x ** 2
                                                                            + bike_activity_measurement_accelerometer_y ** 2
                                                                            + bike_activity_measurement_accelerometer_z ** 2) / 3)

                        row.update({"bike_activity_measurement_accelerometer": bike_activity_measurement_accelerometer})
                        csv_writer.writerow(row)

            print("✔️ Transforming " + file_name)

        print("DataTransformer finished.")
