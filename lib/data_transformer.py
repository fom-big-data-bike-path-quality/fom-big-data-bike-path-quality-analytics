import csv
import glob
import math
import os
from pathlib import Path


#
# Main
#


class DataTransformer:
    BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT = 5

    def run(self, data_path, results_path, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.csv"))
            for f in files:
                os.remove(f)

        file_count_total = 0
        file_count_processed = 0

        for file_path in Path(data_path).rglob("*.csv"):

            file_name = os.path.basename(file_path.name)
            file_base_name = file_name.replace(".csv", "")

            file_count_total += 1

            bike_activity_measurement_speed_valid = True
            bike_activity_measurement_surface_type = True

            with open(str(file_path)) as csv_file:

                csv_reader = csv.DictReader(csv_file)

                with open(os.path.join(results_path, file_name), "w") as out_file:
                    # Make results path
                    os.makedirs(results_path, exist_ok=True)

                    csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames + ["bike_activity_measurement_accelerometer"])
                    csv_writer.writeheader()

                    for row in csv_reader:
                        bike_activity_uid = row["bike_activity_uid"]
                        bike_activity_sample_uid = row["bike_activity_sample_uid"]
                        bike_activity_measurement_speed = float(row["bike_activity_measurement_speed"])
                        bike_activity_surface_type = row["bike_activity_surface_type"]

                        if bike_activity_measurement_speed * 3.6 < self.BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT:
                            bike_activity_measurement_speed_valid = False
                            break

                        if bike_activity_surface_type is not 'mixed':
                            bike_activity_measurement_surface_type = False
                            break

                        bike_activity_measurement_accelerometer_x = float(row["bike_activity_measurement_accelerometer_x"])
                        bike_activity_measurement_accelerometer_y = float(row["bike_activity_measurement_accelerometer_y"])
                        bike_activity_measurement_accelerometer_z = float(row["bike_activity_measurement_accelerometer_z"])
                        bike_activity_measurement_accelerometer = math.sqrt((bike_activity_measurement_accelerometer_x ** 2
                                                                             + bike_activity_measurement_accelerometer_y ** 2
                                                                             + bike_activity_measurement_accelerometer_z ** 2) / 3)

                        row.update({"bike_activity_measurement_accelerometer": bike_activity_measurement_accelerometer})
                        csv_writer.writerow(row)

            if not bike_activity_measurement_speed_valid or not bike_activity_measurement_surface_type:
                os.remove(os.path.join(results_path, file_name))

            if not bike_activity_measurement_speed_valid:
                print("✗️ Skipping " + file_name + " (containing slow measurements)")
            elif not bike_activity_measurement_surface_type:
                print("✗️ Skipping " + file_name + " (containing undefined surface type)")
            else:
                print("✓️ Transforming " + file_name)
                file_count_processed += 1

        print("Processed " + str(file_count_processed) + "/" + str(file_count_total))
        print("DataTransformer finished.")
