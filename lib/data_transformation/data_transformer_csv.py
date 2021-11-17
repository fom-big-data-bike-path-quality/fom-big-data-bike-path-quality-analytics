import csv
import glob
import json
import os
import inspect
from pathlib import Path
from tracking_decorator import TrackingDecorator


def convert_bike_activity_to_csv_file(results_path, results_file_name, data):
    bike_activity = data['bikeActivity']
    bike_activity_samples_with_measurements = data['bikeActivitySamplesWithMeasurements']

    bike_activity_uid = bike_activity["uid"]
    bike_activity_surface_type = bike_activity["surfaceType"]
    bike_activity_smoothness_type = bike_activity["smoothnessType"]
    bike_activity_phone_position = bike_activity["phonePosition"]
    bike_activity_bike_type = bike_activity["bikeType"]
    bike_activity_flagged_lab_conditions = False

    if "flaggedLabConditions" in bike_activity:
        bike_activity_flagged_lab_conditions = bike_activity["flaggedLabConditions"]

    with open(results_path + "/" + results_file_name, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([
            # Descriptive values
            'bike_activity_uid',
            'bike_activity_sample_uid',
            'bike_activity_measurement',
            'bike_activity_measurement_timestamp',
            'bike_activity_measurement_lon',
            'bike_activity_measurement_lat',
            # Input values
            'bike_activity_measurement_speed',
            'bike_activity_measurement_accelerometer_x',
            'bike_activity_measurement_accelerometer_y',
            'bike_activity_measurement_accelerometer_z',
            'bike_activity_phone_position',
            'bike_activity_bike_type',
            'bike_activity_flagged_lab_conditions',
            # Output values
            'bike_activity_surface_type',
            'bike_activity_smoothness_type',
        ])

        for bike_activity_sample_with_measurements in bike_activity_samples_with_measurements:

            bike_activity_sample = bike_activity_sample_with_measurements["bikeActivitySample"]
            bike_activity_measurements = bike_activity_sample_with_measurements["bikeActivityMeasurements"]

            bike_activity_sample_uid = bike_activity_sample["uid"]
            bike_activity_sample_surface_type = bike_activity_sample[
                "surfaceType"] if "surfaceType" in bike_activity_sample else None

            for bike_activity_measurement in bike_activity_measurements:
                bike_activity_measurement_uid = bike_activity_measurement["uid"]
                bike_activity_measurement_timestamp = bike_activity_measurement["timestamp"]
                bike_activity_measurement_lon = bike_activity_measurement["lon"]
                bike_activity_measurement_lat = bike_activity_measurement["lat"]
                bike_activity_measurement_speed = bike_activity_measurement["speed"]
                bike_activity_measurement_accelerometer_x = bike_activity_measurement["accelerometerX"]
                bike_activity_measurement_accelerometer_y = bike_activity_measurement["accelerometerY"]
                bike_activity_measurement_accelerometer_z = bike_activity_measurement["accelerometerZ"]

                csv_writer.writerow([
                    bike_activity_uid,
                    bike_activity_sample_uid,
                    bike_activity_measurement_uid,
                    bike_activity_measurement_timestamp,
                    bike_activity_measurement_lon,
                    bike_activity_measurement_lat,
                    # Input values
                    bike_activity_measurement_speed,
                    bike_activity_measurement_accelerometer_x,
                    bike_activity_measurement_accelerometer_y,
                    bike_activity_measurement_accelerometer_z,
                    bike_activity_phone_position,
                    bike_activity_bike_type,
                    bike_activity_flagged_lab_conditions,
                    # Output values
                    bike_activity_sample_surface_type if bike_activity_sample_surface_type is not None else bike_activity_surface_type,
                    bike_activity_smoothness_type,
                ])


#
# Main
#


class DataTransformerCsv:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, results_path, clean=False, reconvert=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*"))
            for f in files:
                os.remove(f)

        for file_path in glob.iglob(data_path + "/*.json"):

            file_name = os.path.basename(file_path)
            file_base_name = file_name.replace(".json", "")

            results_file_name = file_base_name + ".csv"
            results_file_path = results_path + "/" + results_file_name

            if not Path(results_file_path).exists() or reconvert:
                file = open(file_path)
                data = json.load(file)

                logger.log_line("✓️ Converting into csv " + file_name)
                convert_bike_activity_to_csv_file(results_path, results_file_name, data)

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        logger.log_line(
                class_name + "." + function_name + " finished")
