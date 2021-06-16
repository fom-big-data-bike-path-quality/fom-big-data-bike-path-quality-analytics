import glob
import json
import math
import os
from pathlib import Path

from geojson import FeatureCollection

# Configuration
USER_UID_STEFAN = "00000000-0000-0000-0000-000000000000"


def get_average_accelerometer_data(bike_activity_measurements):
    values = []

    for bike_activity_measurement in bike_activity_measurements:
        x = bike_activity_measurement["accelerometerX"]
        y = bike_activity_measurement["accelerometerY"]
        z = bike_activity_measurement["accelerometerZ"]

        values.append(math.sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)))

    return sum(values) / len(values)


def write_bike_activity_samples_to_geojson(results_path, results_file_name, bike_activity_samples_with_measurements):
    features = []
    for bike_activity_sample_with_measurements in bike_activity_samples_with_measurements:
        feature = {}

        bike_activity_sample = bike_activity_sample_with_measurements["bikeActivitySample"]
        bike_activity_measurements = bike_activity_sample_with_measurements["bikeActivityMeasurements"]

        feature["geometry"] = {"type": "Point", "coordinates": [bike_activity_sample["lon"], bike_activity_sample["lat"]]}
        feature["type"] = "Feature"
        feature["properties"] = {
            "timestamp": bike_activity_sample["timestamp"],
            "speed": bike_activity_sample["speed"],
            "accelerometer": get_average_accelerometer_data(bike_activity_measurements)
        }
        features.append(feature)

    collection = FeatureCollection(features)

    with open(results_path + "/" + results_file_name, "w") as f:
        f.write("%s" % collection)


def get_rider_name(user_data_uid):
    if user_data_uid == USER_UID_STEFAN:
        return "Stefan P"
    else:
        return "Florian S"


#
# Main
#


class DataTransformerGeoJson:

    def run(self, reconvert=False):
        # Set script path
        script_path = os.path.dirname(__file__)
        data_path = script_path + "/../data/measurements/json"
        results_path = script_path + "/../results/measurements/geojson"

        os.makedirs(results_path, exist_ok=True)

        for file_path in glob.iglob(data_path + "/*.json"):

            file_name = os.path.basename(file_path)
            file_base_name = file_name.replace(".json", "")

            results_file_name = file_base_name + ".geojson"
            results_file_path = results_path + "/" + results_file_name

            if not Path(results_file_path).exists() or reconvert:
                file = open(file_path)
                data = json.load(file)

                user_data_uid = data['userData']['uid']
                rider_name = get_rider_name(user_data_uid)
                bike_activity_samples_with_measurements = data['bikeActivitySamplesWithMeasurements']

                print("✔️ Converting into geojson " + file_name + " (" + rider_name + ")")

                write_bike_activity_samples_to_geojson(
                    results_path=results_path,
                    results_file_name=file_base_name + ".geojson",
                    bike_activity_samples_with_measurements=bike_activity_samples_with_measurements
                )

        print("DataTransformerGeoJson finished.")
