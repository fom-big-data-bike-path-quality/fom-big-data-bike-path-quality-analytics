import csv
import glob
import os

from tracking_decorator import TrackingDecorator


#
# Main
#

def get_bike_activity_measurement_speed_min(slice):
    bike_activity_measurement_speed_min = None

    for row in slice:
        bike_activity_measurement_speed = float(row["bike_activity_measurement_speed"])

        if bike_activity_measurement_speed_min == None or bike_activity_measurement_speed < bike_activity_measurement_speed_min:
            bike_activity_measurement_speed_min = bike_activity_measurement_speed

    return bike_activity_measurement_speed_min

class InputDataStatistics:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, measurement_speed_limit, clean=False, quiet=False):

        slices = {}
        surface_types = {}

        for file_path in glob.iglob(data_path + "/*.csv"):

            with open(file_path) as csv_file:

                csv_reader = csv.DictReader(csv_file)

                for row in csv_reader:

                    # Determine bike activity UID and bike activity sample UID
                    bike_activity_uid = row["bike_activity_uid"]
                    bike_activity_sample_uid = row["bike_activity_sample_uid"]

                    # Create result file if not yet existing
                    if bike_activity_sample_uid not in slices:
                        slices[bike_activity_sample_uid] = []

                    # Append row
                    slices[bike_activity_sample_uid].append(row)

        for bike_activity_sample_uid, slice in slices.items():
            bike_activity_flagged_lab_conditions = slice[0]["bike_activity_flagged_lab_conditions"]
            bike_activity_surface_type = slice[0]["bike_activity_surface_type"]
            bike_activity_measurement_speed_min = get_bike_activity_measurement_speed_min(slice)

            if bike_activity_flagged_lab_conditions == "True" and \
                    bike_activity_measurement_speed_min * 3.6 >= measurement_speed_limit:

                if bike_activity_surface_type not in surface_types:
                    surface_types[bike_activity_surface_type] = 0

                surface_types[bike_activity_surface_type] += 1

        if not quiet:
            logger.log_line("Useful samples tracked")

            for bike_activity_surface_type, count in surface_types.items():
                logger.log_line("❄ " + bike_activity_surface_type + ": " + str(count))
