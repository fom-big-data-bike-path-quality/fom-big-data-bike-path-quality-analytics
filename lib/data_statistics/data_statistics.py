from tracking_decorator import TrackingDecorator
from label_encoder import LabelEncoder

def get_bike_activity_measurement_speed_min(slice):
    bike_activity_measurement_speed_min = None

    for row in slice:
        bike_activity_measurement_speed = float(row["bike_activity_measurement_speed"])

        if bike_activity_measurement_speed_min == None or bike_activity_measurement_speed < bike_activity_measurement_speed_min:
            bike_activity_measurement_speed_min = bike_activity_measurement_speed

    return bike_activity_measurement_speed_min


#
# Main
#

class DataStatistics:

    @TrackingDecorator.track_time
    def run(self, dataframes):

        surface_types = {}

        valid_surface_types = [
            "asphalt",
            "concrete lanes",
            "concrete plates",
            "paving stones",
            "sett",
            "compacted",
            "fine gravel",
            "gravel"
        ]

        for surface_type in valid_surface_types:
            surface_types[surface_type] = 0

        for dataframe in dataframes.values():
            bike_activity_surface_type = dataframe["bike_activity_surface_type"][0]
            bike_activity_surface_type_label = LabelEncoder().index_to_label(index=bike_activity_surface_type)

            if bike_activity_surface_type_label in valid_surface_types:
                surface_types[bike_activity_surface_type_label] += 1

        return surface_types
