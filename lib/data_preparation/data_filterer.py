import inspect

from tracking_decorator import TrackingDecorator


#
# Main
#

class DataFilterer:
    BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT = 5  # in km/h

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, quiet=False):

        copied_dataframes = dataframes.copy()

        dataframes_count = len(copied_dataframes.items())

        for name, dataframe in list(copied_dataframes.items()):

            # Exclude dataframes which are not tracked under lab conditions
            if False in dataframe.bike_activity_flagged_lab_conditions.values:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (not tracked under lab conditions)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain less than 500 measurements
            if len(dataframe) < 500:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (less than 500 measurements)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain surface type 'mixed'
            if "mixed" in dataframe.bike_activity_surface_type.values:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (containing undefined surface type)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain speeds slower than BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT
            if (dataframe.bike_activity_measurement_speed.values * 3.6 < self.BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT).any():
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (containing slow measurements)")
                copied_dataframes.pop(name)
                continue

            if not quiet:
                logger.log_line("✓️ Keeping " + name)

        dataframes_filtered_count = len(copied_dataframes.items())

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(class_name + "." + function_name + " kept " + str(dataframes_filtered_count) + "/" + str(dataframes_count)
                            + " dataframes (" + str(round(dataframes_filtered_count / dataframes_count, 2) * 100) + "%)")

        return copied_dataframes
