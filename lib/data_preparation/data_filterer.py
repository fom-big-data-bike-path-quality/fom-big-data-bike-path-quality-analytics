import inspect

from tracking_decorator import TrackingDecorator


#
# Main
#

class DataFilterer:

    @TrackingDecorator.track_time
    def run(self, logger, dataframes, slice_width, measurement_speed_limit, quiet=False):

        copied_dataframes = dataframes.copy()

        dataframes_count = len(copied_dataframes.items())

        for name, dataframe in list(copied_dataframes.items()):

            # Exclude dataframes which are not tracked under lab conditions
            if False in dataframe.bike_activity_flagged_lab_conditions.values:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (not tracked under lab conditions)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain less than x measurements
            if len(dataframe) < slice_width:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (less than " + str(slice_width) + " measurements)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain surface type 'mixed'
            if "mixed" in dataframe.bike_activity_surface_type.values:
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (containing undefined surface type)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain speeds slower than BIKE_ACTIVITY_MEASUREMENT_SPEED_LIMIT
            if (dataframe.bike_activity_measurement_speed.values * 3.6 < measurement_speed_limit).any():
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (containing slow measurements)")
                copied_dataframes.pop(name)
                continue

            # Exclude dataframes which contain invalid location
            if ((dataframe.bike_activity_measurement_lon.values == 0.0).any() and
                    (dataframe.bike_activity_measurement_lat.values == 0.0).any()):
                if not quiet:
                    logger.log_line("✗️ Filtering out " + name + " (containing invalid location)")
                copied_dataframes.pop(name)
                continue

            if not quiet:
                logger.log_line("✓️ Keeping " + name)

        dataframes_filtered_count = len(copied_dataframes.items())

        if not quiet:
            class_name = self.__class__.__name__
            function_name = inspect.currentframe().f_code.co_name

            logger.log_line(
                class_name + "." + function_name + " kept " + str(dataframes_filtered_count) + "/"
                + str(dataframes_count) + " dataframes ("
                + str(round(dataframes_filtered_count / dataframes_count, 2) * 100) + "%)")

        return copied_dataframes
