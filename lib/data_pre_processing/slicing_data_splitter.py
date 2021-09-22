import csv
import glob
import inspect
import os

from tracking_decorator import TrackingDecorator


#
# Main
#

class SlicingDataSplitter:

    @TrackingDecorator.track_time
    def run(self, logger, data_path, slice_width, results_path, clean=False, quiet=False):

        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.csv"))
            for f in files:
                os.remove(f)

        slices_count_total = 0

        for file_path in glob.iglob(data_path + "/*.csv"):

            file_name = os.path.basename(file_path)
            slices_count = 0

            with open(file_path) as csv_file:

                csv_reader = csv.DictReader(csv_file)
                slices = {}

                for row in csv_reader:

                    # Determine bike activity UID and bike activity sample UID
                    bike_activity_uid = row["bike_activity_uid"]
                    bike_activity_sample_uid = row["bike_activity_sample_uid"]

                    result_file = os.path.join(results_path, bike_activity_uid, bike_activity_sample_uid + ".csv")

                    # Create result file if not yet existing
                    if bike_activity_sample_uid not in slices:

                        # Make results path
                        os.makedirs(results_path + "/" + bike_activity_uid, exist_ok=True)

                        # Create file and append header
                        out_file = open(result_file, "w")
                        csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames)
                        csv_writer.writeheader()
                        slices[bike_activity_sample_uid] = out_file, csv_writer, []

                        slices_count += 1

                    # Append row
                    slices[bike_activity_sample_uid][2].append(row)

                # Write row
                for out_file, csv_writer, rows in slices:
                    for row in rows[:slice_width]:
                        csv_writer.writerow(row)

                # Close files
                for out_file, _, _ in slices.values():
                    out_file.close()

            slices_count_total += slices_count

            if slices_count > 0 and not quiet:
                logger.log_line("✓️ Splitting into slices " + file_name)

        class_name = self.__class__.__name__
        function_name = inspect.currentframe().f_code.co_name

        if not quiet:
            logger.log_line(class_name + "." + function_name + " splitted data in " + str(slices_count_total) + " slices")
