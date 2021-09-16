import csv
import glob
import os
from pathlib import Path

import numpy as np


#
# Main
#


class SlicingDataSplitter:

    def run(self, logger, data_path, results_path, clean=False):

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

                    if not Path(result_file).exists() or clean:

                        if bike_activity_sample_uid not in slices:
                            # Make results path
                            os.makedirs(results_path + "/" + bike_activity_uid, exist_ok=True)

                            # Create file and append header
                            out_file = open(result_file, "w")
                            csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames)
                            csv_writer.writeheader()
                            slices[bike_activity_sample_uid] = out_file, csv_writer

                        # Append row
                        slices[bike_activity_sample_uid][1].writerow(row)

                        slices_count += 1

                # Close all the files
                for file, _ in slices.values():
                    file.close()

            slices_count_total += slices_count

            if slices_count > 0:
                logger.log_line("✓️ Splitting into slices " + file_name)

        logger.log_line("Slicing data splitter finished with " + str(slices_count_total) + " slices")
