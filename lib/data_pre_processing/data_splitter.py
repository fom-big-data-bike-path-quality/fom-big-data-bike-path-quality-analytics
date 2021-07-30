import csv
import glob
import os
from pathlib import Path


#
# Main
#


class DataSplitter:

    def run(self, data_path, results_path, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*.csv"))
            for f in files:
                os.remove(f)

        for file_path in glob.iglob(data_path + "/*.csv"):
            file_name = os.path.basename(file_path)
            file_base_name = file_name.replace(".csv", "")

            slices_count = 0

            with open(file_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)

                slices = {}
                for row in csv_reader:

                    # Determine bike activity UID and bike activity sample UID
                    bike_activity_uid = row["bike_activity_uid"]
                    bike_activity_sample_uid = row["bike_activity_sample_uid"]

                    result_file = results_path + "/" + bike_activity_uid + "/" + bike_activity_sample_uid + ".csv"

                    if not Path(result_file).exists() or clean:

                        if bike_activity_sample_uid not in slices:
                            # Make results path
                            os.makedirs(results_path + "/" + bike_activity_uid, exist_ok=True)

                            # Create file and append header
                            out_file = open(results_path + "/" + bike_activity_uid + "/" + bike_activity_sample_uid + ".csv", "w")
                            csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames)
                            csv_writer.writeheader()
                            slices[bike_activity_sample_uid] = out_file, csv_writer

                        # Append row
                        slices[bike_activity_sample_uid][1].writerow(row)

                        slices_count += 1

                # Close all the files
                for file, _ in slices.values():
                    file.close()

            if slices_count > 0:
                print("✓️ Splitting into slices " + file_name)

        print("Data Splitter finished with " + str(len(slices.items())) + " slices")
