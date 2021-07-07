import glob
import os
import csv


#
# Main
#


class EpochSplitter:

    def run(self, data_path, results_path, clean=False):
        # Make results path
        os.makedirs(results_path, exist_ok=True)

        # Clean results path
        if clean:
            files = glob.glob(os.path.join(results_path, "*"))
            for f in files:
                os.remove(f)

        for file_path in glob.iglob(data_path + "/*.csv"):
            file_name = os.path.basename(file_path)
            file_base_name = file_name.replace(".csv", "")

            with open(file_path) as csv_file:
                csv_reader = csv.DictReader(csv_file)

                epochs = {}
                for row in csv_reader:

                    # Determine bike activity UID and bike activity sample UID
                    bike_activity_uid = row["bike_activity_uid"]
                    bike_activity_sample_uid = row["bike_activity_sample_uid"]

                    if bike_activity_sample_uid not in epochs:
                        # Make results path
                        os.makedirs(results_path + "/" + bike_activity_uid, exist_ok=True)

                        # Create file and append header
                        out_file = open(results_path + "/" + bike_activity_uid + "/" + bike_activity_sample_uid + ".csv", "w")
                        csv_writer = csv.DictWriter(out_file, fieldnames=csv_reader.fieldnames)
                        csv_writer.writeheader()
                        epochs[bike_activity_sample_uid] = out_file, csv_writer

                    # Append row
                    epochs[bike_activity_sample_uid][1].writerow(row)
                # Close all the files
                for file, _ in epochs.values():
                    file.close()

            print("✔️ Splitting into epochs " + file_name)

        print("EpochSplitter finished.")
