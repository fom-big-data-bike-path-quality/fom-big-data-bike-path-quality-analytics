import os
import glob

import pyrebase


#
# Main
#


class DataUploaderGeoJson:

    def run(self):
        # Set script path
        script_path = os.path.dirname(__file__)
        data_path = script_path + "/../results/measurements/geojson"

        firebase_private_key_file = "bike-path-quality-firebase-adminsdk-cgjm5-640ce5b722.json"
        config = {
            "apiKey": "apiKey",
            "authDomain": "bike-path-quality.firebaseapp.com",
            "databaseURL": "https://bike-path-quality.firebaseio.com",
            "storageBucket": "bike-path-quality.appspot.com",
            "serviceAccount": os.path.join(script_path, firebase_private_key_file)
        }

        firebase = pyrebase.initialize_app(config)

        for file_path in glob.iglob(data_path + "/*.geojson"):

            file_name = os.path.basename(file_path)
            print("✔️ Uploading " + file_name)
            firebase.storage().child("measurements").child("geojson").child(file_name).put(file_path)

        print("DataUploaderGeoJson finished.")
