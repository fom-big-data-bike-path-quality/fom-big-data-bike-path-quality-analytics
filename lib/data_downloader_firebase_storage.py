import os
from pathlib import Path

import pyrebase


class FirebaseStorageDownloader():

    def run(self, reload=False):
        # Set script path
        script_path = os.path.dirname(__file__)

        firebase_private_key_file = "bike-path-quality-firebase-adminsdk-cgjm5-640ce5b722.json"
        config = {
            "apiKey": "apiKey",
            "authDomain": "bike-path-quality.firebaseapp.com",
            "databaseURL": "https://bike-path-quality.firebaseio.com",
            "storageBucket": "bike-path-quality.appspot.com",
            "serviceAccount": os.path.join(script_path, firebase_private_key_file)
        }

        firebase = pyrebase.initialize_app(config)

        storage = firebase.storage()
        files = storage.list_files()
        for file in files:
            file_name = file.name
            file_path = script_path + "/../data/" + file_name

            if (not Path(file_path).exists() or reload):
                print("✔️ Downloading " + script_path + "/../data/" + file_name)
                storage.child(file_name).download(script_path + "/../data/" + file_name)

        print("FirebaseStorageDownloader finished.")
