import glob
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import firestore_v1 as firestore_v1


def load_private_key(script_path, firebase_private_key_file):
    cert_path = os.path.join(script_path, firebase_private_key_file)
    cred = credentials.Certificate(cert_path)
    return cred


def open_database_connection(cred, firebase_database_url, firebase_collection_name):
    firebase_admin.initialize_app(cred, {"databaseURL": firebase_database_url})
    db = firestore.client()
    coll_ref = db.collection(firebase_collection_name)
    return coll_ref


def download_data(coll_ref):
    existing = list(map(lambda e: coll_ref.document(os.path.basename(e).replace('.json', '')), glob.glob(script_path + "/../data/*.json")))

    for doc in coll_ref.where(firestore_v1.field_path.FieldPath.document_id(), "not-in", existing).stream():
        file_name = doc.id + ".json"

        print("✔️ Downloading " + script_path + "/../data/" + file_name)

        json_file = open(script_path + "/../data/" + file_name, "w")
        json.dump(doc.to_dict(), json_file)
        json_file.close()


#
# Main
#

# Set script path
script_path = os.path.dirname(__file__)

# Set project specific parameters
firebase_database_url = "https://bike-path-quality.firebaseio.com/"
firebase_private_key_file = "bike-path-quality-firebase-adminsdk-cgjm5-640ce5b722.json"
firebase_collection_name = "BikeActivities"

# Load connection credentials
cred = load_private_key(script_path, firebase_private_key_file)

# Retrieve collection reference
coll_ref = open_database_connection(cred, firebase_database_url, firebase_collection_name)

# Download data
download_data(coll_ref)

print("Finished.")
