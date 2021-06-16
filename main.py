import os
import sys

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib')
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from data_downloader_firebase_storage import FirebaseStorageDownloader
from data_transformer_geojson import DataTransformerGeoJson

# Configuration
RELOAD_DATA = False
RECONVERT_DATA = False

# Download data from Firebase Firestore
FirebaseStorageDownloader().run(RELOAD_DATA)

# Convert data from json to geojson
DataTransformerGeoJson().run(RECONVERT_DATA)
