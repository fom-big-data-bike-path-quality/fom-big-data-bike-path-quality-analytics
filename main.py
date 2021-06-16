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
from data_downloader_firebase_firestore import FirebaseFirestoreDownloader
from data_downloader_firebase_storage import FirebaseStorageDownloader

# Configuration
RELOAD_DATA = False

# Download data from Firebase Firestore
# FirebaseFirestoreDownloader().run(RELOAD_DATA)
FirebaseStorageDownloader().run(RELOAD_DATA)
