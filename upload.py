#!/usr/bin/env python3

import os
import sys

file_path = os.path.realpath(__file__)
script_path = os.path.dirname(file_path)

# Make library available in path
library_paths = [
    os.path.join(script_path, 'lib'),
    os.path.join(script_path, 'lib', 'log'),
    os.path.join(script_path, 'lib', 'data_pre_processing'),
    os.path.join(script_path, 'lib', 'data_statistics'),
    os.path.join(script_path, 'lib', 'data_preparation'),
    os.path.join(script_path, 'lib', 'plotters'),
    os.path.join(script_path, 'lib', 'models'),
    os.path.join(script_path, 'lib', 'models', 'base_model_knn_dtw'),
    os.path.join(script_path, 'lib', 'models', 'base_model_cnn'),
    os.path.join(script_path, 'lib', 'models', 'base_model_cnn', 'layers'),
    os.path.join(script_path, 'lib', 'models', 'base_model_lstm'),
    os.path.join(script_path, 'lib', 'cloud'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from logger_facade import LoggerFacade
from result_handler import ResultHandler
from tracking_decorator import TrackingDecorator


#
# Main
#

@TrackingDecorator.track_time
def main(argv):
    # Set default values
    quiet = False

    gcp_project_id = "bike-path-quality-339900"
    gcp_bucket_name = "bike-path-quality-training-results"
    gcp_token_file = "bike-path-quality-339900-a8e468a52c18.json"

    # Initialize logger
    logger = LoggerFacade("/tmp", console=True, file=True)

    ResultHandler().upload_results(
        logger=logger,
        gcp_token_file=gcp_token_file,
        upload_file_path=os.path.join(script_path, "results", "logo.png"),
        project_id=gcp_project_id,
        bucket_name=gcp_bucket_name,
        quiet=quiet
    )


if __name__ == "__main__":
    main(sys.argv[1:])
