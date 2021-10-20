import os
import shutil
from zipfile import ZipFile

from google_cloud_platform_bucket_uploader import GoogleCloudPlatformBucketUploader


class ResultHandler:

    def copy_directory(self, source_dir, destination_dir, symlinks=False, ignore=None):

        if destination_dir is not None:

            shutil.rmtree(destination_dir, ignore_errors=True)
            os.mkdir(destination_dir)

            for item in os.listdir(source_dir):
                source_item = os.path.join(source_dir, item)
                destination_item = os.path.join(destination_dir, item)
                if os.path.isdir(source_item):
                    shutil.copytree(source_item, destination_item, symlinks, ignore)
                else:
                    shutil.copy2(source_item, destination_item)

    def zip_directory(self, source_dir, destination_dir, zip_name, zip_root_dir):

        with ZipFile(os.path.join(destination_dir, zip_name), "w") as zipFile:
            lenDirPath = len(source_dir)
            for root, _, files in os.walk(source_dir):
                for file in files:
                    filePath = os.path.join(root, file)
                    zipFile.write(filePath, zip_root_dir + filePath[lenDirPath:])
            zipFile.close()

    def upload_results(self, logger, upload_file_path, project_id, bucket_name, quiet=False):
        GoogleCloudPlatformBucketUploader().upload_file(
            logger=logger,
            upload_file_path=upload_file_path,
            project_id=project_id,
            bucket_name=bucket_name,
            quiet=quiet
        )
