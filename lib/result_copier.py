import os
import shutil


class ResultCopier:

    def copyDirectory(self, source_dir, destination_dir, symlinks=False, ignore=None):

        shutil.rmtree(destination_dir, ignore_errors=True)
        os.mkdir(destination_dir)

        for item in os.listdir(source_dir):
            source_item = os.path.join(source_dir, item)
            destination_item = os.path.join(destination_dir, item)
            if os.path.isdir(source_item):
                shutil.copytree(source_item, destination_item, symlinks, ignore)
            else:
                shutil.copy2(source_item, destination_item)
