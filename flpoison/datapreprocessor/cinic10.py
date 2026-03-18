import os
import pickle
import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.datasets.utils import extract_archive
from torchvision.datasets.folder import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, check_integrity
import numpy as np

from flpoison.utils.global_utils import get_context_logger


class CINIC10(ImageFolder):
    base_folder = 'cinic-10'

    def __init__(self, root, train, download, transform=None, target_transform=None, logger=None):
        self.url = 'https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz'
        self.archive_filename = 'CINIC-10.tar.gz'
        self.tgz_md5 = '6ee4d0c996905fe93221de577967a372'
        self.root = root
        self.train = train
        self.logger = get_context_logger(logger, logger_name=__name__)
        self.base_folder = os.path.join(self.root, self.base_folder)
        self.data_folder = os.path.join(
            self.base_folder, 'train' if train else 'test')
        self.pkl_names = os.path.join(
            self.base_folder, f'cinic10_{"train" if self.train else "test"}.pkl')
        if download:
            self.download()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')
        super().__init__(self.data_folder, transform, target_transform)

        if self.check_pkls():
            with open(self.pkl_names, 'rb') as f:
                self.data, self.targets = pickle.load(f)
        else:
            self.data = np.array([np.array(self.loader(s[0]))
                                  for s in self.samples])
            self.targets = np.array(self.targets)
            with open(self.pkl_names, 'wb') as f:
                pickle.dump((self.data, self.targets), f)

    def check_pkls(self):
        if os.path.exists(self.pkl_names):
            return True
        return False

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def _check_data_integrity(self):
        return os.path.exists(self.data_folder)

    def _check_integrity(self):
        data_exist = self._check_data_integrity()
        if not data_exist:
            self.logger.info("CINIC-10 dataset not found. Checking archive...")
            archive = os.path.join(self.root, self.archive_filename)
            archive_exist = os.path.exists(archive)
            if archive_exist:
                self.logger.info("CINIC-10 archive found. Verifying integrity...")
                if check_integrity(archive, self.tgz_md5):
                    self.logger.info("CINIC-10 archive verified. Extracting...")
                    extract_archive(from_path=archive,
                                    to_path=self.base_folder)
                    self.logger.info("CINIC-10 extraction completed.")
                    return True
                else:
                    self.logger.error(
                        "CINIC-10 archive is corrupted. Remove it and re-download the dataset."
                    )
                    return False
            else:
                self.logger.warning("CINIC-10 archive not found.")
                return False
        return self._check_data_integrity()

    def download(self):
        if self._check_integrity():
            self.logger.info("CINIC-10 files already downloaded and verified.")
            return
        self.logger.info("Downloading CINIC-10 dataset...")
        download_and_extract_archive(
            self.url, self.root, self.base_folder, filename=self.archive_filename, md5=self.tgz_md5)
        self.logger.info("CINIC-10 download completed.")
