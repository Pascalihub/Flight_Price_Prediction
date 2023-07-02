
import os
import urllib.request as request
import zipfile
from src.flightprice.logger import logging
from src.flightprice.utils.common import get_size
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.flightprice.logger import logging
import pandas as pd
import requests
import urllib
import sys



class DataIngestion:
    def __init__(self, data_ingestion_config):
        self.config = data_ingestion_config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = urllib.request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            print(f"{filename} downloaded with the following info:\n{headers}")
        else:
            print(f"File already exists of size: {os.path.getsize(self.config.local_data_file)}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)