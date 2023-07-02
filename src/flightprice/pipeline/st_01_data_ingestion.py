from src.flightprice.config.configuration import ConfigurationManager
from src.flightprice.components.data_ingestion import DataIngestion
from src.flightprice.logger import logging


class DataIngestionIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()