from src.flightprice.config.configuration import ConfigurationManager
from src.flightprice.components.data_ingestion import DataIngestion
from src.flightprice.components.data_transformation import DataTransformation
from src.flightprice.components.data_validation import DataValiadtion
from src.flightprice.logger import logging


class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_files_exist()