from src.flightprice.config.configuration import ConfigurationManager
from src.flightprice.components.data_ingestion import DataIngestion
from src.flightprice.components.data_transformation import DataTransformation
from src.flightprice.logger import logging


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(data_transformation_config)
        data_transformation.initiate_data_transformation()