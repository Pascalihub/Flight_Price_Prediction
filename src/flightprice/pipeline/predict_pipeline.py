import os
import sys
import pandas as pd


from src.flightprice.logger import logging
from src.flightprice.utils.common import load_object


class predict_pipeline:
    def __init__(self):
        pass

    def predict(self, max_features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds= model.predict(data_scaled)
            return preds
        except Exception as e:
            raise e


class CustomData:
    def __init__(  self,
        Airline: str,
        Date_of_Journey: int,
        Source: str ,
        Destination: str,
        Dep_Time: int,
        Arrival_Time: int,
        Duration: int,
        Total_Stops: str):

        self.Airline = Airline

        self.Date_of_Journey = Date_of_Journey

        self.Source = Source

        self.Destination = Destination

        self.Dep_Time = Dep_Time

        self.Arrival_Time = Arrival_Time

        self.Duration = Duration
        
        self.Total_Stops = Total_Stops

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Airline": [self.Airline],
                "Date_of_Journey": [self.Date_of_Journey],
                "Source": [self.Source],
                "Destination": [self.Destination],
                "Dep_Time": [self.Dep_Time],
                "Arrival_Time": [self.Arrival_Time],
                "Duration": [self.Duration],
                "Total_Stops": [self.Total_Stops],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise e

    