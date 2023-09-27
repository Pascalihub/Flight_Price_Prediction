import pandas as pd
from src.flightprice.exception import CustomException
from src.flightprice.logger import logging
from src.flightprice.utils.common import load_object
import sys
import pandas as pd
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model_trainer/model.pkl"
            preprocessor_path = "artifacts/data_transformation/preprocessing_obj.pkl"
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
   
        except Exception as e:
            raise CustomException(e, sys) 

class CustomData:
    def __init__(  self,
        Total_stops,
        Journey_day,
        Journey_month,
        Dep_hour,
        Dep_min,
        Arrival_hour,
        Arrival_min,
        dur_hour,
        dur_min,
        Air_India,
        GoAir,
        IndiGo,
        Jet_Airways,
        Jet_Airways_Business,
        Multiple_carriers,
        Multiple_carriers_Premium_economy,
        SpiceJet,
        Trujet,
        Vistara,
        Vistara_Premium_economy,
        s_Chennai,
        s_Delhi,
        s_Kolkata,
        s_Mumbai,
        d_Cochin,
        d_Delhi,
        d_Hyderabad,
        d_Kolkata,
        d_New_Delhi):


        self.Total_stops = Total_stops
        self.Journey_day = Journey_day
        self.Journey_month = Journey_month
        self.Dep_hour = Dep_hour
        self.Dep_min = Dep_min
        self.Arrival_hour = Arrival_hour
        self.Arrival_min = Arrival_min
        self.dur_hour = dur_hour
        self.dur_min = dur_min
        self.Air_India = Air_India
        self.GoAir = GoAir
        self.IndiGo = IndiGo
        self.Jet_Airways = Jet_Airways
        self.Jet_Airways_Business = Jet_Airways_Business
        self.Multiple_carriers = Multiple_carriers
        self.Multiple_carriers_Premium_economy = Multiple_carriers_Premium_economy
        self.SpiceJet = SpiceJet
        self.Trujet = Trujet
        self.Vistara = Vistara
        Vistara_Premium_economy = Vistara_Premium_economy
        self.s_Chennai = s_Chennai
        self.s_Delhi = s_Delhi
        self.s_Kolkata = s_Kolkata
        self.s_Mumbai = s_Mumbai
        self.d_Cochin = d_Cochin
        self.d_Delhi = d_Delhi
        self.d_Hyderabad = d_Hyderabad
        self.d_Kolkata = d_Kolkata
        self.d_New_Delhi = d_New_Delhi
        
    @staticmethod
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Total_stops": [self.Total_stops],
                "Journey_day": [self.Journey_day],
                "Journey_month": [self.Journey_month],
                "Dep_hour": [self.Dep_hour],
                "Dep_min": [self.Dep_min],
                "Arrival_hour": [self.Arrival_hour],
                "Arrival_min": [self.Arrival_min],
                "dur_hour": [self.dur_hour],
                "dur_min": [self.dur_min],
                "Air_India": [self.Air_India],
                "GoAir": [self.GoAir],
                "IndiGo": [self.IndiGo],
                "Jet_Airways": [self.Jet_Airways],
                "Jet_Airways_Business": [self.Jet_Airways_Business],
                "Multiple_carriers": [self.Multiple_carriers],
                "Multiple_carriers_Premium_economy": [self.Multiple_carriers_Premium_economy],
                "SpiceJet": [self.SpiceJet],
                "Trujet": [self.Trujet],
                "Vistara": [self.Vistara],
                "Vistara_Premium_economy": [self.Vistara_Premium_economy],
                "s_Chennai": [self.s_Chennai],
                "s_Delhi": [self.s_Delhi],
                "s_Kolkata": [self.s_Kolkata],
                "s_Mumbai": [self.s_Mumbai],
                "d_Cochin": [self.d_Cochin],
                "d_Delhi": [self.d_Delhi],
                "d_Hyderabad": [self.d_Hyderabad],
                "d_Kolkata": [self.d_Kolkata],
                "d_New_Delhi": [self.d_New_Delhi]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
