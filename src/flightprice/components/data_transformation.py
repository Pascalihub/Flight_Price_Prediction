import os
from src.flightprice.logger import logging
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler,OrdinalEncoder
import pickle
'''from src.flightprice.utils.common import save_object'''
from src.flightprice.entity import DataTransformationConfig
import logging
import pickle



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def parse_datetime_columns(self, df):
        df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
        df["Journey_day"] = df["Date_of_Journey"].dt.day
        df["Journey_month"] = df["Date_of_Journey"].dt.month
        df.drop(["Date_of_Journey"], axis=1, inplace=True)

        # Extracting Hours and Minutes for Departure and Arrival times
        df["Dep_hour"] = pd.to_datetime(df["Dep_Time"]).dt.hour
        df["Dep_min"] = pd.to_datetime(df["Dep_Time"]).dt.minute
        df.drop(["Dep_Time"], axis=1, inplace=True)

        df["Arrival_hour"] = pd.to_datetime(df["Arrival_Time"]).dt.hour
        df["Arrival_min"] = pd.to_datetime(df["Arrival_Time"]).dt.minute
        df.drop(["Arrival_Time"], axis=1, inplace=True)
        df.drop(["Route", "Additional_Info"], axis = 1, inplace = True)

        return df
    
    def parse_duration_column(self, df):
        duration = list(df["Duration"])

        for i in range(len(duration)):
            if len(duration[i].split()) != 2:  # Check if duration contains only hour or mins
                if "h" in duration[i]:
                    duration[i] = duration[i].strip() + " 0m"  # Adds 0 minute
                else:
                    duration[i] = "0h " + duration[i]  # Adds 0 hour

        duration_hours = []
        duration_mins = []
        for i in range(len(duration)):
            duration_hours.append(int(duration[i].split(sep="h")[0]))  # Extract hours from duration
            duration_mins.append(int(duration[i].split(sep="m")[0].split()[-1]))  # Extracts only minutes from duration

        # Adding duration_hours and duration_mins columns to the dataframe
        df["Duration_hours"] = duration_hours
        df["Duration_mins"] = duration_mins

        # Drop the original "Duration" column
        df.drop(["Duration"], axis=1, inplace=True)

        return df
    
    def parse_Total_Stops(self, df):
        df.replace({"non-stop": 0, 
                    "1 stop": 1, 
                    "2 stops": 2, 
                    "3 stops": 3, 
                    "4 stops": 4}, 
                    inplace = True)

    def get_data_transformer_obj(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define which columns should be ordinal-encoded and which should be scaled
            numerical_columns=['Journey_day', 'Journey_month', 
                                 'Dep_hour', 'Dep_min', 
                                 'Arrival_hour', 'Arrival_min', 
                                 'Duration_hours', 'Duration_mins']
            categorical_columns=[
                'Airline', 'Source', 'Destination', 'Route', 'Total_Stops', 'Additional_Info'
            ]
            
            # Define the custom ranking for each ordinal variable
            Airline = tuple(['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'Multiple carriers', 'GoAir',
            'Vistara', 'Air Asia', 'Vistara Premium economy', 'Jet Airways Business',
            'Multiple carriers Premium economy', 'Trujet'])
            Source = tuple(['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
            Destination = tuple(['Cochin', 'Banglore', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata'])
            Route = tuple(['BLR → DEL', 'CCU → IXR → BBI → BLR', 'DEL → LKO → BOM → COK',
            'CCU → NAG → BLR', 'BLR → NAG → DEL', 'CCU → BLR', 'BLR → BOM → DEL',   
            'DEL → BOM → COK', 'DEL → BLR → COK', 'MAA → CCU', 'CCU → BOM → BLR',
            'DEL → AMD → BOM → COK', 'DEL → PNQ → COK', 'DEL → CCU → BOM → COK',
            'BLR → COK → DEL', 'DEL → IDR → BOM → COK', 'DEL → LKO → COK',
            'CCU → GAU → DEL → BLR', 'DEL → NAG → BOM → COK', 'CCU → MAA → BLR',
            'DEL → HYD → COK', 'CCU → HYD → BLR', 'DEL → COK', 'CCU → DEL → BLR',
            'BLR → BOM → AMD → DEL', 'BOM → DEL → HYD', 'DEL → MAA → COK', 'BOM → HYD',
            'DEL → BHO → BOM → COK', 'DEL → JAI → BOM → COK', 'DEL → ATQ → BOM → COK',
            'DEL → JDH → BOM → COK', 'CCU → BBI → BOM → BLR', 'BLR → MAA → DEL',
            'DEL → GOI → BOM → COK', 'DEL → BDQ → BOM → COK', 'CCU → JAI → BOM → BLR',
            'CCU → BBI → BLR', 'BLR → HYD → DEL', 'DEL → TRV → COK',
            'CCU → IXR → DEL → BLR', 'DEL → IXU → BOM → COK', 'CCU → IXB → BLR',
            'BLR → BOM → JDH → DEL', 'DEL → UDR → BOM → COK', 'DEL → HYD → MAA → COK',
            'CCU → BOM → COK → BLR', 'BLR → CCU → DEL', 'CCU → BOM → GOI → BLR',
            'DEL → RPR → NAG → BOM → COK', 'DEL → HYD → BOM → COK',
            'CCU → DEL → AMD → BLR', 'CCU → PNQ → BLR', 'BLR → CCU → GAU → DEL',
            'CCU → DEL → COK → BLR', 'BLR → PNQ → DEL', 'BOM → JDH → DEL → HYD',
            'BLR → BOM → BHO → DEL', 'DEL → AMD → COK', 'BLR → LKO → DEL',
            'CCU → GAU → BLR', 'BOM → GOI → HYD', 'CCU → BOM → AMD → BLR',
            'CCU → BBI → IXR → DEL → BLR', 'DEL → DED → BOM → COK',
            'DEL → MAA → BOM → COK', 'BLR → AMD → DEL', 'BLR → VGA → DEL',
            'CCU → JAI → DEL → BLR', 'CCU → AMD → BLR', 'CCU → VNS → DEL → BLR',
            'BLR → BOM → IDR → DEL', 'BLR → BBI → DEL', 'BLR → GOI → DEL',
            'BOM → AMD → ISK → HYD', 'BOM → DED → DEL → HYD', 'DEL → IXC → BOM → COK',
            'CCU → PAT → BLR', 'BLR → CCU → BBI → DEL', 'CCU → BBI → HYD → BLR',
            'BLR → BOM → NAG → DEL', 'BLR → CCU → BBI → HYD → DEL', 'BLR → GAU → DEL',
            'BOM → BHO → DEL → HYD', 'BOM → JLR → HYD', 'BLR → HYD → VGA → DEL',
            'CCU → KNU → BLR', 'CCU → BOM → PNQ → BLR', 'DEL → BBI → COK',
            'BLR → VGA → HYD → DEL', 'BOM → JDH → JAI → DEL → HYD',
            'DEL → GWL → IDR → BOM → COK', 'CCU → RPR → HYD → BLR', 'CCU → VTZ → BLR',
            'CCU → DEL → VGA → BLR', 'BLR → BOM → IDR → GWL → DEL',
            'CCU → DEL → COK → TRV → BLR', 'BOM → COK → MAA → HYD', 'BOM → NDC → HYD',
            'BLR → BDQ → DEL', 'CCU → BOM → TRV → BLR', 'CCU → BOM → HBX → BLR',
            'BOM → BDQ → DEL → HYD', 'BOM → CCU → HYD', 'BLR → TRV → COK → DEL',
            'BLR → IDR → DEL', 'CCU → IXZ → MAA → BLR', 'CCU → GAU → IMF → DEL → BLR',
            'BOM → GOI → PNQ → HYD', 'BOM → BLR → CCU → BBI → HYD', 'BOM → MAA → HYD',
            'BLR → BOM → UDR → DEL', 'BOM → UDR → DEL → HYD', 'BLR → VGA → VTZ → DEL',
            'BLR → HBX → BOM → BHO → DEL', 'CCU → IXA → BLR', 'BOM → RPR → VTZ → HYD',
            'BLR → HBX → BOM → AMD → DEL', 'BOM → IDR → DEL → HYD' ,'BOM → BLR → HYD',
            'BLR → STV → DEL', 'CCU → IXB → DEL → BLR', 'BOM → JAI → DEL → HYD',
            'BOM → VNS → DEL → HYD', 'BLR → HBX → BOM → NAG → DEL',
            'BLR → BOM → IXC → DEL', 'BLR → CCU → BBI → HYD → VGA → DEL',
            'BOM → BBI → HYD'])
            Total_Stops = tuple(['non-stop', '2 stops', '1 stop', '3 stops', '4 stops'])
            Additional_Info = tuple(['No info', 'In-flight meal not included', 'No check-in baggage included',
            '1 Short layover', 'No Info', '1 Long layover', 'Change airports',
            'Business class', 'Red-eye flight', '2 Long layover'])

            # Numerical Pipeline
            num_pipeline = Pipeline(
                steps = [
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())                
                ]
            )

            # Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder',OrdinalEncoder(categories=[Airline,
                Source,
                Destination,
                Route,
                Total_Stops,
                Additional_Info])),
                ('scaler',StandardScaler())
                ]
            )

            logging.info(f'Categorical Columns : {categorical_columns}')
            logging.info(f'Numerical Columns   : {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")

    def initiate_data_transformation(self):
        try:
            train_data_path = 'artifacts/data_ingestion/unzipped_data/train_data.csv'
            test_data_path = 'artifacts/data_ingestion/unzipped_data/test_data.csv'

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            # Read training and test data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read train and test data completed')

            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_obj()

            target_column_name = 'Price'

            # Separate input features and target features
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            # Apply the preprocessing object on training and test input features
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target features
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            # Save preprocessing object
            preprocessing_obj_file = os.path.join("artifacts", 'data_transformation', 'preprocessing_obj.pkl')
            with open(preprocessing_obj_file, 'wb') as file:
                pickle.dump(preprocessing_obj, file)

            logging.info("Saved preprocessing object.")
            logging.info("Transformation of the data is completed")
            
            return (
                train_arr,
                test_arr,
                preprocessing_obj_file
            )
        except Exception as e:
            logging.error(f"Error in initiate_data_transformation: {str(e)}")
        

