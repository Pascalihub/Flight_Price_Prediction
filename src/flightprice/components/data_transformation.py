import os
from src.flightprice.logger import logging
import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pickle



class DataTransformation:
    def __init__(self, config):
        self.data_transformation_config = config

    def get_data_transformer_object(self):
        try:
            data = pd.read_csv(self.data_transformation_config.data_train_path)

            numerical_columns = ["Dep_Time", "Arrival_Time", "Duration"]
            categorical_columns = [
                'Airline', 'Source',
                'Destination', 'Route',
                'Total_Stops', 'Additional_Info'
            ]

            # Exclude non-numeric columns from the numerical columns list
            numerical_columns = [col for col in numerical_columns if col in data.columns and data[col].dtype != object]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numeric", num_pipeline, numerical_columns),
                    ("categorical", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            print(f"Error in get_data_transformer_object: {str(e)}")

    def initiate_data_transformation(self):
        try:
            data = pd.read_csv(self.data_transformation_config.data_train_path)

            preprocessor = self.get_data_transformer_object()

            transformed_data = preprocessor.fit_transform(data)

            output_dir = os.path.join(self.data_transformation_config.root_dir, "artifacts/data_transformation")
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, "preprocessors.pkl")
            with open(output_file, "wb") as file:
                pickle.dump(preprocessor, file)

            print("Data transformation completed and saved as a pickle file.")

        except Exception as e:
            print(f"Error in initiate_data_transformation: {str(e)}")

