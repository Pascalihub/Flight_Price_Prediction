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
from src.flightprice.entity import DataTransformationConfig
import pickle



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['Journey_day', 'Journey_month', 
                                 'Dep_hour', 'Dep_min', 
                                 'Arrival_hour', 'Arrival_min', 
                                 'Duration_hours', 'Duration_mins']
            categorical_columns = ['Airline', 'Source', 'Destination', 'Total_Stops']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ],
                remainder="drop"  # Ignore any columns not explicitly specified
            )

            return preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")

    def initiate_data_transformation(self):
        try:
            train_data_path = 'artifacts/data_ingestion/unzipped_data/train_data.csv'  # Replace with the actual path to your train data file
            test_data_path = 'artifacts/data_ingestion/unzipped_data/test_data.csv'  # Replace with the actual path to your test data file

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Price"

            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

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


