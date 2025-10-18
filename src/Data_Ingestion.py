import os
import sys
import pandas as pd
from dataclasses import dataclass
from utils.logger import logging
from utils.custom_exceptions import CustomException
from utils.Data_utils import (
    drop_nan_rows,
    remove_duplicates,
    detect_outliers_iqr_dataset,
    convert_column_types,
    clean_invalid_values,
    drop_columns,
    encode_categorical_columns
)
from utils.Data_utils import load_json_config

# print(sys.path)

@dataclass
class DataIngestionConfig:
    raw_data_path: str
    cleaned_data_path: str
    transformed_data_path: str
    remove_duplicates_flag: bool
    detect_outliers_flag: bool
    outlier_multiplier: float
    numeric_columns: list[str]
    categorical_columns: list[str]
    drop_columns_list: list[str]
    label_encoder_path: str 
    outliers: list[str]

class DataIngestion:
    def __init__(self, config_path: str = "config.json"):
        self.config = load_json_config(config_path)
        
        ingestion_settings = self.config['data_ingestion_settings']
        column_settings = self.config['column_settings']
        
        self.ingestion_config = DataIngestionConfig(
            raw_data_path=ingestion_settings['raw_data_path'],
            cleaned_data_path=ingestion_settings['cleaned_data_path'],
            transformed_data_path=ingestion_settings['transformed_data_path'],
            remove_duplicates_flag=ingestion_settings['remove_duplicates'],
            detect_outliers_flag=ingestion_settings['detect_outliers'],
            outlier_multiplier=ingestion_settings['outlier_multiplier'],
            numeric_columns=column_settings['numeric'],
            categorical_columns=column_settings['categorical'],
            drop_columns_list=column_settings['drop'],
            label_encoder_path=ingestion_settings['label_encoder_path'],
            outliers=column_settings['outliers']
        )

    def initiate_data_ingestion(self):
        logging.info(f"{'+' * 150}")
        logging.info("1. Initiated Data Ingestion")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logging.info('Reading the dataset as dataframe')

            if self.ingestion_config.remove_duplicates_flag:
                df = remove_duplicates(df)
                logging.info("✅ Duplicates removed successfully")

            if self.ingestion_config.detect_outliers_flag:
                df = detect_outliers_iqr_dataset(df, multiplier=self.ingestion_config.outlier_multiplier, cols=self.ingestion_config.outliers, return_bounds=True)
                logging.info("✅ Outlier detection and cleaning complete")
            
            df = convert_column_types(
                df,
                numeric_cols=self.ingestion_config.numeric_columns,
                categorical_cols=self.ingestion_config.categorical_columns,
                integer_type="Int64"
            )
            logging.info("✅ Column type conversions complete")

            col_rules = self.config['cleaning_rules']
            df = clean_invalid_values(df, col_rules)
            logging.info("✅ Invalid value cleaning complete")

            df = drop_columns(df, cols_to_remove=self.ingestion_config.drop_columns_list)
            logging.info(f"✅ Dropped unnecessary columns: {self.ingestion_config.drop_columns_list}")

            df.to_csv(self.ingestion_config.cleaned_data_path, index=False, header=True)
            logging.info(f"💾 Cleaned data saved to {self.ingestion_config.cleaned_data_path}")
            logging.info(f"✅ Final shape: {df.shape}\n🧩 Columns: {', '.join(df.columns)}")
            logging.info(f"{'-' * 150}")

            logging.info("Initializing Data Encoding")
            df, encoder = encode_categorical_columns(df, self.ingestion_config.categorical_columns)
            with open(self.ingestion_config.label_encoder_path, 'wb') as f:
                import pickle
                pickle.dump(encoder, f)
            logging.info("✅ Data Encoding complete")
            logging.info("Dropping any NANs after encoding")
            df = drop_nan_rows(df)
            df.to_csv(self.ingestion_config.transformed_data_path, index=False, header=True)
            logging.info(f"{'=' * 150}")

            return self.ingestion_config.transformed_data_path

        except Exception as e:
            raise CustomException(e, sys)

# To Check if it works fine ! Dammm 
# if __name__ == "__main__":
#     obj = DataIngestion()
#     cleaned_raw_data_path = obj.initiate_data_ingestion()