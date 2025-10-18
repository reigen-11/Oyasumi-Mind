import os
import sys
import pandas as pd
from dataclasses import dataclass
from Data_Ingestion import DataIngestion
from utils.Data_utils import train_test_split
from utils.logger import logging
from utils.custom_exceptions import CustomException
from lightgbm import LGBMRegressor
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import json
from utils.visual_utils import plot_feature_importance, plot_predicted_vs_actual, plot_residuals

@dataclass
class ModelTrainerconfig:
    trained_model_path: str
    test_size: float
    random_state: int

class ModelTrainer:
    def __init__(self, config_path: str = "config.json"):
        from utils.Data_utils import load_json_config
        self.config = load_json_config(config_path)
        
        trainer_settings = self.config['model_training_settings']
        
        self.trainer_config = ModelTrainerconfig(
            trained_model_path=trainer_settings['trained_model_path'],
            test_size=trainer_settings['test_size'],
            random_state=trainer_settings['random_state']
        )
    
    def initiate_model_trainer(self, data_path: str):
        logging.info(f"{'+' * 150}")
        logging.info("2. Initiated Model Training")
        try:
            # Placeholder for model training logic
            logging.info(f"Training model with data from {data_path}")
            df = pd.read_csv(data_path)
            logging.info(f"Dataset shape for training: {df.shape}")

            X_train, X_test, y_train, y_test = train_test_split(
                df, target_column=self.config['column_settings']['target'],
                test_size=self.trainer_config.test_size,
                random_state=self.trainer_config.random_state
            )
            logging.info(f"Splitted data into train and test sets and stratification on target column")

            logging.info("Initializing model for training")
            # Dummy model training code
            model = LGBMRegressor()
            model.fit(X_train, y_train)
            logging.info("✅ Model training complete")
            with open(self.trainer_config.trained_model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logging.info("Evaluating model performance on test set")
            y_pred = model.predict(X_test)
            metrics = {
                "R2": r2_score(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "Explained_Variance": explained_variance_score(y_test, y_pred)
            }
            # Access the path from the main config object instead
            metrics_path = self.config['model_training_settings']['trained_model_metrics_path']
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            logging.info("✅ Model evaluation complete")

            logging.info("Generating visualizations for model performance")
            plot_feature_importance(model, feature_names=X_train.columns.tolist(),
                        save_path=self.config['model_training_settings']['feature_importance_plot_path'])
            plot_predicted_vs_actual(y_test, y_pred,
                                    save_path=self.config['model_training_settings']['predicted_vs_actual_plot_path'])
            plot_residuals(y_test, y_pred,
                        save_path=self.config['model_training_settings']['residuals_plot_path'])
            logging.info("✅ Model performance visualizations generated")
            logging.info(f"💾 Trained model saved to {self.trainer_config.trained_model_path}")
            logging.info(f"{'=' * 150}")

            return self.trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
    
# To Check if it works fine ! Dammm 
if __name__ == "__main__":
    obj = DataIngestion()
    cleaned_raw_data_path = obj.initiate_data_ingestion()
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(data_path=cleaned_raw_data_path)
