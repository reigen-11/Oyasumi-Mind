import ast
from flask import json
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from utils.custom_exceptions import CustomException
from utils.logger import logging
from sklearn.preprocessing import LabelEncoder


def remove_duplicates(df: pd.DataFrame, subset_cols: Optional[List[str]] = None) -> pd.DataFrame:
    try:
        if subset_cols:
            missing_cols = [c for c in subset_cols if c not in df.columns]
            if missing_cols:
                raise CustomException(f"Columns not found in DataFrame: {missing_cols}")
        df_cleaned = df.drop_duplicates(subset=subset_cols)
        logging.info(f"Removed duplicates based on columns: {subset_cols}")
        return df_cleaned
    except Exception as e:
        logging.error(f"Error in remove_duplicates: {e}")
        raise CustomException(f"Error in remove_duplicates: {e}")



def detect_outliers_iqr_dataset(
    df: pd.DataFrame, 
    multiplier: float = 1.5, 
    return_bounds: bool = False,
    cols: list = None
) -> Union[pd.DataFrame, Dict[str, Tuple[pd.DataFrame, float, float]]]:
    try:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        if cols is not None:
            selected_cols = [col for col in cols if col in numeric_cols]
            if not selected_cols:
                logging.warning("No valid numeric columns found in 'cols' list.")
                return df
        else:
            selected_cols = numeric_cols

        if not selected_cols:
            logging.warning("No numeric columns found in the dataset.")
            return df

        logging.info(f"Removing outliers for columns: {selected_cols}")

        all_outlier_indices = set()
        bounds_info = {}

        for column in selected_cols:
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            outlier_indices = outliers.index.tolist()
            all_outlier_indices.update(outlier_indices)

            count = len(outlier_indices)
            total = df.shape[0]
            pct = (count / total) * 100 if total > 0 else 0

            logging.info(
                f"Column '{column}': {count} outliers ({pct:.2f}%) removed using bounds [{lower_bound}, {upper_bound}]."
            )

            if return_bounds:
                bounds_info[column] = (lower_bound, upper_bound)

        df_cleaned = df.drop(index=all_outlier_indices)
        logging.info(f"Total rows removed: {len(all_outlier_indices)}. Remaining rows: {df_cleaned.shape[0]}")

        if return_bounds:
            return {"data": df_cleaned, "bounds": bounds_info}
        return df_cleaned

    except Exception as e:
        logging.error(f"Error in detect_outliers_iqr_dataset: {e}")
        raise CustomException(f"Error in detect_outliers_iqr_dataset: {e}")



def convert_column_types(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None, integer_type: str = "Int64") -> pd.DataFrame:
    try:
        numeric_cols = numeric_cols or []
        categorical_cols = categorical_cols or []
        for col in numeric_cols:
            if col in df.columns:
                # First coerce any non-numeric values to NaN
                coerced = pd.to_numeric(df[col], errors="coerce")
                try:
                    # Attempt to cast to the requested integer/nullable integer type.
                    df[col] = coerced.astype(integer_type)
                    logging.info(f"Converted column '{col}' to numeric type {integer_type}.")
                except Exception as cast_err:
                    # If casting to integer fails (mixed types/non-equivalent objects), keep the coerced numeric (float) series
                    df[col] = coerced
                    logging.warning(
                        f"Could not cast column '{col}' to {integer_type}: {cast_err}. "
                        "Kept as numeric (float) with non-numeric values coerced to NaN."
                    )
            else:
                logging.warning(f"Numeric column '{col}' not found in DataFrame.")
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")
                logging.info(f"Converted column '{col}' to categorical type.")
            else:
                logging.warning(f"Categorical column '{col}' not found in DataFrame.")
        return df
    except Exception as e:
        logging.error(f"Error in convert_column_types: {e}")
        raise CustomException(f"Error in convert_column_types: {e}")


def clean_invalid_values(df: pd.DataFrame, col_rules: Dict[str, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    try:
        for col, bounds in col_rules.items():
            if col not in df.columns:
                logging.warning(f"Column '{col}' not found in DataFrame. Skipping.")
                continue
            lower, upper = bounds
            if lower is not None and upper is not None:
                df.loc[~df[col].between(lower, upper), col] = pd.NA
            elif lower is not None:
                df.loc[df[col] < lower, col] = pd.NA
            elif upper is not None:
                df.loc[df[col] > upper, col] = pd.NA
            logging.info(f"Applied cleaning rules to column '{col}' with bounds {bounds}.")
        return df
    except Exception as e:
        logging.error(f"Error in clean_invalid_values: {e}")
        raise CustomException(f"Error in clean_invalid_values: {e}")



def drop_columns(df: pd.DataFrame, cols_to_remove: List[str]) -> pd.DataFrame:
    try:
        existing_cols = [col for col in cols_to_remove if col in df.columns]
        if not existing_cols:
            logging.warning("No columns from the list exist in the DataFrame. Nothing to drop.")
            return df

        df = df.drop(columns=existing_cols, errors='ignore')
        logging.info(f"Dropped columns: {existing_cols}")
        return df
    except Exception as e:
        logging.error(f"Error in drop_columns: {e}")
        raise CustomException(f"Error in drop_columns: {e}")



def load_json_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        raise CustomException(f"Configuration file not found at: {config_path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the config file: {config_path}")
        raise CustomException(f"Error decoding JSON from the config file: {config_path}")



def encode_categorical_columns(df, categorical_cols):
    try:
        encoders = {}
        for col in categorical_cols:
            if col not in df.columns:
                logging.warning(f"⚠️ Column '{col}' not found in DataFrame. Skipping encoding.")
                continue

            le = LabelEncoder()
            # Convert values to string to handle NaNs or categories consistently
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logging.info(f"✅ Encoded column '{col}' with {len(le.classes_)} unique classes.")

        return df, le

    except Exception as e:
        logging.error(f"Error while encoding categorical columns: {e}")
        raise CustomException(f"Error while encoding categorical columns: {e}")
    

def drop_nan_rows(df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
    try:
        before = len(df)
        df_cleaned = df.dropna(subset=subset)
        after = len(df_cleaned)

        if subset:
            logging.info(
                f"🧹 Dropped {before - after} rows containing NaN values in columns: {subset}"
            )
        else:
            logging.info(f"🧹 Dropped {before - after} rows containing any NaN values.")

        return df_cleaned

    except Exception as e:
        logging.error(f"Error while dropping NaN rows: {e}")
        raise CustomException(f"Error while dropping NaN rows: {e}")
    

def train_test_split(
    df: pd.DataFrame, 
    target_column: str, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    try:
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        logging.info(
            f"Performed train-test split with test size {test_size}. "
            f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"
        )

        return X_train, X_test, y_train, y_test

    except Exception as e:
        logging.error(f"Error in train_test_split: {e}")
        raise CustomException(f"Error in train_test_split: {e}")