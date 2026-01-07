import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df: pd.DataFrame, target_cols: list[str]):
  X = df.drop(columns=target_cols)

  numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
  categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

  numeric_pipeline = SimpleImputer(strategy='median')
  categorical_pipline = OneHotEncoder(handle_unknown='ignore')

  preprocessor = ColumnTransformer(
      transformers=[
          ('num', StandardScaler(), num_cols),
          ('cat', categorical_pipline, cat_cols)  
      ]
  )

  joblib.dump(preprocessor, 'models/preprocessor.joblib')

  return preprocessor, X;