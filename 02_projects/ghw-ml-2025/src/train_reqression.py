import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_standard_model, load_biased_model
from preprocess import preprocess_data

df = load_standard_model()

df['total_score'] = (df['Midterm_Score'] + df['Final_Score'] + df['Projects_Score']) / 3

y = df['total_score']

preprocessor, X = preprocess_data(df, target_cols=['total_score'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test) 
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
joblib.dump(model, 'models/regression_model.joblib')