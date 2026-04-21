import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("data/processed/cleaned_sales.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.month
df['Year'] = df['Order Date'].dt.year

X = df[['Month', 'Year']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

joblib.dump(model, "models/sales_forecast_xgb.pkl")
print("✅ Model saved successfully!")
