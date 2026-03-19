import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, PowerTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('notebooks/zomato_cleaned.csv')

columns_to_drop = ['rider_id', 'restaurant_latitude', 'restaurant_longitude', 'delivery_latitude', 'delivery_longitude', 'order_date', 'order_time_hour', 'order_day', 'city_name', 'order_day_of_week', 'order_month']
df.drop(columns=columns_to_drop, inplace=True)
df = df.dropna()

X = df.drop(columns='time_taken')
y = df['time_taken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Building preprocessor...")
num_cols = ["age","ratings","pickup_time_minutes","distance"]
nominal_cat_cols = ['weather','type_of_order','type_of_vehicle',"festival","city_type","is_weekend","order_time_of_day"]
ordinal_cat_cols = ["traffic","distance_type"]
traffic_order = ["low","medium","high","jam"]
distance_type_order = ["short","medium","long","very_long"]

preprocessor = ColumnTransformer(transformers=[
    ("scale", MinMaxScaler(), num_cols),
    ("nominal_encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), nominal_cat_cols),
    ("ordinal_encode", OrdinalEncoder(categories=[traffic_order, distance_type_order], encoded_missing_value=-999, handle_unknown="use_encoded_value", unknown_value=-1), ordinal_cat_cols)
], remainder="passthrough", n_jobs=-1, force_int_remainder_cols=False, verbose_feature_names_out=False)

print("Fitting preprocessor...")
X_train_trans = preprocessor.fit_transform(X_train)

print("Building Stacking Regressor...")
best_rf = RandomForestRegressor(n_estimators=479, criterion='squared_error', max_depth=17, max_features=None, min_samples_split=9, min_samples_leaf=2, max_samples=0.6603673526197067, n_jobs=-1)
best_lgbm = LGBMRegressor(n_estimators=154, max_depth=27, learning_rate=0.22234435854395157, subsample=0.7592213724048168, min_child_weight=20, min_split_gain=0.004604680609280751, reg_lambda=97.81002379097947, n_jobs=-1, verbose=-1)
lr = LinearRegression()

stacking_reg = StackingRegressor(estimators=[("rf", best_rf), ("lgbm", best_lgbm)], final_estimator=lr, cv=5, n_jobs=-1)

print("Training TransformedTargetRegressor wrapper...")
pt = PowerTransformer()
model = TransformedTargetRegressor(regressor=stacking_reg, transformer=pt)
model.fit(X_train_trans, y_train)

print(f"Metrics (Train R2): {model.score(X_train_trans, y_train)}")

print("Saving correctly fitted models...")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
joblib.dump(model.transformer_, MODELS_DIR / "power_transformer.joblib")
# CRITICAL: Save the inner fitted regressor!
joblib.dump(model.regressor_, MODELS_DIR / "stacking_regressor.joblib")

print("Done! Artifacts dumped successfully.")
