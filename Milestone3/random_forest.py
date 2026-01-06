import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv("outputs/dataset/visa_ml_ready_dataset.csv"
                 , low_memory=False)

X = df[
    [
        "citizenship_country",
        "visa_type",
        "employer_state",
        "month",
        "quarter",
        "season",
        "avg_wage",
        "country_avg_processing",
        "visa_type_avg_processing",
        "state_avg_processing"
    ]
]

y = df["processing_time_days"]

cat_cols = ["citizenship_country", "visa_type", "employer_state", "season"]
num_cols = [c for c in X.columns if c not in cat_cols]

preprocessor = ColumnTransformer(
    [
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42))
    ]
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Random Forest Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))


# Not Good for this model on this data
# Observation:
# Random Forest shows high computational cost on this large, high-dimensional dataset due to extensive one-hot encoding.
# The improvement in MAE/RMSE is limited compared to Gradient Boosting, making it less suitable for this problem.


