import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Loading dataset...")

df = pd.read_csv(
    "outputs/dataset/visa_ml_ready_dataset.csv",
    low_memory=False
)

# HARD CAP DATA SIZE (CRITICAL)
df = df.sample(n=20000, random_state=42)
print("Dataset sampled:", df.shape)

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
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=1  # IMPORTANT for Windows
            )
        )
    ]
)

# SMALL GRID (ENOUGH FOR MODULE 3)
param_grid = {
    "model__n_estimators": [50],
    "model__max_depth": [10, 20]
}

print("Starting GridSearch...")

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=1,   # IMPORTANT
    verbose=2   # SHOW PROGRESS
)

grid.fit(X_train, y_train)

print("GridSearch completed.")

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\nTuned Random Forest Results")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

joblib.dump(best_model, "saved_models/best_model.pkl")
print("Best model saved.")
