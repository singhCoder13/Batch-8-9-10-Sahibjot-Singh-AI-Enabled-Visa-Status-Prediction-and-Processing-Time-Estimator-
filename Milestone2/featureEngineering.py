import pandas as pd
from pathlib import Path

# =============================
# PATHS
# =============================
DATA_PATH = Path("Milestone1/Dataset/processed_dataset.csv")
OUTPUT_PATH = Path("outputs/dataset")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH, low_memory=False)

# =============================
# DATE CONVERSION
# =============================
df["received_date"] = pd.to_datetime(df["received_date"])
df["decision_date"] = pd.to_datetime(df["decision_date"])

# =============================
# TIME-BASED FEATURES
# =============================
df["received_month"] = df["received_date"].dt.month
df["received_week"] = df["received_date"].dt.isocalendar().week
df["received_year"] = df["received_date"].dt.year

# =============================
# AGGREGATED FEATURES (KEY PART)
# =============================

# Country-wise historical average
df["country_avg_processing"] = (
    df.groupby("citizenship_country")["processing_time_days"]
      .transform("mean")
)

# Visa-type historical average
df["visa_type_avg_processing"] = (
    df.groupby("visa_type")["processing_time_days"]
      .transform("mean")
)

# Employer-state historical average (proxy for processing office)
df["state_avg_processing"] = (
    df.groupby("employer_state")["processing_time_days"]
      .transform("mean")
)

# =============================
# WAGE FEATURE (USEFUL SIGNAL)
# =============================
df["avg_wage"] = (df["wage_from"] + df["wage_to"]) / 2

# =============================
# FINAL CLEANING
# =============================
df = df.dropna()

# =============================
# SAVE ML-READY DATASET
# =============================
df.to_csv(OUTPUT_PATH / "visa_ml_ready_dataset.csv", index=False)

print("✅ Feature engineering completed. ML-ready dataset saved.")
