import pandas as pd
import numpy as np

#  dataset
df = pd.read_csv("Milestone1/Dataset/orignaldataset.csv", low_memory=False)

print("Initial rows:", len(df))

# Remove useless duplicate column if exists
if "citizenship_country.1" in df.columns:
    df = df.drop(columns=["citizenship_country.1"])

# INITIAL 
print("\n CLEAN DATA (Top 5 rows BEFORE processing) \n")
print(df.head())

print("\n CHECKING MISSING VALUES BEFORE PROCESSING\n")
print(df.isnull().sum())

# Convert date columns
df["received_date"] = pd.to_datetime(df["received_date"], errors="coerce")
df["decision_date"] = pd.to_datetime(df["decision_date"], errors="coerce")

# Remove rows where dates are missing
df = df.dropna(subset=["received_date", "decision_date"])
print("\nAfter removing rows with invalid dates:", len(df))

# Create processing time column
df["processing_time_days"] = (df["decision_date"] - df["received_date"]).dt.days

# Remove negative processing times
df = df[df["processing_time_days"] >= 0]
print("After removing negative processing times:", len(df))

print("\n PROCESSING TIME PREVIEW \n")
print(df[["received_date", "decision_date", "processing_time_days"]].head())

# Fix numeric columns
numeric_cols = ["wage_from", "wage_to"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Fix categorical columns
categorical_cols = [
    "case_status",
    "application_type",
    "citizenship_country",
    "employer_state",
    "job_title",
    "wage_level",
    "naics_code",
    "wage_unit",
    "visa_type",
    "visa_to_country"
]

for col in categorical_cols:
    if col in df.columns:
        if df[col].mode().size > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

print("\n MISSING VALUES AFTER FIXING \n")
print(df.isnull().sum())

# Create date-based features
df["month"] = df["received_date"].dt.month
df["quarter"] = df["received_date"].dt.quarter

def season_from_month(m):
    if m in [12, 1, 2]:
        return "Winter"
    elif m in [3, 4, 5]:
        return "Spring"
    elif m in [6, 7, 8]:
        return "Summer"
    return "Fall"

df["season"] = df["month"].apply(season_from_month)

print("\n NEW DATE FEATURES ADDED \n")
print(df[["month", "quarter", "season"]].head())


print("\n CLEAN DATA (Top 5 rows AFTER processing) \n")
print(df.head())

# SAMPLE ENCODING 
encoding_preview = pd.get_dummies(df[["citizenship_country", "application_type", "season"]]).head()


# Save final processed dataset
output_path = "Milestone1/Dataset/processed_dataset.csv"
df.to_csv(output_path, index=False)

print("\n PREPROCESSING COMPLETED")
print("Final rows:", len(df))
print("Processed dataset saved at:", output_path)
