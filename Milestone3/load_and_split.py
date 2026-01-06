import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    "outputs/dataset/visa_ml_ready_dataset.csv",
    low_memory=False
)

y = df["processing_time_days"]

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("Step 1 completed: Data loaded and split.")
