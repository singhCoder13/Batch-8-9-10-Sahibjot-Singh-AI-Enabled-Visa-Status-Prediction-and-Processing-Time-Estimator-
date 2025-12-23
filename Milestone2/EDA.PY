import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("Milestone1/Dataset/processed_dataset.csv")
OUTPUT_DIR = Path("outputs/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH, low_memory=False)

print(df.info())
print(df.describe())

# 1️⃣ Processing Time Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["processing_time_days"], bins=50, kde=True)
plt.title("Distribution of Visa Processing Time (Days)")
plt.xlabel("Days")
plt.ylabel("Count")
plt.savefig(OUTPUT_DIR / "processing_time_distribution.png")
plt.close()

# 2️⃣ Outlier Detection
plt.figure(figsize=(8,4))
sns.boxplot(x=df["processing_time_days"])
plt.title("Outliers in Processing Time")
plt.savefig(OUTPUT_DIR / "processing_time_outliers.png")
plt.close()

# 3️⃣ Visa Type vs Processing Time
plt.figure(figsize=(10,5))
sns.boxplot(
    data=df,
    x="visa_type",
    y="processing_time_days"
)
plt.xticks(rotation=45)
plt.title("Processing Time by Visa Type")
plt.savefig(OUTPUT_DIR / "visa_type_processing_time.png")
plt.close()

# 4️⃣ Country-wise Avg Processing Time
country_avg = (
    df.groupby("citizenship_country")["processing_time_days"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
country_avg.plot(kind="bar")
plt.title("Top 10 Citizenship Countries by Avg Processing Time")
plt.ylabel("Days")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "country_avg_processing.png")
plt.close()

# 5️⃣ Seasonal Trend
plt.figure(figsize=(8,5))
season_avg = df.groupby("season")["processing_time_days"].mean()
season_avg.plot(kind="bar")
plt.title("Processing Time by Season")
plt.ylabel("Days")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "seasonal_processing.png")
plt.close()

print("EDA completed successfully. Plots saved to outputs/plots/")
