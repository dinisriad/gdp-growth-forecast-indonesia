import pandas as pd

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/dataset_indonesia.csv"
)

print("\n===== DATASET INFO =====")
print(df.info())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

print("\n===== SUMMARY STATISTICS =====")
print(df.describe())

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== YEAR RANGE =====")
print(
    f"{df['Year'].min()} - {df['Year'].max()}"
)

print("\n✅ Data Quality Check Selesai")