import pandas as pd

# =========================
# LOAD DATA
# =========================

gdp = pd.read_csv("data/raw/gdp_growth.csv")
inflation = pd.read_csv("data/raw/inflation.csv")
unemployment = pd.read_csv("data/raw/unemployment.csv")
population = pd.read_csv("data/raw/population_growth.csv")
exports = pd.read_csv("data/raw/exports.csv")
imports = pd.read_csv("data/raw/imports.csv")
fdi = pd.read_csv("data/raw/fdi.csv")
exchange = pd.read_csv("data/raw/exchange_rate.csv")

# =========================
# RENAME COLUMNS
# =========================

gdp.columns = ["Year", "GDP_Growth"]
inflation.columns = ["Year", "Inflation"]
unemployment.columns = ["Year", "Unemployment"]
population.columns = ["Year", "Population_Growth"]
exports.columns = ["Year", "Exports"]
imports.columns = ["Year", "Imports"]
fdi.columns = ["Year", "FDI"]
exchange.columns = ["Year", "Exchange_Rate"]

# =========================
# MERGE DATASET
# =========================

df = gdp

datasets = [
    inflation,
    unemployment,
    population,
    exports,
    imports,
    fdi,
    exchange
]

for dataset in datasets:
    df = pd.merge(df, dataset, on="Year")

# =========================
# CLEAN DATA
# =========================

df["Year"] = df["Year"].astype(int)

df = df.sort_values("Year")

df = df.reset_index(drop=True)

# =========================
# SAVE DATASET
# =========================

df.to_csv(
    "data/processed/dataset_indonesia.csv",
    index=False
)

# =========================
# CHECK RESULT
# =========================

print("\n===== HEAD =====")
print(df.head())

print("\n===== TAIL =====")
print(df.tail())

print("\n===== SHAPE =====")
print(df.shape)

print("\n===== YEAR RANGE =====")
print("Start Year :", df["Year"].min())
print("End Year   :", df["Year"].max())

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n✅ Dataset berhasil dibuat!")
print("📁 File disimpan di:")
print("data/processed/dataset_indonesia.csv")