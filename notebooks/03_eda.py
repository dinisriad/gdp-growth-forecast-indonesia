import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/dataset_indonesia.csv"
)

# =========================
# BASIC INFO
# =========================

print("\n===== INFO =====")
print(df.info())

print("\n===== DESCRIBE =====")
print(df.describe())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# =========================
# GDP TREND
# =========================

plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["GDP_Growth"])

plt.title("Indonesia GDP Growth (1991-2024)")
plt.xlabel("Year")
plt.ylabel("GDP Growth (%)")

plt.tight_layout()

plt.savefig(
    "reports/figures/gdp_growth_trend.png"
)

plt.close()

# =========================
# INFLATION TREND
# =========================

plt.figure(figsize=(10, 5))
plt.plot(df["Year"], df["Inflation"])

plt.title("Indonesia Inflation (1991-2024)")
plt.xlabel("Year")
plt.ylabel("Inflation (%)")

plt.tight_layout()

plt.savefig(
    "reports/figures/inflation_trend.png"
)

plt.close()

# =========================
# CORRELATION HEATMAP
# =========================

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.drop(columns=["Year"]).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png"
)

plt.close()

print("\n✅ EDA selesai")
print("📁 Grafik disimpan di reports/figures/")