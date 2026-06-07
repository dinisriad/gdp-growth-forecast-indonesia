import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/dataset_indonesia.csv"
)

# =========================
# FEATURES & TARGET
# =========================

X = df[
    [
        "Inflation",
        "Unemployment",
        "Population_Growth",
        "Exports",
        "Imports",
        "FDI",
        "Exchange_Rate"
    ]
]

y = df["GDP_Growth"]

# =========================
# TRAIN TEST SPLIT
# =========================

train = df[df["Year"] <= 2019]
test = df[df["Year"] > 2019]

X_train = train[
    [
        "Inflation",
        "Unemployment",
        "Population_Growth",
        "Exports",
        "Imports",
        "FDI",
        "Exchange_Rate"
    ]
]

X_test = test[
    [
        "Inflation",
        "Unemployment",
        "Population_Growth",
        "Exports",
        "Imports",
        "FDI",
        "Exchange_Rate"
    ]
]

y_train = train["GDP_Growth"]
y_test = test["GDP_Growth"]

# =========================
# MODELS
# =========================

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

results = []

# =========================
# EVALUATION
# =========================

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(
            y_test,
            pred
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                y_test,
                pred
            )
        ),
        "R2": r2_score(
            y_test,
            pred
        )
    })

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="RMSE"
)

results_df.to_csv(
    "data/processed/gdp_model_comparison.csv",
    index=False
)

print("\n====================")
print("GDP MODEL RESULTS")
print("====================\n")

print(results_df)

print("\nSaved:")
print(
    "data/processed/gdp_model_comparison.csv"
)