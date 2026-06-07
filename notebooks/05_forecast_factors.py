import pandas as pd
import numpy as np

from prophet import Prophet

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =========================
# LOAD DATA
# =========================

df = pd.read_csv(
    "data/processed/dataset_indonesia.csv"
)

# =========================
# FACTORS
# =========================

factors = [
    "Inflation",
    "Unemployment",
    "Population_Growth",
    "Exports",
    "Imports",
    "FDI",
    "Exchange_Rate"
]

results = []

# =========================
# LOOP FACTORS
# =========================

for factor in factors:

    print(f"\nEvaluating {factor}...")

    # -------------------------
    # Train-Test Split
    # -------------------------

    train = df[df["Year"] <= 2019]
    test = df[df["Year"] > 2019]

    y_train = train[factor]
    y_test = test[factor]

    X_train = train[["Year"]]
    X_test = test[["Year"]]

    # =====================================================
    # MODEL 1 - LINEAR REGRESSION
    # =====================================================

    lr = LinearRegression()

    lr.fit(X_train, y_train)

    pred_lr = lr.predict(X_test)

    results.append({
        "Factor": factor,
        "Model": "Linear Regression",
        "MAE": mean_absolute_error(y_test, pred_lr),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_lr)),
        "R2": r2_score(y_test, pred_lr)
    })

    # =====================================================
    # MODEL 2 - PROPHET
    # =====================================================

    prophet_train = pd.DataFrame({
        "ds": pd.to_datetime(train["Year"], format="%Y"),
        "y": train[factor]
    })

    prophet_model = Prophet(
        yearly_seasonality=False,
        daily_seasonality=False,
        weekly_seasonality=False
    )

    prophet_model.fit(prophet_train)

    future = pd.DataFrame({
        "ds": pd.to_datetime(test["Year"], format="%Y")
    })

    forecast = prophet_model.predict(future)

    pred_prophet = forecast["yhat"].values

    results.append({
        "Factor": factor,
        "Model": "Prophet",
        "MAE": mean_absolute_error(y_test, pred_prophet),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_prophet)),
        "R2": r2_score(y_test, pred_prophet)
    })

    # =====================================================
    # MODEL 3 - EXPONENTIAL SMOOTHING
    # =====================================================

    es_model = ExponentialSmoothing(
        y_train,
        trend="add"
    )

    es_fit = es_model.fit()

    pred_es = es_fit.forecast(
        len(test)
    )

    results.append({
        "Factor": factor,
        "Model": "Exponential Smoothing",
        "MAE": mean_absolute_error(y_test, pred_es),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred_es)),
        "R2": r2_score(y_test, pred_es)
    })

# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "data/processed/forecast_model_comparison.csv",
    index=False
)

print("\n========================")
print("MODEL COMPARISON")
print("========================")

print(results_df)

print("\nSaved:")
print("data/processed/forecast_model_comparison.csv")