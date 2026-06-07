import requests
import pandas as pd

INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
    "population_growth": "SP.POP.GROW",
    "exports": "NE.EXP.GNFS.ZS",
    "imports": "NE.IMP.GNFS.ZS",
    "fdi": "BX.KLT.DINV.WD.GD.ZS",
    "exchange_rate": "PA.NUS.FCRF"
}

for file_name, indicator_code in INDICATORS.items():

    url = (
        f"https://api.worldbank.org/v2/country/IDN/"
        f"indicator/{indicator_code}"
        f"?format=json&per_page=100"
    )

    response = requests.get(url)

    data = response.json()[1]

    df = pd.DataFrame([
        {
            "Year": item["date"],
            file_name: item["value"]
        }
        for item in data
    ])

    df = df.dropna()

    df.to_csv(
        f"data/raw/{file_name}.csv",
        index=False
    )

    print(f"✅ {file_name}.csv saved")