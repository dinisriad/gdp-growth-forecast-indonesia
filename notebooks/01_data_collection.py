import requests
import pandas as pd

url = "https://api.worldbank.org/v2/country/IDN/indicator/NY.GDP.MKTP.KD.ZG?format=json&per_page=100"

response = requests.get(url)

data = response.json()[1]

df = pd.DataFrame([
    {
        "Year": item["date"],
        "GDP_Growth": item["value"]
    }
    for item in data
])

df = df.dropna()

print(df.head())

df.to_csv(
    "data/raw/gdp_growth.csv",
    index=False
)

print("GDP data saved successfully!")