# save this as app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR

st.set_page_config(page_title="Partiers oppslutning (stortingsvalg, Norge)", layout="wide")
st.title("OneSixtyNine prediksjon av partiers oppslutning")

# --- Load data ---
url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
df = df.sort_index()
df.index = df.index.to_period('M').to_timestamp('M')  # month-end

# Sidebar inputs
lags = st.sidebar.number_input("måneder bakover å bruke til trening (maks 12)", min_value=1, max_value=12, value=6, step=1)
n_months = st.sidebar.number_input("Måneder framover å predikere", min_value=1, max_value=24, value=6, step=1)
months_back = st.sidebar.number_input("Måneder bakover å vise i plottet", min_value=1, max_value=36, value=12, step=1)

# --- Fit VAR model ---
model = VAR(df)
model_fitted = model.fit(maxlags=lags, method="ols", trend="n", verbose=False)

# --- Forecast ---
forecast, forecast_lower, forecast_upper = model_fitted.forecast_interval(model_fitted.endog, steps=n_months)
forecast_index = pd.date_range(start=df.index[-1], periods=n_months+1, freq='M')[1:]

forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
forecast_lower_df = pd.DataFrame(forecast_lower, index=forecast_index, columns=df.columns)
forecast_upper_df = pd.DataFrame(forecast_upper, index=forecast_index, columns=df.columns)

# --- Plot ---
colors = {
    'Ap': '#FF0000', 'Hoyre': '#0000FF', 'Frp': '#00008B', 'SV': '#FF6347',
    'SP': '#006400', 'KrF': '#FFD700', 'Venstre': '#ADD8E6',
    'MDG': '#008000', 'Rodt': '#8B0000', 'Andre': '#808080'
}

fig, ax = plt.subplots(figsize=(14, 7))
df_recent = df.iloc[-months_back:]

for party, color in colors.items():
    # Historical
    plt.plot(df_recent.index, df_recent[party], marker="o", color=color, label=f"{party}")
    # Forecast
    plt.plot(forecast_df.index, forecast_df[party], linestyle="dashed", color=color)
    # Connect last actual to first forecast
    plt.plot([df_recent.index[-1], forecast_df.index[0]],
             [df_recent[party].iloc[-1], forecast_df[party].iloc[0]],
             color=color, linestyle="dashed")
    # Confidence interval
    plt.fill_between(forecast_df.index, forecast_lower_df[party], forecast_upper_df[party], color=color, alpha=0.15)

plt.xlim(df_recent.index[0], forecast_df.index[-1])
plt.ylim(0, 40)
plt.xlabel("Måned")
plt.ylabel("Oppslutning (%)")
plt.title(f"Prediksjon av meningsmåling med bruka av ({lags} måneders historie,  + {n_months} måneder framover i tid)")
plt.legend(loc="upper left", ncol=2)
plt.grid(alpha=0.2)
plt.tight_layout()

# Show plot in Streamlit
plt.tight_layout()
st.pyplot(fig, use_container_width=False)
