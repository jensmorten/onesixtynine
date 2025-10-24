# save this as app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.api import VAR
import matplotlib.ticker as mticker

# --- Streamlit config ---
st.set_page_config(
    page_title="OneSixtyNine: Prediksjon av norske politiske partiers oppslutning (stortingsvalg)",
    layout="wide"
)
st.title("OneSixtyNine: Prediksjon av norske politiske partiers oppslutning (stortingsvalg)")

# --- Load data ---
url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
df = df.sort_index()
df.index = df.index.to_period('M').to_timestamp('M')  # month-end

# --- Sidebar inputs ---
st.sidebar.markdown("### Sett parametere:", unsafe_allow_html=True)

lags = st.sidebar.number_input(
    "Antall måneder å bruke til tilpassing (trening) av modellen (maks 12):",
    min_value=1, max_value=12, value=6, step=1
)
n_months = st.sidebar.number_input(
    "Måneder framover å predikere",
    min_value=1, max_value=24, value=6, step=1
)
months_back = st.sidebar.number_input(
    "Måneder bakover i tid å vise i plottet:",
    min_value=6, max_value=36, value=12, step=1
)
months_back_start = st.sidebar.number_input(
    "Måneder bakover å starte prediksjon: ",
    min_value=0, max_value=months_back, value=0, step=1
)

if months_back_start>0:
	df=df[:-months_back_start]

# --- Fit VAR model ---
model = VAR(df)
model_fitted = model.fit(maxlags=lags, method="ols", trend="n", verbose=False)

# --- Forecast ---
forecast, forecast_lower, forecast_upper = model_fitted.forecast_interval(model_fitted.endog, steps=n_months)
forecast_index = pd.date_range(start=df.index[-1], periods=n_months+1, freq='M')[1:]

forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0) * 100 #normalisering til 100%
forecast_lower_df = pd.DataFrame(forecast_lower, index=forecast_index, columns=df.columns)
forecast_upper_df = pd.DataFrame(forecast_upper, index=forecast_index, columns=df.columns)

norske_mnd = {
    1: "jan", 2: "feb", 3: "mars", 4: "apr", 5: "mai", 6: "juni",
    7: "juli", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "des"
}

def norsk_dato_formatter(x, pos=None):
    dato = mdates.num2date(x)
    return f"{dato.day}. {norske_mnd[dato.month]} {dato.year}"

# --- Plot colors ---
colors = {
    'Ap': '#FF0000', 'Hoyre': '#0000FF', 'Frp': '#00008B', 'SV': '#FF6347',
    'SP': '#006400', 'KrF': '#FFD700', 'Venstre': '#ADD8E6',
    'MDG': '#008000', 'Rodt': '#8B0000', 'Andre': '#808080'
}

df_recent = df.iloc[-months_back:]

df_recent_eom = df_recent.copy()
df_recent_eom.index = df_recent_eom.index + MonthEnd(0)

forecast_df_eom = forecast_df.copy()
forecast_df_eom.index = forecast_df_eom.index + MonthEnd(0)
forecast_lower_df_eom = forecast_lower_df.copy()
forecast_lower_df_eom.index = forecast_lower_df_eom.index + MonthEnd(0)
forecast_upper_df_eom = forecast_upper_df.copy()
forecast_upper_df_eom.index = forecast_upper_df_eom.index + MonthEnd(0)

valg_dato = pd.Timestamp("2025-09-08")
valg_resultat = {
    'Ap': 28,
    'Hoyre': 14.6,
    'Frp': 23.8,
    'SV': 5.6,
    'SP': 5.6,
    'KrF': 4.2,
    'Venstre': 3.7,
    'MDG': 4.7,
    'Rodt': 5.3,
    'Andre': 4.5
}

# --- Plot setup ---
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(14, 7))

for party, color in colors.items():
    ax.plot(df_recent_eom.index, df_recent_eom[party],
            color=color, linewidth=2, label=party)
    ax.plot(forecast_df_eom.index, forecast_df_eom[party],
            color=color, linestyle="--", linewidth=1.5)
    ax.plot([df_recent_eom.index[-1], forecast_df_eom.index[0]],
            [df_recent_eom[party].iloc[-1], forecast_df_eom[party].iloc[0]],
            color=color, linestyle="--", linewidth=1.5)
    ax.fill_between(forecast_df_eom.index,
                    forecast_lower_df_eom[party],
                    forecast_upper_df_eom[party],
                    color=color, alpha=0.08)

for party, prosent in valg_resultat.items():
    ax.scatter(valg_dato, prosent, marker="x", color=colors[party], s=15, zorder=5)

ax.set_xlim(df_recent_eom.index[0], forecast_df_eom.index[-1])
ax.set_ylim(0, 40)
ax.set_ylabel("Oppslutning (%)", fontsize=12)



# 
ax.xaxis.set_major_formatter(mticker.FuncFormatter(norsk_dato_formatter))

siste_dato = df.index[-1]
siste_dato_norsk = f"{siste_dato.day}. {norske_mnd[siste_dato.month]} {siste_dato.year}"

ax.set_title(
    f"Prediksjon basert på {lags} måneders historikk, {n_months} måneder framover fra {siste_dato_norsk}",
    fontsize=12, pad=20
)

ax.legend(title="Partier", loc="upper left", ncol=2,facecolor="white",edgecolor="lightgray",
          frameon=True, fontsize=11, title_fontsize=12)
ax.grid(alpha=0.5, linestyle=":")
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(
    axis="x",
    which="major",
    length=6,       # lengde på ticks
    width=1.2,      # tykkelse
    color="black",
    labelsize=12
)

# Y-ticks kan også justeres tilsvarende om ønskelig
ax.tick_params(
    axis="y",
    which="major",
    length=6,
    width=1.2,
    color="black",
    labelsize=12
)

ax.text(pd.Timestamp("2025-09-08"), 30, "x Valgresultat 2025:",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(facecolor="white"))

plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# --- Disclaimer / info ---
st.sidebar.markdown("""
<hr>
<p>
Prediksjonsmodellen <b>OneSixtyNine</b>* baserer seg på historiske meningsmålinger hentet fra 
<a href="https://www.pollofpolls.no" target="_blank">www.pollofpolls.no</a>, 
men har ingen tilknytning til denne siden utover bruk av data som gjøres offentlig tilgjengelig.  
</p>
<p>
Modellen bruker vektor-autoregresjon 
(<a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html" target="_blank">VAR</a>)
for å samtidig tilpasse 10 korrelerte tidsserier. Du kan selv justere modellparametere for å se effekten.  
</p>
<p>
Kontakt gjerne <a href="mailto:jens.morten.nilsen@gmail.com">jens.morten.nilsen@gmail.com</a> for spørsmål eller kommentarer.  
</p>
<p>
*Navnet er en homage til Nate Silvers 
<a href="https://en.wikipedia.org/wiki/FiveThirtyEight" target="_blank">FiveThirtyEight</a>.
</p>
""", unsafe_allow_html=True)
