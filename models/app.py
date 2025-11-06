import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.api import VAR
import matplotlib.ticker as mticker

# --- Streamlit-oppsett ---
st.set_page_config(
    page_title="OneSixtyNine: Prediksjon av oppslutninga til norske parti (stortingsval)",
    layout="wide"
)
st.title("OneSixtyNine: Prediksjon av oppslutninga til norske parti (stortingsval)")

# --- Last inn data ---
url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
df = df.sort_index()
df.index = df.index.to_period('M').to_timestamp('M')  # månadsslutt

# --- Map kolonnenamn til nynorsk for å unngå KeyError ---
kolonne_map = {
    'Hoyre': 'Høgre',
    'Rodt': 'Raudt',
    'SP': 'Sp'
}
df = df.rename(columns=kolonne_map)

# --- Sidemeny ---
st.sidebar.markdown("### Set modellparametrar:", unsafe_allow_html=True)

lags = st.sidebar.number_input(
    "Talet på månader å bruke til tilpassing (trening) av modellen (maks 12):",
    min_value=1, max_value=12, value=6, step=1
)
n_months = st.sidebar.number_input(
    "Månader framover å predikere:",
    min_value=1, max_value=24, value=6, step=1
)
months_back = st.sidebar.number_input(
    "Månader bakover i tid å vise i plottet:",
    min_value=6, max_value=36, value=12, step=1
)
months_back_start = st.sidebar.number_input(
    "Månader bakover å starte prediksjon frå:",
    min_value=0, max_value=months_back, value=0, step=1
)

if months_back_start > 0:
    df = df[:-months_back_start]

# --- Tilpass VAR-modell ---
model = VAR(df)
model_fitted = model.fit(maxlags=lags, method="ols", trend="n", verbose=False)

# --- Prediksjon ---
forecast, forecast_lower, forecast_upper = model_fitted.forecast_interval(model_fitted.endog, steps=n_months)
forecast_index = pd.date_range(start=df.index[-1], periods=n_months+1, freq='M')[1:]

forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0) * 100  # normalisering til 100 %
forecast_lower_df = pd.DataFrame(forecast_lower, index=forecast_index, columns=df.columns)
forecast_upper_df = pd.DataFrame(forecast_upper, index=forecast_index, columns=df.columns)

# --- Formatering av datoar ---
norske_mnd = {
    1: "jan", 2: "feb", 3: "mars", 4: "apr", 5: "mai", 6: "juni",
    7: "juli", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "des"
}

def norsk_dato_formatter(x, pos=None):
    dato = mdates.num2date(x)
    return f"{dato.day}. {norske_mnd[dato.month]} {dato.year}"

# --- Fargar ---
colors = {
    'Ap': '#FF0000', 'Høgre': '#0000FF', 'Frp': '#00008B', 'SV': '#FF6347',
    'Sp': '#006400', 'KrF': '#FFD700', 'Venstre': '#ADD8E6',
    'MDG': '#008000', 'Raudt': '#8B0000', 'Andre': '#808080'
}

# --- Juster data for plotting ---
df_recent = df.iloc[-months_back:]

df_recent_eom = df_recent.copy()
df_recent_eom.index = df_recent_eom.index + MonthEnd(0)

forecast_df_eom = forecast_df.copy()
forecast_df_eom.index = forecast_df_eom.index + MonthEnd(0)
forecast_lower_df_eom = forecast_lower_df.copy()
forecast_lower_df_eom.index = forecast_lower_df_eom.index + MonthEnd(0)
forecast_upper_df_eom = forecast_upper_df.copy()
forecast_upper_df_eom.index = forecast_upper_df_eom.index + MonthEnd(0)

# --- legg til valresultat 2025 ---
val_dato = pd.Timestamp("2025-09-08")
val_resultat = {
    'Ap': 28,
    'Høgre': 14.6,
    'Frp': 23.8,
    'SV': 5.6,
    'Sp': 5.6,
    'KrF': 4.2,
    'Venstre': 3.7,
    'MDG': 4.7,
    'Raudt': 5.3,
    'Andre': 4.5
}

# --- Plotting ---
sns.set_theme(style="white", context="talk")
fig, ax = plt.subplots(figsize=(14, 7))

for parti, farge in colors.items():
    ax.plot(df_recent_eom.index, df_recent_eom[parti],
            color=farge, linewidth=2, label=parti)
    ax.plot(forecast_df_eom.index, forecast_df_eom[parti],
            color=farge, linestyle="--", linewidth=1.5)
    ax.plot([df_recent_eom.index[-1], forecast_df_eom.index[0]],
            [df_recent_eom[parti].iloc[-1], forecast_df_eom[parti].iloc[0]],
            color=farge, linestyle="--", linewidth=1.5)
    ax.fill_between(forecast_df_eom.index,
                    forecast_lower_df_eom[parti],
                    forecast_upper_df_eom[parti],
                    color=farge, alpha=0.08)

for parti, prosent in val_resultat.items():
    ax.scatter(val_dato, prosent, marker="x", color=colors[parti], s=15, zorder=5)

ax.set_xlim(df_recent_eom.index[0], forecast_df_eom.index[-1])
ax.set_ylim(0, 40)
ax.set_ylabel("Oppslutning (%)", fontsize=12)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(norsk_dato_formatter))

siste_dato = df.index[-1]
siste_dato_norsk = f"{siste_dato.day}. {norske_mnd[siste_dato.month]} {siste_dato.year}"

ax.set_title(
    f"Prediksjon basert på {lags} månaders historikk, {n_months} månader framover frå {siste_dato_norsk}",
    fontsize=12, pad=20
)

ax.legend(title="Parti", loc="upper left", ncol=2, facecolor="white", edgecolor="lightgray",
          frameon=True, fontsize=11, title_fontsize=12)
ax.grid(alpha=0.5, linestyle=":")
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(False)

ax.tick_params(
    axis="x",
    which="major",
    length=6,       # lengd på merker (ticks)
    width=1.2,      # tjukkleik
    color="black",
    labelsize=12
)

ax.tick_params(
    axis="y",
    which="major",
    length=6,
    width=1.2,
    color="black",
    labelsize=12
)

ax.text(pd.Timestamp("2025-09-08"), 30, "x Valresultat 2025:",
        fontsize=10, ha="center", va="bottom",
        bbox=dict(facecolor="white"))

plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# --- Fråsegn / info ---
st.sidebar.markdown("""
<hr>
<p>
Prediksjonsmodellen <b>OneSixtyNine</b>* byggjer på historiske meiningsmålingar henta frå 
<a href="https://www.pollofpolls.no" target="_blank">www.pollofpolls.no</a>, 
men har inga tilknyting til denne sida utover bruk av data som vert gjort offentleg tilgjengeleg.  
</p>
<p>
Modellen brukar vektor-autoregresjon 
(<a href="https://www.statsmodels.org/stable/generated/statsmodels.tsa.vector_ar.var_model.VAR.html" target="_blank">VAR</a>)
for å tilpasse ti korrelerte tidsseriar samtidig. Du kan sjølv justere modellparametrane for å sjå effekten.  
</p>
<p>
Ta gjerne kontakt med <a href="mailto:jens.morten.nilsen@gmail.com">jens.morten.nilsen@gmail.com</a> for spørsmål eller kommentarar.  
</p>
<p>
*Namnet er ei hyllest til Nate Silvers 
<a href="https://en.wikipedia.org/wiki/FiveThirtyEight" target="_blank">FiveThirtyEight</a>.
</p>
""", unsafe_allow_html=True)
