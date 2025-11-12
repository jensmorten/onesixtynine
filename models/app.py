import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pandas.tseries.offsets import MonthEnd
from statsmodels.tsa.api import VAR
import matplotlib.ticker as mticker
from collections import Counter
from pandas.tseries.offsets import MonthEnd

# --- Streamlit-oppsett ---
st.set_page_config(
    page_title="OneSixtyNine: Prediksjon av oppslutninga til norske parti (stortingsval)",
    layout="wide"
)
st.title("📈 OneSixtyNine: Prediksjon av oppslutninga til norske parti")

# --- Last inn data ---
url = "https://raw.githubusercontent.com/jensmorten/onesixtynine/main/data/pollofpolls_master.csv"
df = pd.read_csv(url, index_col="Mnd", parse_dates=True)
df = df.sort_index()
df.index = df.index.to_period('M').to_timestamp('M')  # månadsslutt

# --- Map kolonnenamn til nynorsk ---
kolonne_map = {
    'Hoyre': 'Høgre',
    'Rodt': 'Raudt',
    'SP': 'Sp'
}
df = df.rename(columns=kolonne_map)

# --- Sidebar: parametre ---
st.sidebar.title("Kontrollpanel")
st.sidebar.markdown("### ⚙️ Set modellparametrar:", unsafe_allow_html=True)

lags = st.sidebar.number_input(
    "📅 Talet på månader å bruke til tilpassing (trening) av modellen (maks 12):",
    min_value=1, max_value=12, value=3, step=1
)

smooth = st.sidebar.checkbox(
    "🔀Utjamna prediksjon, tilpassing med +/-2 månader", value=True
)

adjust = st.sidebar.checkbox(
    "🔧 Juster prediksjon basert på val i 2021", value=False
)

n_months = st.sidebar.number_input(
    "📅 Månader framover å predikere:",
    min_value=1, max_value=24, value=6, step=1
)
months_back = st.sidebar.number_input(
    "📅 Månader bakover i tid å vise i plottet:",
    min_value=6, max_value=36, value=12, step=1
)
months_back_start = st.sidebar.number_input(
    "📅Månader bakover å starte prediksjon frå:",
    min_value=0, max_value=months_back, value=0, step=1
)

if months_back_start > 0:
    df = df[:-months_back_start]

# --- Tilpass VAR-modell for hovedlags ---
model = VAR(df)
model_fitted = model.fit(maxlags=lags, method="ols", trend="n", verbose=False)

# --- Prediksjon ---
if smooth:
    lags_list = [l for l in range(lags-2, lags+3) if 1 <= l <= 12]
    preds = []
    lowers = []
    uppers = []
    
    for l in lags_list:
        model_fitted_tmp = model.fit(maxlags=l, method="ols", trend="n", verbose=False)
        forecast_tmp, forecast_lower_tmp, forecast_upper_tmp = model_fitted_tmp.forecast_interval(
            model_fitted_tmp.endog, steps=n_months
        )
        preds.append(forecast_tmp)
        lowers.append(forecast_lower_tmp)
        uppers.append(forecast_upper_tmp)
    
    forecast = np.mean(preds, axis=0)
    forecast_lower = np.mean(lowers, axis=0)
    forecast_upper = np.mean(uppers, axis=0)
else:
    forecast, forecast_lower, forecast_upper = model_fitted.forecast_interval(
        model_fitted.endog, steps=n_months
    )

forecast_index = pd.date_range(start=df.index[-1], periods=n_months+1, freq='M')[1:]

forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0) * 100
forecast_lower_df = pd.DataFrame(forecast_lower, index=forecast_index, columns=df.columns)
forecast_upper_df = pd.DataFrame(forecast_upper, index=forecast_index, columns=df.columns)

# --- Juster prediksjon for val ---
if adjust:
    justeringar = {
        'Ap': 2.0,
        'Høgre': -0.2,
        'Frp': 1.5,
        'SV': -0.5,
        'Sp': -0.5,
        'KrF': 0.4,
        'Venstre': 1.1,
        'MDG': -0.8,
        'Raudt': -0.6,
        'Andre': -0.2
    }

    for parti, adj in justeringar.items():
        if parti in forecast_df.columns:
            forecast_df[parti] += adj
            #forecast_lower_df[parti] += adj
            #forecast_upper_df[parti] += adj

    # Normaliser til 100 % igjen
    forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0) * 100
    #forecast_lower_df = forecast_lower_df.div(forecast_lower_df.sum(axis=1), axis=0) * 100
    #forecast_upper_df = forecast_upper_df.div(forecast_upper_df.sum(axis=1), axis=0) * 100

# --- Formatering av datoar ---
norske_mnd = {
    1: "jan", 2: "feb", 3: "mars", 4: "apr", 5: "mai", 6: "juni",
    7: "juli", 8: "aug", 9: "sep", 10: "okt", 11: "nov", 12: "des"
}

def norsk_dato_formatter(x, pos=None):
    dato = mdates.num2date(x)
    return f"{dato.day}. {norske_mnd[dato.month]} {dato.year}"

colors = {
    'Ap': '#FF6666', 'Høgre': '#6699FF', 'Frp': '#3366CC', 'SV': '#FF9999',
    'Sp': '#339966', 'KrF': '#FFCC66', 'Venstre': '#99CCFF',
    'MDG': '#33CC33', 'Raudt': '#CC3333', 'Andre': '#AAAAAA'
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

# --- Valresultat 2025 ---
pred_datoer= [pd.Timestamp("2025-08-31"), pd.Timestamp("2025-09-30")]
val_dato = pd.Timestamp("2025-09-08")
val_resultat = {
    'Ap': 28.0,
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
    ax.text(val_dato, prosent, "*", color=colors[parti], fontsize=20,
            ha="center", va="center", zorder=6)

indikator_tekst = "Utjamna prediksjon" if smooth else "Standard prediksjon"
if adjust:
    indikator_tekst += " | valjustert"

ax.text(forecast_df_eom.index[-1], 38, indikator_tekst, fontsize=10, color='gray', ha='right')

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

ax.tick_params(axis="x", which="major", length=6, width=1.2, color="black", labelsize=12)
ax.tick_params(axis="y", which="major", length=6, width=1.2, color="black", labelsize=12)

ax.text(pd.Timestamp("2025-09-08"), 30, "* Valresultat 2025:",
        fontsize=10, ha="center", va="bottom",  color='gray',
        bbox=dict(facecolor="white"))

plt.tight_layout()
st.pyplot(fig, use_container_width=False)

# --- Validering (same as before) ---
sjekk_dato = pd.Timestamp("2025-08-01")

if months_back_start > 0 and df.index[-1] < sjekk_dato and all(d in forecast_df.index for d in pred_datoer):
    tekst = "### 🎯 Sjekk kor godt OneSixtyNine predikerer valresultatet i 2025!\n\n"
    tekst += f" Vi set dagens dato til {df.index[-1].strftime('%d. %b %Y')} \n\n"
    
    resultat_per_parti = {}
    total_diff_poll = 0
    total_diff_modell = 0
    
    for parti, val_perc in val_resultat.items():
        siste_dato_poll = df.index[-1]
        siste_poll = df[parti].iloc[-1]
        
        pred_values = []
        for mnd in [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-31")]:
            if mnd in forecast_df.index:
                pred_values.append(forecast_df[parti].loc[mnd])
        modell_pred = np.mean(pred_values) if pred_values else float('nan')
        
        diff_poll = abs(siste_poll - val_perc)
        diff_modell = abs(modell_pred - val_perc)
        
        total_diff_poll += diff_poll
        total_diff_modell += diff_modell

        nærmast = "✨ OneSixtyNine" if diff_modell <= diff_poll else "📊 Poll of polls"
        resultat_per_parti[parti] = nærmast
        
        tekst += (
            f"- **{parti}** (valresultat 2025 {val_perc}%): siste poll ({siste_dato_poll.strftime('%d. %b %Y')}) var {siste_poll:.1f}%, "
            f"OneSixtyNine predikerte {modell_pred:.1f}% | differanse poll-valresultat {diff_poll:.1f}, differanse modell-valresultat {diff_modell:.1f} → {nærmast} var nærmast\n"
        )
    
    teller = Counter(resultat_per_parti.values())
    if teller.get("✨ OneSixtyNine",0) > teller.get("📊 Poll of polls",0):
        konklusjon = f"\n**Totalt:** OneSixtyNine var nærmast for {teller['✨ OneSixtyNine']} parti, Poll of polls for {teller.get('📊 Poll of polls',0)} → **OneSixtyNine er best i antall parti! 🚀**"
    elif teller.get("📊 Poll of polls",0) > teller.get("✨ OneSixtyNine",0):
        konklusjon = f"\n**Totalt:** Poll of polls var nærmast for {teller['📊 Poll of polls']}, OneSixtyNine for {teller.get('✨ OneSixtyNine',0)} → **Poll of polls er best i antall parti!**"
    else:
        konklusjon = f"\n**Totalt:** OneSixtyNine og Poll of polls var nærmast for like mange parti ⚖️"
    
    tekst += konklusjon

    best_total = "OneSixtyNine" if total_diff_modell <= total_diff_poll else "Poll of polls"
    tekst += (
        f"\n\n**Total differanse for alle parti:** Poll of polls = {total_diff_poll:.1f}, OneSixtyNine = {total_diff_modell:.1f} → **{best_total}** var best i sum! "
    )

    tekst += "\n\nIkkje fornøygd? Juster parameterar i kontrollpanel og prøv igjen! ✨"
    
    st.markdown(tekst)

# --- Info ---
st.sidebar.markdown("""
<hr>
<p>
ℹ️ Prediksjonsmodellen <b>OneSixtyNine</b>* byggjer på historiske meiningsmålingar henta frå 
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
