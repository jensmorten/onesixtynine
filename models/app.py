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
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor

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

aar = st.sidebar.number_input(
    "📅 Modellens treningsdata starter fra år",
    min_value=2008, max_value=2018, value=2011, step=1
)

start_år=aar
antall_ar=aar-2008
if antall_ar>0:
    df = df[antall_ar*12:]

prediksjonsmodus = st.sidebar.radio(
    "🧠 Prediksjonsmetode:",
    options={
        "ML-optimert VAR (med LightGBM)",
        "Standard VAR",
    },
    index=1
)

# Avleidde kontrollvariablar (brukast vidare i koden)
ml_opt = (prediksjonsmodus == "ML-optimert VAR (med LightGBM)")

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

# st.sidebar.markdown("---")

# run_model = st.sidebar.button(
#     "🚀 Køyr modell",
#     help="Klikk for å berekne prognose med vald modellrekkevidde"
# )

# if not run_model and ml_opt:
#     st.info("👈 Du har vald ML-optimert prognose, vil ta litt tid. For å lage progrnose, trykk **Køyr modell**")
#     st.stop()



if months_back_start > 0:
    df = df[:-months_back_start]

def hybrid_var_ml_forecast(df, n_months, var_lags, lags_ML, tau, vol_window, min_alpha, max_alpha):
    """
    Hybrid VAR + ML with:
      - adaptive α per party
      - regime gating based on volatility ratio
      - horizon decay

    Regime strength = recent_vol / long_run_vol
    """
    model = VAR(df)
    var_res = model.fit(maxlags=var_lags, method="ols", trend="n")

    mean_var, lower_var, upper_var = var_res.forecast_interval(
        var_res.endog, steps=n_months
    )

    # VAR residuals (in-sample)
    fitted = var_res.fittedvalues
    true = df.iloc[var_res.k_ar:]
    resid = true.values - fitted.values

    # Build ML dataset
    X, y = [], []
    for i in range(lags_ML, len(df)):
        if i - var_res.k_ar < 0:
            continue
        X.append(df.iloc[i-lags_ML:i].values.flatten())
        y.append(resid[i - var_res.k_ar])

    X = np.asarray(X)
    y = np.asarray(y)

    n_parties = df.shape[1]
    ml_resid_forecast = np.zeros((n_months, n_parties))

    if len(X) < 100:
        return mean_var, lower_var, upper_var

    # Pre-compute volatility regime indicators
    ddf = df.diff()

    recent_vol = (
        ddf.iloc[-vol_window:]
        .abs()
        .mean(axis=0)
        .values
    )

    long_run_vol = (
        ddf.abs()
        .mean(axis=0)
        .values
        + 1e-8
    )

    regime_strength = recent_vol / long_run_vol  # per party

    for j in range(n_parties):
        yj = y[:, j]

        # --- ML model ---
        model_ml = LGBMRegressor(
            n_estimators=1500,
            num_leaves=16,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=123,
            verbose=-1
        )
        model_ml.fit(X, yj)

        # --- adaptive α (in-sample calibration) ---
        y_hat_train = model_ml.predict(X)

        num = np.dot(yj, y_hat_train)
        den = np.dot(y_hat_train, y_hat_train) + 1e-8
        alpha_j = num / den
        alpha_j = float(np.clip(alpha_j, min_alpha, max_alpha))

        # --- regime-gated weight ---
        rs = regime_strength[j]

        if rs < 0.8:
            regime_weight = 0.0         # calm regime → VAR only
        elif rs < 1.2:
            regime_weight = 2*(rs - 0.8) / (1.2 - 0.8)  # linear ramp 
        else:
            regime_weight = 2.0         # regime change
        
        #regime_weight = 1.0 
        # --- forecasting ---
        win = df.values[-lags_ML:].copy()

        for t in range(n_months):
            r_raw = model_ml.predict(win.reshape(1, -1))[0]

            if tau is not None:
                time_decay = np.exp(-t / tau)
            else:
                time_decay = 1.0

            gamma=2

            r = alpha_j * regime_weight * time_decay * r_raw * gamma

            ml_resid_forecast[t, j] = r

            # IMPORTANT: no feedback of corrected forecast
            win = np.vstack([win[1:], mean_var[t]])

    forecast = mean_var + ml_resid_forecast
    lower = lower_var + ml_resid_forecast
    upper = upper_var + ml_resid_forecast

    return forecast, lower, upper

# --- Tilpass VAR-modell for hovedlags ---
model = VAR(df)

# --- Prediksjon ---
if  ml_opt:
    with st.spinner("Reknar ML-optimert prognose, dette tar litt tid. Maskinlæringsmodellen LightGBM  blir tilpassa til VAR-modellens residualar. "):
        forecast, forecast_lower, forecast_upper = hybrid_var_ml_forecast(
            df=df,
            n_months=n_months,
            var_lags=4,
            lags_ML=12,
            tau=6,
            vol_window=3,
            min_alpha=0.2,
            max_alpha=1.0,
    )
else:
    model_fitted = model.fit(maxlags=4, method="ols", trend="n", verbose=False)
    forecast, forecast_lower, forecast_upper = model_fitted.forecast_interval(
        model_fitted.endog, steps=n_months
    )

forecast_index = pd.date_range(start=df.index[-1], periods=n_months+1, freq='M')[1:]

forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=df.columns)
forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0) * 100
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

if ml_opt:
    indikator_tekst = "ML-optimert estimat"
else:
    indikator_tekst = "Standard VAR"

ax.text(forecast_df_eom.index[-1], 38, indikator_tekst, fontsize=10, color='gray', ha='right')

ax.set_xlim(df_recent_eom.index[0], forecast_df_eom.index[-1])
ax.set_ylim(0, 40)
ax.set_ylabel("Oppslutning (%)", fontsize=12)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(norsk_dato_formatter))

siste_dato = df.index[-1]
siste_dato_norsk = f"{siste_dato.day}. {norske_mnd[siste_dato.month]} {siste_dato.year}"
ax.set_title(
    f"Prediksjon basert på historikk fra {aar},  {n_months} månader framover frå {siste_dato_norsk}",
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
    
    modell_preds = {}
    siste_polls = {}

    for parti, val_perc in val_resultat.items():
        siste_dato_poll = df.index[-1]
        siste_poll = df[parti].iloc[-1]
        
        pred_values = []
        for mnd in [pd.Timestamp("2025-09-30"), pd.Timestamp("2025-10-31")]:
            if mnd in forecast_df.index:
                pred_values.append(forecast_df[parti].loc[mnd])

        weights = np.array([3, 1])             
        #modell_pred = np.mean(pred_values) if pred_values else float('nan')
        modell_pred = np.average(pred_values, weights=weights[:len(pred_values)])
        
        modell_preds[parti] = np.round(modell_pred,1)
        siste_polls[parti] = np.round(siste_poll,1)

        diff_poll = round(abs(siste_poll - val_perc),1)
        diff_modell = round(abs(modell_pred - val_perc),1)
        
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
    # --- Nytt: Blokk-samanlikning (Raudgrøn vs Blå) ---
    raudgron = ['Raudt', 'SV', 'Ap', 'Sp', 'MDG']
    bla = ['Høgre', 'Frp', 'KrF', 'Venstre']

    # valresultat blokksum
    val_raud = np.nansum([val_resultat.get(p, np.nan) for p in raudgron])
    val_bla = np.nansum([val_resultat.get(p, np.nan) for p in bla])
    val_vinnar = "raudgrøn" if val_raud > val_bla else "blå" if val_bla > val_raud else "ingen (likt)"

    # poll blokksum
    poll_raud = np.nansum([siste_polls.get(p, np.nan) for p in raudgron])
    poll_bla = np.nansum([siste_polls.get(p, np.nan) for p in bla])
    poll_vinnar = "raudgrøn" if poll_raud > poll_bla else "blå" if poll_bla > poll_raud else "ingen (likt)"

    # modell blokksum
    modell_raud = np.nansum([modell_preds.get(p, np.nan) for p in raudgron])
    modell_bla = np.nansum([modell_preds.get(p, np.nan) for p in bla])
    modell_vinnar = "raudgrøn" if modell_raud > modell_bla else "blå" if modell_bla > modell_raud else "ingen (likt)"

    tekst += "\n\n### 🧾 Blokk-samanlikning — Raudgrøn vs Blå\n"

    tekst += f"- **Valresultatet (2025):** {val_vinnar} side var størst. Raudgrøn side: {np.round(val_raud,1)}, blå side: {np.round(val_bla,1)} \n"
    tekst += f"- **Siste poll (pollofpolls):** {poll_vinnar} side vart berekna størst. Raudgrøn side: {np.round(poll_raud,1)}, blå side: {np.round(poll_bla,1)}.\n"
    tekst += f"- **OneSixtyNine-modellen:** {modell_vinnar} side vart berekna størst.  Raudgrøn side: {np.round(modell_raud,1)}, blå side: {np.round(modell_bla,1)}.\n\n"

    # vurder kven som hadde rett
    poll_rett = (poll_vinnar == val_vinnar)
    modell_rett = (modell_vinnar == val_vinnar)

    if poll_rett and modell_rett:
        tekst += "**Konklusjon:** Både Poll of polls og modellen berekna riktig blokk-vinnar. 🎯\n"
    elif poll_rett and not modell_rett:
        tekst += "**Konklusjon:** Poll of polls traff riktig blokk-vinnar, modellen gjorde det ikkje. 📊\n"
    elif modell_rett and not poll_rett:
        tekst += "**Konklusjon:** Modellen traff riktig blokk-vinnar, Poll of polls gjorde det ikkje. ✨\n"
    else:
        tekst += "**Konklusjon:** Ingen av dei berekna korrekt blokk-vinnar. ❌\n"

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
