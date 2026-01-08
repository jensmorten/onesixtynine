import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes

# ============================================================
# CONFIG
# ============================================================
DATA_URL = (
    "https://raw.githubusercontent.com/jensmorten/"
    "onesixtynine/main/data/pollofpolls_master.csv"
)

VAR_LAGS = 4
LAGS_ML = 12

PARTY = "Ap"


# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    df = pd.read_csv(DATA_URL, index_col="Mnd", parse_dates=True)
    df = df.sort_index()
    df.index = df.index.to_period("M").to_timestamp("M")
    return df


# ============================================================
# VAR + RESIDUALS
# ============================================================
def build_residual_dataset(df):
    model = VAR(df)
    res = model.fit(maxlags=VAR_LAGS, trend="n")

    fitted = res.fittedvalues
    true = df.iloc[res.k_ar:]
    residuals = true - fitted

    X, y = [], []
    for i in range(LAGS_ML, len(df)):
        if i - res.k_ar < 0:
            continue
        X.append(df.iloc[i - LAGS_ML:i].values.flatten())
        y.append(residuals.iloc[i - res.k_ar][PARTY])

    X = np.asarray(X)
    y = np.asarray(y)

    df_X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    df_y = pd.Series(y, name="target")

    return pd.concat([df_X, df_y], axis=1)


# ============================================================
# SAVE FOR AUTOML
# ============================================================
def main():
    df = load_data()
    train_df = build_residual_dataset(df)

    train_df.to_csv("train.csv", index=False)
    print("Saved train.csv")


if __name__ == "__main__":
    main()
