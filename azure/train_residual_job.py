import argparse
import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR

from azure.ai.ml import automl, MLClient
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
VAR_LAGS = 4
LAGS_ML = 12

ALLOWED_ALGOS = ["LightGBM", "ElasticNet"]

AUTO_ML_LIMITS = dict(
    timeout_minutes=10,
    max_trials=8,
)


# --------------------------------------------------
# VAR + RESIDUALS
# --------------------------------------------------
def build_var_residual_dataset(df):
    model = VAR(df)
    var_res = model.fit(maxlags=VAR_LAGS, method="ols", trend="n")

    fitted = var_res.fittedvalues
    true = df.iloc[var_res.k_ar:]
    resid = true.values - fitted.values

    X, y = [], []
    for i in range(LAGS_ML, len(df)):
        if i - var_res.k_ar < 0:
            continue
        X.append(df.iloc[i - LAGS_ML:i].values.flatten())
        y.append(resid[i - var_res.k_ar])

    return np.asarray(X), np.asarray(y)


def make_training_df(X, y):
    df_X = pd.DataFrame(X, columns=[f"x_{i}" for i in range(X.shape[1])])
    df_y = pd.Series(y, name="target")
    return pd.concat([df_X, df_y], axis=1)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main(args):
    print("Loading polling data...")
    df = pd.read_csv(args.input_data)
    df["Mnd"] = pd.to_datetime(df["Mnd"])
    df = df.sort_values("Mnd").set_index("Mnd")

    print("Fitting VAR and computing residuals...")
    X, y = build_var_residual_dataset(df)

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group,
        workspace_name=args.workspace,
    )

    parties = df.columns.tolist()

    for j, party in enumerate(parties):
        print(f"\nTraining AutoML residual model for party: {party}")

        train_df = make_training_df(X, y[:, j])
        train_path = f"train_{party}.csv"
        train_df.to_csv(train_path, index=False)

        job = automl.regression(
            compute=args.compute,
            experiment_name=f"residuals_{party}",
            training_data=train_path,
            target_column_name="target",
            primary_metric="normalized_root_mean_squared_error",
            allowed_training_algorithms=ALLOWED_ALGOS,
            limits={
            "timeout_minutes": 10,
            "max_trials": 8,
            },
        )

        returned_job = ml_client.jobs.create_or_update(job)
        ml_client.jobs.stream(returned_job.name)

        best_model = ml_client.jobs.get_output(returned_job.name, "best_model")

        model = Model(
            path=best_model.path,
            name=f"residual_model_{party}",
            description="VAR + AutoML residual correction",
        )

        ml_client.models.create_or_update(model)
        print(f"Registered model: residual_model_{party}")

    print("\n✅ All residual models trained successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--subscription_id", type=str)
    parser.add_argument("--resource_group", type=str)
    parser.add_argument("--workspace", type=str)
    parser.add_argument("--compute", type=str)

    args = parser.parse_args()
    main(args)
