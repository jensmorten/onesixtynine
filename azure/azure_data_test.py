from azure.ai.ml import automl, Input
from azure.ai.ml.constants import AssetTypes
from azure_ml_setup import get_ml_client

ml_client = get_ml_client()

job = automl.regression(
    compute="jmn-ml-compute",
    experiment_name="polls_Ap_test",
    training_data=Input(
        type=AssetTypes.MLTABLE,
        path="azureml:pollofpolls:1",
    ),
    target_column_name="Ap",   # <-- MUST EXIST
    primary_metric="normalized_root_mean_squared_error",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
