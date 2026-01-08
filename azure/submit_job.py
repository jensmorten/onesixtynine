from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes
from azure_ml_setup import get_ml_client

ml_client = get_ml_client()

job = command(
    code=".",
    command="""
    python train_residual_job.py \
      --input_data ${{inputs.polls}} \
      --subscription_id d66c9f2f-c191-456e-ae0a-9e2601c648f5 \
      --resource_group rg-dp100-labs \
      --workspace ws_dp100_labs \
      --compute jmn-ml-compute
    """,
    inputs={
        "polls": Input(
            type=AssetTypes.URI_FILE,
            path="azureml:pollofpolls_uri:1",
        )
    },
    environment="azureml://registries/azureml/environments/sklearn-1.1/labels/latest",
    compute="jmn-ml-compute",
    experiment_name="var_residual_automl",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
