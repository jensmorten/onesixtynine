from azure.ai.ml import command, Input
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

from azure_ml_setup import get_ml_client

ml_client = get_ml_client()

# --------------------------------------------------
# Custom environment WITH statsmodels
# --------------------------------------------------
env = Environment(
    name="var-residual-env",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

# --------------------------------------------------
# Command job
# --------------------------------------------------
job = command(
    code=".",
    command="python train_residual_job.py --input_data ${{inputs.polls}}",
    inputs={
        "polls": Input(
            type=AssetTypes.URI_FILE,
            path="azureml:pollofpolls_uri:1",
        )
    },
    environment=env,
    compute="jmn-ml-compute",
    experiment_name="var_residual_automl",
)

returned_job = ml_client.jobs.create_or_update(job)
ml_client.jobs.stream(returned_job.name)
