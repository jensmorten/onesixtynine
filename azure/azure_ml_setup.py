from azure.ai.ml import MLClient
from azure.identity import InteractiveBrowserCredential

def get_ml_client():
    return MLClient(
        InteractiveBrowserCredential(),
        subscription_id="d66c9f2f-c191-456e-ae0a-9e2601c648f5",
        resource_group_name="rg-dp100-labs",
        workspace_name="ws_dp100_labs",
    )

if __name__ == "__main__":
    ml_client = get_ml_client()
    ws = ml_client.workspaces.get("ws_dp100_labs")
    print(ws.location)
