# # using_custom_models/register_deploy_custom_model.py
#
# import time
#
# from utils import client
#
#
# # STEP 1: Register a model group
# def register_model_group(name: str, description: str) -> str:
#     body = {
#         "name": name,
#         "description": description,
#         "access_mode": "public"
#     }
#     response = client.transport.perform_request(
#         method="POST",
#         url="/_plugins/_ml/model_groups/_register",
#         body=body
#     )
#     print(f"[✔] Registered model group: {name}")
#     return response["model_group_id"]
#
#
# # STEP 2: Register the model
# def register_model(model_name: str, model_format: str, model_group_id: str) -> str:
#     body = {
#         "name": model_name,
#         "version": "1.0",
#         "model_group_id": model_group_id,
#         "model_format": model_format
#     }
#     response = client.transport.perform_request(
#         method="POST",
#         url="/_plugins/_ml/models/_register",
#         body=body
#     )
#     print(f"[✔] Registration task created for model: {model_name}")
#     return response["task_id"]
#
#
# # STEP 3: Get model_id from task_id
# def get_model_id_from_task(task_id: str, max_attempts: int = 20, delay_secs: int = 10) -> str:
#     for attempt in range(max_attempts):
#         response = client.transport.perform_request(
#             method="GET",
#             url=f"/_plugins/_ml/tasks/{task_id}"
#         )
#
#         state = response.get("state", "UNKNOWN")
#         model_id = response.get("model_id")
#
#         print(f"Attempt {attempt + 1}: Task state = {state}")
#
#         if state == "COMPLETED" and model_id:
#             print(f"[✔] Model ID resolved: {model_id}")
#             return model_id
#         elif state == "FAILED":
#             raise RuntimeError(f"[✖] Model registration failed: {response}")
#
#         time.sleep(delay_secs)
#
#     raise TimeoutError(f"[✖] Model registration timed out after {max_attempts * delay_secs} seconds.")
#
# # STEP 4: Deploy the model
# def deploy_model(model_id: str) -> str:
#     response = client.transport.perform_request(
#         method="POST",
#         url=f"/_plugins/_ml/models/{model_id}/_deploy"
#     )
#     print(f"[✔] Deployment initiated for model ID: {model_id}")
#     return response["task_id"]
#
#
# # STEP 5: Confirm deployment status
# def wait_for_deployment(task_id: str, max_attempts: int = 20, delay_secs: int = 10):
#     for attempt in range(max_attempts):
#         response = client.transport.perform_request(
#             method="GET",
#             url=f"/_plugins/_ml/tasks/{task_id}"
#         )
#
#         state = response.get("state", "UNKNOWN")
#         print(f"Deployment status check {attempt + 1}: State = {state}")
#
#         if state == "COMPLETED":
#             print("[✔] Model successfully deployed.")
#             return
#         elif state == "FAILED":
#             raise RuntimeError(f"[✖] Model deployment failed: {response}")
#
#         time.sleep(delay_secs)
#
#     raise TimeoutError("[✖] Model deployment timed out.")
#
#
# if __name__ == "__main__":
#     print("\n🔧 Register and Deploy Custom Model to OpenSearch ML Plugin\n")
#
#     model_name = input("👉 Enter full HuggingFace model name (e.g. BAAI/bge-small-en-v1.5): ").strip()
#     model_format = input("👉 Enter model format [default: TORCH_SCRIPT]: ").strip() or "TORCH_SCRIPT"
#     model_group_name = input("👉 Enter model group name [default: EUF Custom Models]: ").strip() or "EUF Custom Models"
#     description = input("👉 Enter model group description [default: Custom model for semantic search]: ").strip() or "Custom model for semantic search"
#
#     print("\n[🚀] Starting registration...")
#
#     model_group_id = register_model_group(model_group_name, description)
#     task_id = register_model(model_name, model_format, model_group_id)
#     model_id = get_model_id_from_task(task_id)
#     deploy_task_id = deploy_model(model_id)
#     wait_for_deployment(deploy_task_id)
#
#     print(f"\n✅ [DONE] Model ID: {model_id}")
#     print(f"ℹ️ Deployment Task ID: {deploy_task_id}")
#
