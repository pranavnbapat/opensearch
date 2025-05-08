# using_custom_models/generate_all_config.py

'''
Why do we need all_config in OpenSearch model registration?
The all_config field helps OpenSearch understand and document the architecture, inference behaviour, and tokenizer
expectations of the model you're registering. While it's not strictly executed, it's used for:

Reference: So others (or future you) can see how the model was configured.

Debugging: If inference gives unexpected results, this config can reveal if something was off (e.g. padding token,
hidden size).

Tooling support: Some features like RAG or hybrid search pipelines can use this info to align inputs/outputs
automatically.

So even though your model is TorchScript-compiled and has its behaviour “baked in”, providing all_config ensures
transparency and compatibility.
'''

from transformers import AutoConfig
import json

model_id = "BAAI/bge-base-en-v1.5"
cfg = AutoConfig.from_pretrained(model_id)

# Dump and escape JSON for OpenSearch
escaped_json = json.dumps(cfg.to_dict()).replace('"', '\\"')
print(escaped_json)
