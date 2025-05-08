# using_custom_models/download_models/download_and_prepare_models.py

import json
import hashlib
import os
import torch
import torch.nn as nn
import zipfile

from transformers import AutoModel, AutoTokenizer

model_name = "BAAI/bge-small-en-v1.5"
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "models", model_name.replace("/", "_"))
zip_filename = os.path.join(script_dir, f"{model_name.split('/')[-1]}.zip")

os.makedirs(output_dir, exist_ok=True)

print(f"🔽 Downloading model and tokenizer for: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModel.from_pretrained(model_name)
model_config = hf_model.config

print("💾 Saving tokenizer files...")
tokenizer.save_pretrained(output_dir)

# Wrapper to extract the desired embedding (e.g. CLS token or pooler_output)
class WrappedModel(nn.Module):
    def __init__(self, model):
        super(WrappedModel, self).__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # CLS token


print("💾 Scripting model as TorchScript...")
wrapped = WrappedModel(hf_model)
dummy_inputs = tokenizer("example sentence", return_tensors="pt")
traced_model = torch.jit.trace(wrapped, dummy_inputs["input_ids"], strict=False)
traced_model.save(os.path.join(output_dir, "model.pt"))


print(f"\n✅ All files saved in: {output_dir}")

print("🗜️ Zipping model files...")
with zipfile.ZipFile(zip_filename, "w") as zipf:
    for filename in os.listdir(output_dir):
        full_path = os.path.join(output_dir, filename)
        if os.path.isfile(full_path):
            zipf.write(full_path, arcname=filename)
            print(f"✅ Added to zip: {filename}")


def sha256sum(filename):
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


zip_hash = sha256sum(zip_filename)
zip_size = os.path.getsize(zip_filename)

print(f"\n✅ All files saved in: {output_dir}")
print(f"📦 Zip file created: {zip_filename}")
print(f"🔐 SHA-256: {zip_hash}")
print(f"📏 Size (bytes): {zip_size}")

model_config_payload = {
    "model_type": "bert",
    "embedding_dimension": model_config.hidden_size,
    "framework_type": "sentence_transformers",
    "all_config": json.dumps({
        **model_config.to_dict(),
        "_name_or_path": model_name.split("/")[-1],
        "transformers_version": "4.48.3",
        "torchscript": False
    })
}

print("\n🧾 Suggested `model_config` for OpenSearch:")
print(json.dumps({"model_config": model_config_payload}, indent=2))


# Once done, run following
# cd models/MODEL_NAME
# zip -r model_name.zip model.pt tokenizer.json tokenizer_config.json vocab.txt special_tokens_map.json
# Once zipping is done, compute checksum
# shasum -a 256 bge-base-en-v1.5.zip
# Then copy the file to server
# scp -P 19257 bge-base-en-v1.5.zip root@173.212.248.102:/root/euf/opensearch/models/
# Get the filesize from the server using: stat -c%s bge-base-en-v1.5.zip
#

