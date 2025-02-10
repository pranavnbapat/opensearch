import os
import shutil

from transformers import AutoTokenizer, AutoModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Current directory of ths script
MODELS_DIR = os.path.join(BASE_DIR, "models")  # Path to the "models" folder
os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure "models" directory exists

# List of models to download
model_names = [
    "sentence-transformers/msmarco-distilbert-base-tas-b",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

for model_name in model_names:
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Create a directory inside "models" for the model
    model_dir = os.path.join(MODELS_DIR, model_name.replace("/", "_"))
    os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists

    # Save the model and tokenizer inside "models" directory
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)

    # Create a zip archive inside the "models" directory
    zip_path = os.path.join(MODELS_DIR, model_name.replace("/", "_"))
    shutil.make_archive(zip_path, 'zip', root_dir=model_dir)  # Creates models/{model_dir}.zip

    # ✅ Remove the model directory after zipping
    # shutil.rmtree(model_dir)

    print(f"✅ Model {model_name} downloaded and saved in '{model_dir}', zipped as '{zip_path}.zip'")

# scp models/sentence-transformers_msmarco-distilbert-base-tas-b.zip root@173.212.248.102:/root/euf/opensearch/models/
# scp models/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2.zip root@173.212.248.102:/root/euf/opensearch/models/
