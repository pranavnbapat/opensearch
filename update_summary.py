# update_summary.py

import json
import os
import requests
from urllib.parse import urlparse

OLLAMA_RUNPOD_HOST = "https://v9zpf5kzgkvf1r-11434.proxy.runpod.net"
LLM_MODEL_NAME = "mistral"


# Function to get the latest file from the 'raw_data' folder
def get_latest_json_file(folder="raw_data"):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON files found in the raw_data folder!")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


# Function to check if a URL appears to be a downloadable text-based file
def is_text_file_url(url):
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1].lower()
    return ext in [".txt", ".pdf", ".doc", ".docx"]

# Simulated function to call the summarisation service (replace with actual call)
def call_summary_service(file_url):
    """
    Replace this with actual API call logic, e.g.:
    response = requests.post("http://your-api/summary", json={"url": file_url})
    return response.json()["summary"]
    """
    return f"Simulated summary from Runpod for: {file_url}"


def summarise_with_llm(text, filename="", doc_type="auto", metadata=None, model=LLM_MODEL_NAME):
    if metadata is None:
        metadata = {}

    meta_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])

    prompt = f"""
You are an intelligent assistant that analyses a wide variety of text documents — including invoices, scientific 
reports, meeting notes, technical summaries, project deliverables, and research findings.

Document filename: {filename}
Document type: {doc_type}
Additional metadata:
{meta_str}

Extracted document text:
{text}

Please do the following:
- Identify the type of document (e.g. factsheet, scientific or technical paper, summary note, project information, 
practice abstracts, etc.).
- Then, write a short paragraph (approximately five to ten sentences) that clearly summarises the document.
- Do not include bullet points, lists, or headings. Write as if you're explaining the document naturally to a colleague.
- Focus on what is clearly present in the text. Do not guess or invent details.

Return your response in plain text.
"""

    try:
        response = requests.post(
            f"{OLLAMA_RUNPOD_HOST}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            verify=False
        )
        return response.json()['message']['content']

    except Exception as e:
        print(f"[LLM ERROR] Failed to summarise: {e}")
        return None


def extract_text_from_ko_content(obj):
    try:
        pages = obj["ko_content"][0]["content"]["content_pages"]
        return "\n\n".join(pages)
    except Exception:
        return None


def process_latest_file():
    folder = "raw_data"  # change if needed
    latest_file_path = get_latest_json_file(folder)

    with open(latest_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for i, obj in enumerate(data[:10]):
        url = obj.get("@id", "")
        if not url or not is_text_file_url(url):
            continue  # skip if no usable file

        try:
            head = requests.head(url, timeout=10)
            if head.status_code != 200:
                continue

            extracted_text = extract_text_from_ko_content(obj)
            if not extracted_text:
                print(f"[WARN] No content extracted for: {url}")
                continue

            # Include only selected metadata fields
            metadata = {
                "title": obj.get("title", ""),
                "projectName": obj.get("projectName", ""),
                "topics": ", ".join(obj.get("topics", []))
            }

            summary = summarise_with_llm(
                text=extracted_text,
                filename=os.path.basename(urlparse(url).path),
                doc_type=obj.get("fileType", "auto"),
                metadata=metadata
            )

            if summary:
                obj["summary_runpod"] = summary

        except Exception as e:
            print(f"[ERROR] Skipped due to: {e}")
            continue

    # Save updated JSON to a new file
    updated_file_path = latest_file_path.replace(".json", "_with_runpod_summary.json")
    with open(updated_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Updated file written to: {updated_file_path}")

if __name__ == "__main__":
    process_latest_file()
