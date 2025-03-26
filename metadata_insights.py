import json
from transformers import AutoTokenizer

def get_value_lengths(data, prefix=''):
    value_lengths = {}

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            # Calculate length depending on type
            if isinstance(value, str):
                value_lengths[full_key] = len(value)
            elif isinstance(value, (int, float)):
                value_lengths[full_key] = len(str(value))
            elif isinstance(value, list):
                # Optional: sum of string lengths inside, or just number of items
                value_lengths[full_key] = sum(len(str(v)) for v in value)
                # Recurse into list if it contains dicts
                for index, item in enumerate(value):
                    value_lengths.update(get_value_lengths(item, f"{full_key}[{index}]"))
            elif isinstance(value, dict):
                value_lengths.update(get_value_lengths(value, full_key))
            else:
                # Fallback for other data types
                value_lengths[full_key] = len(str(value))

    elif isinstance(data, list):
        for index, item in enumerate(data):
            full_key = f"{prefix}[{index}]"
            value_lengths.update(get_value_lengths(item, full_key))

    return value_lengths


with open('raw_data/final_output_14_03-2025_22-40-38.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

# key_lengths = get_value_lengths(json_data)
#
# # Print the results nicely
# for key_path, length in key_lengths.items():
#     print(f"{key_path}: {length}")

# Load tokenizer for MS MARCO model
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-distilbert-base-tas-b")

# Token count function
def count_tokens(text):
    """Return number of tokens in a given text."""
    return len(tokenizer.encode(text, truncation=False)) if text else 0

results = []

for entry in json_data:
    _id = entry.get("_orig_id", "N/A")
    title = entry.get("title", "").strip()
    summary = entry.get("summary", "").strip()

    # Safely extract all content_pages
    content_text = ""
    if isinstance(entry.get("ko_content"), list):
        content_parts = []
        for item in entry["ko_content"]:
            content_pages = item.get("content", {}).get("content_pages", [])
            if isinstance(content_pages, list):
                content_parts.extend([p for p in content_pages if isinstance(p, str)])
        content_text = " ".join(content_parts).strip()

    # Token counts
    title_tokens = count_tokens(title)
    summary_tokens = count_tokens(summary)
    content_tokens = count_tokens(content_text)

    results.append({
        "_orig_id": _id,
        "title": title,
        "title_tokens": title_tokens,
        "summary_tokens": summary_tokens,
        "content_tokens": content_tokens
    })

# Sort by content token count (descending)
sorted_results = sorted(results, key=lambda x: x["content_tokens"], reverse=True)

# Save results to JSON
output_path = "token_counts_output.json"
with open(output_path, "w", encoding="utf-8") as out_file:
    json.dump(sorted_results, out_file, ensure_ascii=False, indent=2)

print(f"Token count results saved to {output_path}")
