import json
import os
import sys
import textwrap

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# from test_glued_words import find_glued_words
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from utils import (
    clean_text_light,
    clean_text_moderate,
    clean_text_extensive,
)

# print("Loading grammar correction model...")
# grammar_corrector = pipeline(
#     "text2text-generation",
#     model="prithivida/grammar_error_correcter_v1",
#     tokenizer="prithivida/grammar_error_correcter_v1"
# )
# print("Model loaded.\n")

# Initialise transformer-based NER pipeline
# ner_pipe = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
#                     aggregation_strategy="simple")

# Function to get the latest file from the 'raw_data' folder
def get_latest_json_file(folder="../raw_data/"):
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON files found in the raw_data folder!")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


# Function to safely extract content pages
def safe_get_content_pages(obj):
    ko_content = obj.get("ko_content", [])
    if not isinstance(ko_content, list) or len(ko_content) == 0:
        return []
    content_pages = ko_content[0].get("content", {}).get("content_pages", [])
    if not isinstance(content_pages, list):
        return [str(content_pages)]
    return content_pages

# def correct_grammar(text):
#     """Correct glued words and grammar issues using transformer model."""
#     prompt = f"gec: {text.strip()}"
#     try:
#         result = grammar_corrector(prompt, do_sample=False)
#         return result[0]["generated_text"] if result else text
#     except Exception as e:
#         print(f"Grammar correction failed: {e}")
#         return text


def correct_grammar_in_chunks(text, grammar_pipeline, max_chunk_chars=400):
    """
    Splits long text into manageable chunks, corrects each via HuggingFace pipeline, and merges.
    """
    chunks = textwrap.wrap(text, width=max_chunk_chars, break_long_words=False, replace_whitespace=False)
    corrected_chunks = []

    for chunk in chunks:
        try:
            result = grammar_pipeline(f"gec: {chunk}", do_sample=False)
            corrected = result[0]["generated_text"] if result else chunk
        except Exception as e:
            print(f"Correction failed for chunk: {chunk[:50]}... ({e})")
            corrected = chunk

        corrected_chunks.append(corrected.strip())

    return " ".join(corrected_chunks).strip()


# Load the latest file
json_path = get_latest_json_file()
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Take only the first object
obj = data[1]

# Clean fields
title = clean_text_light(obj.get("title", ""))
summary = clean_text_light(obj.get("summary", ""))
project_name = clean_text_light(obj.get("projectName", ""))
project_acronym = clean_text_light(obj.get("projectAcronym", ""))

keywords = [clean_text_moderate(k) for k in obj.get("keywords", [])]
topics = [clean_text_moderate(t) for t in obj.get("topics", [])]
subtopics = [clean_text_moderate(s) for s in obj.get("subtopics", [])]
project_type = clean_text_moderate(obj.get("project_type", ""))
locations = [clean_text_moderate(l) for l in obj.get("locations", [])]
languages = [clean_text_moderate(l) for l in obj.get("languages", [])]

content_pages = safe_get_content_pages(obj)
cleaned_pages = [clean_text_extensive(p, preserve_numbers=True) for p in content_pages]
# corrected_pages = [correct_grammar_in_chunks(p, grammar_corrector) for p in cleaned_pages]

# Print results
print("Title:", title)
print("Summary:", summary)
print("Project Name:", project_name)
print("Project Acronym:", project_acronym)
print("Keywords:", keywords)
print("Topics:", topics)
print("Subtopics:", subtopics)
print("Project Type:", project_type)
print("Locations:", locations)
print("Languages:", languages)
print("\nContent Pages:")

for i, page in enumerate(cleaned_pages, 1):
    print(f"--- Page {i} ---\n{page}\n")
