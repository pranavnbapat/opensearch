import json
from datetime import datetime
from utils import *


def process_json_for_opensearch(input_file, output_file):
    """
    Reads a JSON file, applies cleaning functions appropriately, and prepares data for OpenSearch ingestion.
    - Stores both cleaned and original versions of the data.
    - Only indexes cleaned data while keeping original data for search results.
    """

    # Load JSON data
    with open(input_file, "r", encoding="utf-8") as file:
        documents = json.load(file)

    processed_documents = []

    for doc in documents:
        cleaned_doc = {}  # Stores cleaned fields for OpenSearch indexing
        original_doc = {}  # Stores original fields for search results

        # Fields that need light cleaning
        for key in ["title", "summary", "projectName", "projectAcronym"]:
            if key in doc:
                value = remove_extra_quotes(str(doc[key]))  # Convert to string and remove extra quotes
                value = value.strip()  # Final trim for safety
                if value:
                    cleaned_doc[key] = clean_text_light(value)  # Cleaned version
                    original_doc[key] = value  # Keep cleaned original

        # Fields that need moderate cleaning (lists remain as lists)
        for key in ["keywords", "topics", "subtopics", "locations", "languages"]:
            if key in doc:
                if isinstance(doc[key], list):
                    # Step 1: Remove empty values and whitespace BEFORE cleaning
                    pre_cleaned_values = [str(item).strip() for item in doc[key] if isinstance(item, str) and item.strip()]

                    # Step 2: Apply cleaning to each element
                    cleaned_values = [clean_text_moderate(item) for item in pre_cleaned_values]

                    # Step 3: Remove empty values AGAIN after cleaning
                    cleaned_values = [item for item in cleaned_values if item.strip()]

                    if cleaned_values:  # Only store if there's at least one valid value
                        cleaned_doc[key] = cleaned_values
                        original_doc[key] = cleaned_values
                    elif key in cleaned_doc:  # Remove empty lists if they exist
                        del cleaned_doc[key]

                else:
                    value = str(doc[key]).strip()  # Ensure it's a string and remove spaces
                    if value:  # Only store if not empty
                        cleaned_doc[key] = clean_text_moderate(value)
                        original_doc[key] = value

        # Extract and clean content_pages from ko_content
        if "ko_content" in doc and isinstance(doc["ko_content"], list):
            content_text = " ".join([
                " ".join(content.get("content", {}).get("content_pages", []))
                for content in doc["ko_content"]
            ])
            if content_text.strip():
                cleaned_doc["content_pages"] = clean_text_extensive(content_text)

        # Store only these fields in the original version (returned in search results)
        search_result_fields = ["title", "creators", "topics", "fileType", "keywords", "dateCreated", "_orig_id", "@id"]
        for key in search_result_fields:
            if key in doc:
                original_doc[key] = doc[key]

        # Convert dateCreated to ISO 8601 format
        if "dateCreated" in doc:
            original_date = str(doc["dateCreated"]).strip()  # Ensure it's a string
            try:
                if "-" in original_date and len(original_date.split("-")[0]) == 4:
                    # Already in YYYY-MM-DD format, no need to convert
                    cleaned_doc["dateCreated"] = original_date
                else:
                    # Convert from DD-MM-YYYY to YYYY-MM-DD
                    parsed_date = datetime.strptime(original_date, "%d-%m-%Y").strftime("%Y-%m-%d")
                    cleaned_doc["dateCreated"] = parsed_date
            except ValueError:
                cleaned_doc["dateCreated"] = original_date  # Use original if parsing fails

            original_doc["dateCreated"] = original_date  # Keep the original format

        # Prepare the final document for OpenSearch ingestion
        processed_doc = {
            **cleaned_doc,  # Cleaned data for indexing
            **original_doc  # Original data for returning in search
        }

        processed_documents.append(processed_doc)

    # Write processed JSON data to a file
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(processed_documents, file, ensure_ascii=False, indent=4)

    print(f"Processed data saved to {output_file}")


# Usage
process_json_for_opensearch("final_output_05_03-2025_16-10-06.json", "opensearch_data_ingestion.json")