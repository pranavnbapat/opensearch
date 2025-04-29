import json
import os
from opensearchpy import helpers
from utils import client

PROJECTS_INDEX_NAME = os.getenv("PROJECTS_INDEX_NAME", "projects_index")

def reset_index():
    """Deletes and recreates the index with only projectName and projectAcronym fields."""
    try:
        if client.indices.exists(index=PROJECTS_INDEX_NAME):
            client.indices.delete(index=PROJECTS_INDEX_NAME)
            print(f"Deleted index: {PROJECTS_INDEX_NAME}")
        else:
            print(f"Index {PROJECTS_INDEX_NAME} does not exist. Creating fresh.")

        client.indices.create(
            index=PROJECTS_INDEX_NAME,
            body={
                "settings": {"number_of_shards": 1},
                "mappings": {
                    "properties": {
                        "projectName": {"type": "text"},
                        "projectAcronym": {"type": "keyword"}
                    }
                }
            }
        )
        print(f"Recreated index: {PROJECTS_INDEX_NAME}")
    except Exception as e:
        print(f"Error resetting index: {e}")


def get_latest_json_file(folder="raw_data"):
    """Get the latest JSON file from a folder."""
    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        raise FileNotFoundError("No JSON files found in raw_data folder")
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(folder, x)))
    return os.path.join(folder, latest_file)


def process_projects(file_path):
    """Return a list of dicts with only projectName and projectAcronym."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    for doc in raw_data:
        project_name = doc.get("projectName", "").strip()
        acronym = doc.get("projectAcronym", "").strip()

        if project_name and acronym:
            processed.append({
                "_id": doc.get("@id", None),
                "_source": {
                    "projectName": project_name,
                    "projectAcronym": acronym
                }
            })
    return processed


def bulk_index(data):
    """Ingest documents into OpenSearch."""
    if not data:
        print("No data to ingest.")
        return

    success, errors = helpers.bulk(client, data, index=PROJECTS_INDEX_NAME, refresh="wait_for")
    print(f"Ingested {success} documents.")
    if errors:
        print(f"{len(errors)} errors occurred during ingestion.")


if __name__ == "__main__":
    try:
        json_file = get_latest_json_file()
        print(f"Processing file: {json_file}")

        reset_index()

        data = process_projects(json_file)
        bulk_index(data)

        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
