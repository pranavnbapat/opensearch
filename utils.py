import re
import unicodedata


def clean_text_light(text):
    """
    Light cleaning: For title, summary, project name, acronym.
    - Lowercase conversion
    - Trim whitespace
    - Remove extra spaces and newlines
    - Preserve alphanumeric characters, accents, and essential punctuation
    """
    if not isinstance(text, str):
        return text

    text = text.strip()  # Remove leading/trailing spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    return text


def clean_text_moderate(text):
    """
    Moderate cleaning: For keywords, topics, subtopics, locations, languages, file type.
    - Lowercase conversion
    - Remove excessive punctuation (keep hyphens & underscores)
    - Normalize spaces
    """
    if not isinstance(text, str):
        return text

    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'[“”"‘’]', '', text)  # Remove fancy quotes
    text = re.sub(r'[|]', '', text)  # Remove pipe characters
    text = re.sub(r'[^\w\s\-_]', '', text)  # Remove all other punctuation except hyphen/underscore
    return text


def chunk_text_by_tokens(text, tokenizer, max_tokens=480, overlap=80):
    # tokens = tokenizer.encode(text, truncation=False)

    # Tokenize the text into token IDs (no truncation, returns warning if >512)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]

        # Safety check: never let a chunk exceed 512 tokens (just in case overlap math breaks it)
        if len(chunk_tokens) > 512:
            chunk_tokens = chunk_tokens[:512]

        # Decode token IDs back into text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)

        token_count = len(tokenizer.encode(chunk_text, add_special_tokens=False))
        if token_count > 512:
            print(f"Warning: Overlong chunk detected ({token_count} tokens)")

        chunks.append(chunk_text.strip())

        # Move to next chunk with overlap
        start += (max_tokens - overlap)
    return chunks


def clean_text_extensive(text):
    """
    Extensive cleaning: For content pages.
    - Convert to lowercase
    - Remove all special characters except hyphens and underscores
    - Remove standalone numbers
    - Remove escape sequences (\n, \r, \t, etc.)
    - Normalize accented characters
    - Remove extra spaces
    """
    if not isinstance(text, str):
        return text

    text = text.lower().strip()

    # Normalize Unicode characters (preserves accents)
    text = unicodedata.normalize('NFKC', text)

    # Remove all escape sequences (e.g., \n, \r, \t, \x, \u)
    text = re.sub(r'\\[a-zA-Z0-9]+', '', text)

    # Remove newlines, tabs, carriage returns explicitly
    text = re.sub(r'[\n\r\t\f\v]', ' ', text)

    # Remove quotation marks, pipes, and other unnecessary punctuation
    text = re.sub(r'[“”"‘’]', '', text)  # Remove fancy quotes
    text = re.sub(r'[|]', '', text)  # Remove pipe characters

    # Remove any remaining non-alphanumeric characters (except hyphens and underscores)
    text = re.sub(r'[^a-zA-Z0-9\s\-_]', '', text)

    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def remove_extra_quotes(text):
    """
    Removes extra leading/trailing double quotes but keeps quotes within the text.
    """
    if isinstance(text, str):
        text = text.strip()
        text = re.sub(r'^[\'"]+|[\'"]+$', '', text)  # Removes extra quotes from both ends
    return text
