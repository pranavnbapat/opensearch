import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the unified grammar + punctuation model
model_name = "flexudy/t5-small-wav2vec2-grammar-fixer"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def correct_english_text_pipeline(raw_text: str) -> str:
    # Step 1: Cleanup basic spacing issues
    raw_text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', raw_text)
    raw_text = re.sub(r'(?<=[a-zA-Z])(?=\d)', ' ', raw_text)
    raw_text = re.sub(r'(?<=\d)(?=[a-zA-Z])', ' ', raw_text)
    raw_text = re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_text)

    # Step 2: Chunk the text into ~450 character chunks
    chunk_size = 450
    chunks = [raw_text[i:i+chunk_size] for i in range(0, len(raw_text), chunk_size)]

    corrected_chunks = []
    for chunk in chunks:
        input_ids = tokenizer.encode(chunk, return_tensors="pt", truncation=True)
        outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
        corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
        corrected_chunks.append(corrected)

    # Step 3: Join the corrected output
    return ' '.join(corrected_chunks)



# === Example usage ===
text = """
\nTake home message\nHygienograms aresurface bacterial counts that areused tomonitor the\nefficiency ofcleaning and disinfection oftheempty house between flocks .\nEvaluation ofcleaned surfaces after C&D helps tobetter prevent infections\nthrough residual sources ofinfectious material .Who does the sampling and testing ?\n\u2751Only authorized laboratories should perform sampling and testing.\n\u2751Sampling and testing in breeder farms are done by DGZ/ARSIA.\n\u2751For Broilers and layer farms HOSOWO certified companies/ vets can also do the sampling.\nHOSOWO -recognised organisation have earned accreditations for performing  the following laboratory activities:\n\u2751Sample collection for hygienograms\n\u2751Sample analysis for hygienograms\n\u2751Sample collection for stall testing: presence of Salmonella and Campylobacter after cleaning and disinfectingCHECKING THE EFFECTIVENESS OF CLEANING AND DISINFECTION IN POULTRY HOUSES\nFor more information:\n-NETPOULSAFE project : https://www.netpoulsafe.eu\n\"This project has received funding from the European Union's Horizon 2020 research and innovation programme under grant agreement No.101000728 ( NetPoulSafe ).
"""

# Process and print the result
print(correct_english_text_pipeline(text))
