# whisper_model.py

import whisper

MODEL_SIZE = "large-v3"
model = whisper.load_model(MODEL_SIZE)
