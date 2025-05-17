# whisper_transcriber.py

import os
from whisper_model import model  # import preloaded model

def transcribe_file(file_path: str, task: str = "transcribe") -> dict:
    """
    Transcribes or translates a media file using preloaded Whisper model.

    Args:
        file_path (str): Path to the input video/audio file.
        task (str): 'transcribe' to keep original language, 'translate' to convert to English.

    Returns:
        dict: {
            'language': Detected language,
            'duration': Audio duration (seconds),
            'text': Transcribed or translated text
        }
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    result = model.transcribe(file_path, task=task)

    duration = sum([seg["end"] - seg["start"] for seg in result["segments"]])

    return {
        "language": result.get("language"),
        "duration": duration,
        "text": result.get("text")
    }

file_path = "../temp/Powder Laali.mp4"

try:
    result = transcribe_file(file_path)
    print("\n✅ Language Detected:", result["language"])
    print("⏱️ Duration:", result["duration"], "seconds")
    print("\n📝 Transcription:\n", result["text"])
except Exception as e:
    print("❌ Error:", str(e))
