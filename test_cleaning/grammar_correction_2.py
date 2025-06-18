import os
import sys

from dotenv import load_dotenv
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Load environment variables from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

messages = cast(list[ChatCompletionMessageParam], [
    {"role": "system", "content": "You are a professional editor. Please fix grammar and punctuation."},
    {"role": "user", "content": """
    \nThe number of bacterial colonies ( cfu: colony forming units )in the house aremeasured and given scores\n0 cfu(score \"0\")\n1-40 cfu(score 1)41-120 cfu(score 2)\n121-400 cfu(score 3)> 400 cfu(score 4 ) innumerable (score 5)\nThe report includes the individual sample scores as well as the average scores\nMeasures to be taken according to the scores obtained\n\u22641.5 1.5 \u2264 3.0 > 3.0\nIn agreement.\nNew stock can be introduced Repeat disinfection Introduce new stock after the next vacancy periodRe-cleaning and disinfection and new hygienogram . Introduce new stock after the next vacancy period\nFeed system\nWallFloor\nAnteroomCHECKING THE EFFECTIVENESS OF CLEANING AND DISINFECTION IN POULTRY HOUSES\n
    """}
])

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0
)

print(response.choices[0].message.content.strip())
