
# Load API Key
GROQ_API_KEY = "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"
import os

from groq import Groq

client = Groq(
    api_key=GROQ_API_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of fast language models",
        }
    ],
    model="llama-3.2-11b-vision-preview",
)

print(chat_completion.choices[0].message.content)