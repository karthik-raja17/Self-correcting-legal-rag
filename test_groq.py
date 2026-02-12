#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from groq import Groq

# Load your API key from .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY not found in .env file")
    print("   Create a .env file with: GROQ_API_KEY=gsk_your_key_here")
    exit(1)

# Create client
client = Groq(api_key=api_key)

# Test completion
try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Say 'Groq connection OK'"}],
        temperature=0.1,
        max_tokens=20
    )
    response = completion.choices[0].message.content
    print(f"✅ Groq test successful: {response}")
except Exception as e:
    print(f"❌ Groq test failed: {e}")
