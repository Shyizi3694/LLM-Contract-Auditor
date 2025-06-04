from google import genai
import pandas as pd

client = genai.Client(api_key="AIzaSyBIQu8Rsv-fZ7MH12TNEjjQ43jEZYjwbzo")

response = client.models.generate_content(
    model="gemini-2.0-flash", contents="Explain how AI works in a few words"
)
print(response.text)