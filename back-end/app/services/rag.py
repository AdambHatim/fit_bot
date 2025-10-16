import os
import json
from dotenv import load_dotenv
from app.services.openai_client import get_embedding  # your embedding function

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("URL")
API_KEY = os.getenv("API_KEY")


"""
def query_qdrant_and_get_text(text, collection="database", limit=1):
    Generate an embedding for the input text,
    query Qdrant Cloud, and print the top matching payload text.
    
    # Step 1: Get the embedding
    embedding = get_embedding(text)

    # Step 2: Build request payload
    payload = {
        "query": embedding,
        "limit": limit,
        "with_payload": True  # 👈 this makes Qdrant return stored payload data
    }

    # Step 3: Set headers
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    # Step 4: Send request
    response = requests.post(
    f"{QDRANT_URL}/collections/{collection}/points/search",
    headers=headers,
    data=json.dumps(payload),
    timeout=30
)


    # Step 5: Parse results
    if response.status_code != 200:
        print("❌ Query failed:", response.text)
        return None

    results = response.json()["result"]["points"]
    if not results:
        print("No results found.")
        return None

    # Step 6: Extract top point info
    top_point = results[0]
    point_id = top_point["id"]
    score = top_point["score"]
    payload_data = top_point.get("payload", {})
    text_value = payload_data.get("text", "[No text field in payload]")

    print("✅ Top match found:")
    print(f"ID: {point_id}")
    print(f"Score: {score}")
    print(f"Text: {text_value}\n")

    return text_value
"""

def query_qdrant_and_get_text(openai_client, qdrant_client ,text, collection="database", limit=1):
    embedding = get_embedding(openai_client, text)
    data = qdrant_client.search(collection_name="database", query_vector=embedding, limit=1)
    text = data[0].payload['text']
    return text





