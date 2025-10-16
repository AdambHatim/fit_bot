from openai import OpenAI
import os
os.chdir("/Users/adamh/Desktop/fit_bot/back-end/app")

def get_embedding(openai_client, texts: str) -> list[float]:
    """
    Converts a text string into an embedding (vector of floats)
    using OpenAI's text-embedding-3-small model.
    """
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embedding = response.data[0].embedding
    return embedding

