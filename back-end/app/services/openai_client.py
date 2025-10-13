from openai import OpenAI
from app.core.config import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)

def get_embedding(texts: str) -> list[float]:
    """
    Converts a text string into an embedding (vector of floats)
    using OpenAI's text-embedding-3-small model.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embedding = response.data[0].embedding
    return embedding

