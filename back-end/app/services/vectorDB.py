import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
import numpy as np
import json

def main():
    load_dotenv()
    QDRANT_API_KEY = os.getenv("API_KEY")
    QDRANT_URL = os.getenv("URL")
    path_embeddings = r"C:\Users\adamh\Desktop\fit_bot\back-end\app\data\embeddings.npy"
    path_fitness_book = r"C:\Users\adamh\Desktop\fit_bot\back-end\app\data\fitness_books.json"
    embeddings = np.load(path_embeddings)

    client = QdrantClient(
        url= QDRANT_URL, 
        api_key= QDRANT_API_KEY,
    )

    print(embeddings.shape)

    with open(path_fitness_book, "r", encoding="utf-8") as f:
        data = json.load(f)
    num_vectors, dimension = embeddings.shape
    print(len(data))

    client.recreate_collection(
        collection_name = "database" ,
        vectors_config = VectorParams(size = dimension, distance=Distance.COSINE)
                            )

    for i in range(num_vectors):
            client.upsert(
        collection_name="database",
        points=[
            models.PointStruct(
                id=i,
                vector=embeddings[i,:],
                payload={"author": data[i]["author"], "text": data[i]["text"]}
            )
        ]
    )

if __name__ == "__main__":
        main()