import time
import json
import numpy as np
from openai import OpenAI
from app.core.config import OPENAI_KEY

client = OpenAI(api_key=OPENAI_KEY)

def embed_texts_in_batches(
    texts,
    batch_size=50,
    sleep_time=2,
    output_path=r"C:\Users\adamh\Desktop\fit_bot\back-end\app\data\embeddings.npy"
):
    """
    Embeds a list of texts in batches and saves the result to a .npy file.

    Args:
        texts (list[str]): List of text strings to embed
        batch_size (int): How many texts per API call
        sleep_time (int): Seconds to wait between calls
        output_path (str): Filepath for the saved .npy embeddings
    """

    n = len(texts)
    print(f"📘 Starting embedding of {n} texts in batches of {batch_size}...")

    # Preallocate array (1536 dimensions for text-embedding-3-small)
    embeddings_array = np.zeros((n, 1536), dtype=np.float32)

    for i in range(0, n, batch_size):
        batch = texts[i:i + batch_size]

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)

        embeddings_array[i:i + len(batch_embeddings)] = batch_embeddings

        batch_num = i // batch_size + 1
        print(f"✅ Batch {batch_num} embedded ({len(batch_embeddings)} texts)")

        # Sleep only if not the last batch
        if i + batch_size < n:
            print(f"⏳ Waiting {sleep_time}s before next batch...")
            time.sleep(sleep_time)

    # Save embeddings
    np.save(output_path, embeddings_array)
    print(f"\n💾 Saved embeddings to: {output_path}")
    print(f"✅ All done! Final shape: {embeddings_array.shape}")

    return embeddings_array


def main():
    """Example: Load dataset and run embeddings"""
    dataset_path = r"C:\Users\adamh\Desktop\fit_bot\back-end\app\data\fitness_books.json"

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # If your JSON is a dict, convert to a list of texts
    if isinstance(data, dict):
        data = list(data.values())

    print(f"Loaded {len(data)} texts from dataset.")

    text_data = [0 for i in range(len(data))]
    for i in range(len(data)):
        text_data[i] = data[i]['text']


    # Run embeddings
    embed_texts_in_batches(text_data)


if __name__ == "__main__":
    main()
