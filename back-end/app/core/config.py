from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient
from openai import OpenAI
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
QDRANT_API_KEY = os.getenv("API_KEY")
QDRANT_URL = os.getenv("URL")

qdrant_client = QdrantClient(
    url="https://270d1d27-de00-4512-aff8-fc2cee181307.europe-west3-0.gcp.cloud.qdrant.io", 
    api_key=QDRANT_API_KEY,
    port=443
)
openai_client = OpenAI(api_key=OPENAI_KEY)

