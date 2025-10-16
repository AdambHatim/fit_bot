from openai import OpenAI
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))



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

def specify_intent(openai_client, message):
    system_prompt = """Classify the user message into one of:
    - small_talk
    - fitness_query
    Return only the label."""
    
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message}
        ]
    )
    
    intent = completion.choices[0].message.content.strip().lower()
    return intent

def get_response(openai_client, prompt, rag_text):
    intent = specify_intent(openai_client, prompt)
    if intent == "small_talk":

        completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful chatbot that answer questions."},
            {"role": "user", "content": prompt},
        ]
    )
        response = completion.choices[0].message.content.strip()

        return response
    
    else:

        completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful fitness assistant. Here is some context you may use:\n{rag_text}"},
            {"role": "user", "content": prompt},
        ]
    )
        response = completion.choices[0].message.content.strip().lower()

        return response

        



