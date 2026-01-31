from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fer2013_dataset.api_fetcher import get_joke, get_quote

# Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

intent_labels = [
    "tell a joke to cheer me up",
    "give me an inspirational quote"
]

intent_embeddings = model.encode(intent_labels)

def get_transformer_response(emotion):
    if emotion == "Sad":
        query = "I am feeling sad and need something funny"
    else:
        query = "I feel okay and want inspiration"

    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, intent_embeddings)
    intent = intent_labels[similarities.argmax()]

    if "joke" in intent:
        return get_joke()
    else:
        return get_quote()
