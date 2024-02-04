from flask import Flask
from sentence_transformers import SentenceTransformer
from flask import request
import logging
import requests

OLLAMA_API_EMBEDDINGS = "http://localhost:11434/api/embeddings"

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

model: SentenceTransformer = SentenceTransformer('all-MiniLM-L6-v2')


@app.route('/')
def hello():
    return 'Hello, World!'


@app.route('/embeddings', methods=['POST'])
def embeddings():
    request_data = request.get_json()
    logging.info(request_data)
    text_value = request_data['text']
    vectors = model.encode(text_value)
    logging.info("Embeddings %s", vectors.size)

    return {
        "text": text_value,
        "length": vectors.size,
        "vector": vectors.tolist()
    }


@app.route('/ollama_embeddings', methods=['POST'])
def ollama_embeddings():
    request_data = request.get_json()
    logging.info(request_data)
    text_value = request_data['text']
    model_name = request_data['model']
    payload = {
        "model": model_name,
        "prompt": text_value
    }
    reply = requests.post(OLLAMA_API_EMBEDDINGS, json=payload)
    reply_json = reply.json()
    vectors = reply_json['embedding']

    return {
        "text": text_value,
        "length": len(vectors),
        "vector": vectors
    }


if __name__ == '__main__':
    logging.info("Starting server")
    app.run()
