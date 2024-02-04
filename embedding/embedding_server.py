from flask import Flask
from sentence_transformers import SentenceTransformer
from flask import request
import logging

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
        "vector": vectors.tolist()
    }


if __name__ == '__main__':
    logging.info("Starting server")
    app.run()
