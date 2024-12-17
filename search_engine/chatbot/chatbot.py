from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch, ConnectionError
from llm_utilities import LLMUtilities
import logging
import platform
import time
import os

# Flask app setup
app = Flask(__name__)
CORS(app)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware Information (local laptop or HPC reference; purely informational)
SYSTEM_INFO = {
    "Machine": platform.node(),
    "Processor": platform.processor(),
    "Local GPU": "NVIDIA RTX 500 Ada Generation Laptop GPU", 
    "NOTE": "Inference is delegated to the HPC KISSKI LLM service."
}
logger.info(f"System Info: {SYSTEM_INFO}")

# Function to connect to Elasticsearch with retry logic
def connect_elasticsearch():
    """
    Connects to Elasticsearch with retry logic.
    Returns:
        Elasticsearch instance if connection is successful, otherwise raises an exception.
    """
    es = None
    max_attempts = 120  # Increase max attempts to 120 (20 minutes)
    es_host = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
    es_port = os.getenv("ELASTICSEARCH_PORT", "9200")

    # Convert es_port to integer
    try:
        es_port = int(es_port)
    except ValueError:
        logger.error(f"ELASTICSEARCH_PORT is not a valid integer: {es_port}")
        raise

    for attempt in range(max_attempts):
        try:
            es = Elasticsearch(
                [{"host": es_host, "port": es_port, "scheme": "http"}],
                request_timeout=30
            )
            if es.ping():
                logger.info("Connected to Elasticsearch")
                return es
            else:
                logger.error("Elasticsearch ping failed")
        except ConnectionError:
            logger.warning(f"Elasticsearch not ready, attempt {attempt + 1}/{max_attempts}, retrying in 15 seconds...")
            time.sleep(15)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(15)
    raise Exception("Could not connect to Elasticsearch after several attempts")

# Connect to Elasticsearch
es = connect_elasticsearch()

# Initialize LLM Utilities with environment-driven model name
# Default to "meta-llama/Meta-Llama-3.1-70B-Instruct" unless otherwise specified
model_name = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3.1-70B-Instruct")

llm_util = LLMUtilities(model_name=model_name)

# Elasticsearch-based document retrieval
def retrieve_documents(query, top_k=3):
    """
    Retrieves relevant documents from Elasticsearch based on the query.
    Args:
        query (str): The search query.
        top_k (int): Number of top documents to retrieve.
    Returns:
        list: A list of retrieved documents.
    """
    try:
        response = es.search(
            index="bioimage-training",
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name^3", "description", "tags", "authors", "type", "license"],
                        "type": "best_fields",
                    }
                }
            },
            size=top_k,
        )
        documents = []
        for hit in response["hits"]["hits"]:
            source = hit["_source"]
            documents.append({
                "name": source.get("name", "Unnamed"),
                "description": source.get("description", "No description available"),
                "url": source.get("url", ""),
            })
        return documents
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        return []

# RAG-based response generation (with remote HPC LLM call)
def generate_response(query, documents):
    """
    Generates a response using retrieved documents and the query.
    Args:
        query (str): The search query.
        documents (list): List of retrieved documents.
    Returns:
        str: Generated response from the KISSKI LLM service.
    """
    context = "\n".join(
        [f"- {doc['name']}: {doc['description']} (URL: {doc['url']})" for doc in documents]
    )
    prompt = f"""
Based on the following documents, answer the user's question concisely and include relevant links.

## Documents
{context}

## Question
{query}
"""
    return llm_util.generate_response(prompt)

# Chatbot API endpoint
@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Chat endpoint to process user queries and generate responses.
    """
    user_query = request.json.get("query", "")
    if not user_query:
        return jsonify({"error": "Query cannot be empty"}), 400

    # Retrieve relevant documents
    documents = retrieve_documents(user_query)

    if not documents:
        return jsonify({"response": "No relevant documents found.", "sources": []})

    # Generate chatbot response
    reply = generate_response(user_query, documents)

    return jsonify({"response": reply, "sources": documents})

# Main entry point
if __name__ == "__main__":
    logger.info("Starting chatbot, using HPC KISSKI LLM endpoint for GPU inference...")
    app.run(host="0.0.0.0", port=5000, debug=True)
