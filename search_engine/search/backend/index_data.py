from flask import Flask, jsonify, request
from flask_cors import CORS
import logging
import requests
import yaml
from elasticsearch import Elasticsearch, ConnectionError
import time

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to connect to Elasticsearch
def connect_elasticsearch():
    es = None
    max_attempts = 60  # Increase the number of attempts
    for attempt in range(max_attempts):
        try:
            es = Elasticsearch(
                [{'host': 'elasticsearch', 'port': 9200, 'scheme': 'http'}],
                timeout=30  # Increase timeout for each request
            )
            if es.ping():
                logger.info("Connected to Elasticsearch")
                return es
            else:
                logger.error("Elasticsearch ping failed")
        except ConnectionError:
            logger.error(f"Elasticsearch not ready, attempt {attempt+1}/{max_attempts}, retrying in 10 seconds...")
            time.sleep(10)  # Increase the interval between retries
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(10)
    raise Exception("Could not connect to Elasticsearch after several attempts")

es = connect_elasticsearch()

# GitHub raw URL for the latest version of nfdi4bioimage.yml
github_url = 'https://raw.githubusercontent.com/NFDI4BIOIMAGE/training/refs/heads/main/resources/nfdi4bioimage.yml'

# Function to download the latest version of the YAML file from GitHub
def download_yaml_file():
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise error if the download fails
        
        yaml_content = response.text
        logger.info("Downloaded the latest YAML file from GitHub")
        return yaml.safe_load(yaml_content)  # Parse YAML content
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading the YAML file: {e}")
        return None

# Function to delete the existing index
def delete_index(index_name):
    try:
        es.indices.delete(index=index_name, ignore=[400, 404])
        logger.info(f"Deleted existing index: {index_name}")
    except Exception as e:
        logger.error(f"Error deleting index {index_name}: {e}")

# Function to index the downloaded YAML file content into Elasticsearch
def index_yaml_files():
    try:
        # Download the latest YAML file from GitHub
        yaml_content = download_yaml_file()
        if yaml_content is None:
            raise Exception("Failed to download the YAML file from GitHub")

        # Define the index mapping for search-as-you-type
        mapping = {
            "mappings": {
                "properties": {
                    "name": {
                        "type": "search_as_you_type"  # Enables search-as-you-type on name
                    },
                    "description": {
                        "type": "search_as_you_type"  # Enables search-as-you-type on description
                    },
                    "tags": {"type": "text"},
                    "authors": {"type": "text"},
                    "type": {"type": "text"},
                    "license": {"type": "text"},
                    "url": {"type": "text"}
                }
            }
        }

        # Create the index with the new mapping
        es.indices.create(index='bioimage-training', body=mapping, ignore=400)

        # Get the 'resources' section from the YAML file
        data = yaml_content.get('resources', [])
        if isinstance(data, list):
            for item in data:
                try:
                    if isinstance(item, dict):
                        es.index(index='bioimage-training', body=item)
                        logger.info(f"Indexed item: {item}")
                    else:
                        logger.error(f"Item is not a dictionary: {item}")
                except Exception as e:
                    logger.error(f"Error indexing item: {item} - {e}")
        else:
            logger.error(f"Data is not a list: {data}")

        # Refresh the index after indexing is done
        es.indices.refresh(index='bioimage-training')

    except Exception as e:
        logger.error(f"Error indexing YAML files: {e}")


# Flask route to return materials using the Scroll API (more efficient for large datasets)
@app.route('/api/materials', methods=['GET'])
def get_materials():
    try:
        materials = []
        scroll_time = '2m'  # Time window for each scroll
        scroll_size = 1000  # Number of documents per scroll request

        # Initial request for the first scroll
        response = es.search(
            index='bioimage-training',
            scroll=scroll_time,
            size=scroll_size,
            body={"query": {"match_all": {}}}
        )

        # Get the scroll ID and first batch of results
        scroll_id = response['_scroll_id']
        materials += [doc['_source'] for doc in response['hits']['hits']]

        # Keep fetching while there are still results
        while len(response['hits']['hits']) > 0:
            response = es.scroll(scroll_id=scroll_id, scroll=scroll_time)
            scroll_id = response['_scroll_id']
            materials += [doc['_source'] for doc in response['hits']['hits']]

        return jsonify(materials)

    except Exception as e:
        logger.error(f"Error fetching data from Elasticsearch: {e}")
        return jsonify({"error": str(e)}), 500

# Flask route for search functionality
@app.route('/api/search', methods=['GET'])
def search():
    query = request.args.get('q', '')

    # Basic sanitation of the query string
    sanitized_query = query.replace('+', ' ').replace(':', '')

    try:
        # Use match_phrase to ensure the whole phrase is matched exactly
        es_response = es.search(
            index='bioimage-training',
            body={
                "query": {
                    "match_phrase": {  # Change from multi_match to match_phrase for exact matching
                        "name": sanitized_query  # Match against the name field exactly
                    }
                }
            },
            size=1000  # Retrieve up to 1000 results
        )
        return jsonify(es_response['hits']['hits'])
    except Exception as e:
        # Log the error for debugging
        print(f"Error searching in Elasticsearch: {e}")
        return jsonify({"error": str(e)}), 500

    
@app.route('/api/suggest', methods=['GET'])
def suggest():
    try:
        query = request.args.get('q', '')  # Get the partial search query
        es_response = es.search(
            index='bioimage-training',
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name", "description"],  # Search in name and description
                        "type": "bool_prefix"  # Allows matching as-you-type queries
                    }
                }
            }
        )
        suggestions = es_response['hits']['hits']
        return jsonify([suggestion['_source'] for suggestion in suggestions])
    except Exception as e:
        logger.error(f"Error fetching suggestions from Elasticsearch: {e}")
        return jsonify({"error": str(e)}), 500


# Main entry point
if __name__ == '__main__':
    delete_index('bioimage-training')  # Optionally delete the index to refresh data
    index_yaml_files()  # Index the latest data from the YAML file
    app.run(host='0.0.0.0', port=5000, debug=True)
