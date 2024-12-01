from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from threading import Timer
import math
import json
import os
from models3 import QueryModel, APIResponse, PaginationModel, SearchResult
from pipeline import initialize
from elasticsearch import Elasticsearch, exceptions




# Some global variables
algorithm = initialize()
pagination_cache = {}
timer_mgr = {}
docid_to_path_mapping = {}

# Configurations
PAGE_SIZE = 10
CACHE_TIME = 3600

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")

# Initialize the Elasticsearch client
ELASTIC_CLOUD_ID = "My_deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDhjZTQyMzBhNTM3MzQzN2ViMGI2NmQ3NWE0ZmIzY2JmJDkyM2FkNDkxYzc4ZDQ5YWI5YWM0ZmU0YWQzMDMwYTYy"
ELASTIC_API_KEY = "WjE2NlpwTUJxYW9xczQySU5IQ2c6X2ZyTWRId2hTY3Fzd20tRElBamxvZw=="

client = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)





def load_document_paths():
    global docid_to_path_mapping
    try:
        with open('docs_to_paths.jsonl', 'r') as file:
            for line in file:
                if line.strip():
                    doc = json.loads(line)
                    if 'docid' in doc and 'filename' in doc:
                        docid_to_path_mapping[str(doc['docid'])] = doc['filename']
    except Exception as e:
        print(f"Error loading document paths: {e}")

def lookup_document_details(docid):
    global docid_to_path_mapping
    if docid not in docid_to_path_mapping:
        return {'title': 'No Title', 'description': 'No Description'}

    file_path = docid_to_path_mapping[docid]
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return {
                'title': data.get('Title', 'No Title'),
                'description': data.get('Description', 'No Description'),
                'address': data.get('Address', 'No Address'),
                'link': data['Menu']['source']
            }
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return {'title': 'No Title', 'description': 'No Description'}


@app.on_event("startup")
async def startup_event():
    load_document_paths()



# Cache deletion function
def delete_from_cache(query):
    global pagination_cache, timer_mgr
    if query in pagination_cache:
        del pagination_cache[query]
        del timer_mgr[query]
        print(f"Cache for query '{query}' has been deleted.")

# Home route to serve the combined HTML page
@app.get('/', response_class=HTMLResponse)
async def home():
    with open('./web/home3.html') as f:
        return f.read()

# Elasticsearch search endpoint
@app.post('/search_elasticsearch', response_model=APIResponse)
async def search_elasticsearch(body: QueryModel) -> APIResponse:
    request_query = body.query
    global pagination_cache, timer_mgr

    # Perform search in Elasticsearch
    try:
        response = client.search(
            index="elser-example-restaurants",
            size=PAGE_SIZE,  # Limit to top PAGE_SIZE results
            query={
                "text_expansion": {
                    "description_embedding": {
                        "model_id": ".elser_model_2",
                        "model_text": request_query,
                    }
                }
            },
        )

        results = [
            SearchResult(
                docid=hit["_id"],
                title=hit["_source"].get("Title"),
                description=hit["_source"].get("Description"),
                score=hit["_score"]
            ) for hit in response["hits"]["hits"]
        ]
    except exceptions.ElasticsearchException as e:
        print(f"Search Query: {request_query}, Error: {str(e)}")
        return APIResponse(results=[], page=None)

    if not results:
        return APIResponse(results=[], page=None)  # Handle no results case

    # Cache results
    cache_key = f"elasticsearch_{request_query}"
    pagination_cache[cache_key] = results

    # Cache expiration handling
    t = Timer(CACHE_TIME, delete_from_cache, [cache_key])
    timer_mgr[cache_key] = t
    t.start()

    return APIResponse(results=results, page=None)

# BM25 search endpoint
@app.post('/search_bm25', response_model=APIResponse)
async def search_bm25(body: QueryModel) -> APIResponse:
    request_query = body.query
    response = algorithm.search(request_query)
    global pagination_cache, timer_mgr

    print(f"Search Query: {request_query}")
    print(f"Search Results: {response}")

    if not response:
        print(f"No results found for query: {request_query}")
        return APIResponse(results=[], page=None)  # Handle no results case

    # Transform the response results into SearchResult instances with additional lookup
    results = [
        SearchResult(
            docid=str(r.docid),  # Ensure docid is a string
            **lookup_document_details(str(r.docid)),  # Look up and add title and description
            score=r.score
        ) for r in response[:PAGE_SIZE]  # Limit to top PAGE_SIZE results
    ]

    # Cache results
    cache_key = f"bm25_{request_query}"
    pagination_cache[cache_key] = results

    # Cache expiration handling
    t = Timer(CACHE_TIME, delete_from_cache, [cache_key])
    timer_mgr[cache_key] = t
    t.start()

    return APIResponse(results=results, page=None)


@app.on_event('shutdown')
def timer_shutdown():
    print("Shutting down and canceling timers.")
    [timer_mgr[key].cancel() for key in timer_mgr]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
