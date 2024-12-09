from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from threading import Timer
import math
import json
from typing import List, Dict, Union
from elasticsearch import Elasticsearch, exceptions
from pipeline import initialize
import random
from models import QueryModel, APIResponse, CombinedAPIResponse, PaginationModel, SearchResult

# Initialize Elasticsearch
ELASTIC_CLOUD_ID = "dfc561b4569d4a918cff6109db12bed5:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDM5NTUzYjU0MzA4YjQxY2E4ZTNlM2M1NmUyMDdmYjVhJGZkMmIwOWJhNzBiZTQ4NGZhMWNlM2IxOWFhMjI1N2E2"
ELASTIC_API_KEY = "Y0RzOHFaTUJqSnEtWUVUbWpEMVA6ZmkxYThHdWFSYm04ek5aZ24yN2dIQQ=="
client = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)

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

# Load document paths
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
                'link': data.get('Menu', {}).get('link', '')
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

@app.get('/', response_class=HTMLResponse)
async def home():
    with open('web/home.html') as f:
        return f.read()

@app.post('/search', response_model=Union[APIResponse, CombinedAPIResponse])
async def do_search(body: QueryModel) -> Union[APIResponse, CombinedAPIResponse]:
    request_query = body.query
    version = body.version
    print(f"Search Query: {request_query}, Version: {version}")

    if version == "elasticsearch":
        response = await search_elasticsearch(request_query)
    elif version == "bm25":
        response = await search_bm25(request_query)
    elif version == "combined":
        response = await search_combined(request_query)
        return response
    else:
        return APIResponse(results=[], page=None)

    global pagination_cache, timer_mgr
    if not response:
        return APIResponse(results=[], page=None)  # Handle no results case

    # Store search results and calculate max_page
    pagination_cache[request_query] = response
    pagination_cache[f'{request_query}_max_page'] = math.floor((len(response) - 1) / PAGE_SIZE)
    max_pages = pagination_cache[f'{request_query}_max_page']
    print(f"Max Page for query '{request_query}': {max_pages}")

    # Cache expiration handling
    t = Timer(CACHE_TIME, delete_from_cache, [request_query])
    timer_mgr[request_query] = t
    t.start()

    # Return first page of results
    next_page = f'/cache/{request_query}/page/1' if max_pages > 0 else ""
    return APIResponse(
        results=response[:PAGE_SIZE],
        page=PaginationModel(
            prev="",
            next=next_page
        )
    )

async def search_elasticsearch(query: str) -> List[SearchResult]:
    try:
        response = client.search(
            index="elser-example-restaurants",
            size=10,
            query={
                "text_expansion": {
                    "description_embedding": {
                        "model_id": ".elser_model_2",
                        "model_text": query,
                    }
                }
            },
        )
        return [
            SearchResult(
                docid=hit["_id"],
                title=hit["_source"].get("Title"),
                description=hit["_source"].get("Description"),
                address=hit["_source"].get("Address"),
                link=f'https://{hit["_source"]["Menu"]["source"]}',
                score=hit["_score"]
            ) for hit in response["hits"]["hits"]
        ]
    except exceptions.ElasticsearchException as e:
        print(f"Search Query:{query}, Error: {str(e)}")
        return []

async def search_bm25(query: str) -> List[SearchResult]:
    response = algorithm.search(query)
    return [
        SearchResult(
            docid=str(res.docid),
            **lookup_document_details(str(res.docid)),
            score=res.score
        ) for res in response[:10]]

async def get_random_documents(num_docs: int = 10) -> List[SearchResult]:
    all_doc_ids = list(docid_to_path_mapping.keys())

    # Ensure there are enough documents to sample from
    if len(all_doc_ids) < num_docs:
        num_docs = len(all_doc_ids)
    random_doc_ids = random.sample(all_doc_ids, num_docs)

    random_documents = []

    for docid in random_doc_ids:
        details = lookup_document_details(docid)
        random_documents.append(
            SearchResult(
                docid=docid,
                title=details.get('title', 'No Title'),
                description=details.get('description', 'No Description'),
                address=details.get('address', 'No Address'),  # Ensure address is provided
                score=0,  # Random documents may not have a relevance score
                link=details.get('link', '')
            )
        )
    return random_documents


async def search_combined(query: str) -> Dict[str, List[SearchResult]]:
    bm25_results = await search_bm25(query)
    elastic_results = await search_elasticsearch(query)
    random_results = await get_random_documents()

    return CombinedAPIResponse(
        elasticsearch_results=elastic_results,
        bm25_results=bm25_results,
        random_results=random_results
    )

@app.get('/cache/{query}/page/{page}')
async def get_cache(query: str, page: int) -> APIResponse:
    if query in pagination_cache:
        max_page = pagination_cache[f'{query}_max_page']

        # Boundary checks
        if page < 0:
            page = 0
        prev_page = f'/cache/{query}/page/{page - 1}' if page > 0 else ""
        next_page = f'/cache/{query}/page/{page + 1}' if page < max_page else ""

        results = pagination_cache[query][page * PAGE_SIZE:(page + 1) * PAGE_SIZE]

        print(f"Cache hit for query: {query}, page: {page}, Results: {results}")

        return APIResponse(
            results=results,
            page=PaginationModel(
                prev=prev_page,
                next=next_page
            )
        )
    else:
        print(f"Cache miss for query: {query}. Performing a new search.")
        return await do_search(QueryModel(query=query, version="elasticsearch"))

@app.on_event('shutdown')
def timer_shutdown():
    print("Shutting down and canceling timers.")
    [timer_mgr[key].cancel() for key in timer_mgr]

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)