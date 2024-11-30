from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from elasticsearch import Elasticsearch, exceptions
from threading import Timer
import math
from models2 import QueryModel, APIResponse, SearchResult, PaginationModel


# Initialize the Elasticsearch client
ELASTIC_CLOUD_ID = "My_deployment:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJDhjZTQyMzBhNTM3MzQzN2ViMGI2NmQ3NWE0ZmIzY2JmJDkyM2FkNDkxYzc4ZDQ5YWI5YWM0ZmU0YWQzMDMwYTYy"
ELASTIC_API_KEY = "WjE2NlpwTUJxYW9xczQySU5IQ2c6X2ZyTWRId2hTY3Fzd20tRElBamxvZw=="

# Global variables
client = Elasticsearch(
    cloud_id=ELASTIC_CLOUD_ID,
    api_key=ELASTIC_API_KEY,
)
pagination_cache = {}
timer_mgr = {}

# Configurations
PAGE_SIZE = 10
CACHE_TIME = 3600

# Initialize FastAPI
app = FastAPI()
app.mount("/static", StaticFiles(directory="."), name="static")


# Cache deletion function
def delete_from_cache(query):
    global pagination_cache, timer_mgr
    if query in pagination_cache:
        del pagination_cache[query]
        del timer_mgr[query]
        print(f"Cache for query '{query}' has been deleted.")


@app.get('/', response_class=HTMLResponse)
async def home():
    with open('./web/home2.html') as f:
        return f.read()


@app.post('/search', response_model=APIResponse)
async def doSearch(body: QueryModel) -> APIResponse:
    request_query = body.query
    global pagination_cache, timer_mgr

    # Perform search in Elasticsearch
    try:
        response = client.search(
            index="elser-example-restaurants",
            size=20,
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
        print(f"Search Query:{request_query}, Error: {str(e)}")
        return APIResponse(results=[], page=None)



    if not results:
        return APIResponse(results=[], page=None)  # Handle no results case

    # Store search results and calculate max_page
    pagination_cache[request_query] = results
    pagination_cache[f'{request_query}_max_page'] = math.floor((len(results) - 1) / PAGE_SIZE)
    max_pages = pagination_cache[f'{request_query}_max_page']
    print(f"Max Page for query '{request_query}': {max_pages}")

    # Cache expiration handling
    t = Timer(CACHE_TIME, delete_from_cache, [request_query])
    timer_mgr[request_query] = t
    t.start()

    # Return first page of results
    next_page = f'/cache/{request_query}/page/1' if max_pages > 0 else ""
    return APIResponse(
        results=results[:PAGE_SIZE],
        page=PaginationModel(
            prev="",
            next=next_page
        )
    )


@app.get('/cache/{query}/page/{page}', response_model=APIResponse)
async def getCache(query: str, page: int) -> APIResponse:
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
        return await doSearch(QueryModel(query=query))


@app.on_event('shutdown')
def timer_shutdown():
    print("Shutting down and canceling timers.")
    [timer_mgr[key].cancel() for key in timer_mgr]