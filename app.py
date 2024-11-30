from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from threading import Timer
import math

from models import QueryModel, APIResponse, PaginationModel
from pipeline import initialize

# Some global variables
algorithm = initialize()
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
    with open('./web/home.html') as f:
        return f.read()

@app.post('/search')
async def doSearch(body: QueryModel) -> APIResponse:
    request_query = body.query
    response = algorithm.search(request_query)
    global pagination_cache, timer_mgr

    # Debug information
    print(f"Search Query: {request_query}")
    print(f"Search Results: {response}")

    if not response:
        print(f"No results found for query: {request_query}")
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

@app.get('/cache/{query}/page/{page}')
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
