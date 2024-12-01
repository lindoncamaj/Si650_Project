from pydantic import BaseModel
from typing import List, Optional

# Query Model
class QueryModel(BaseModel):
    query: str

# Full result model for API response
class SearchResult(BaseModel):
    docid: str
    title: Optional[str]
    description: Optional[str]
    score: float

# Model for pagination information
class PaginationModel(BaseModel):
    prev: str
    next: str

# Full API response model
class APIResponse(BaseModel):
    results: List[SearchResult]
    page: Optional[PaginationModel]

# Model for experimental response
class ExperimentResponse(BaseModel):
    ndcg: float
    query: str

# Base class for search engines
class BaseSearchEngine():
    def __init__(self, path: str) -> None:
        pass

    def index(self):
        pass

    def search(self, query: str) -> List[SearchResult]:
        pass
