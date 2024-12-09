from pydantic import BaseModel
from typing import List, Optional


class QueryModel(BaseModel):
    query: str
    version: str

class SearchResult(BaseModel):
    docid: str
    title: Optional[str]
    description: Optional[str]
    address: Optional[str]
    score: float
    link: Optional[str]

class PaginationModel(BaseModel):
    prev: Optional[str]
    next: Optional[str]

class CombinedAPIResponse(BaseModel):
    elasticsearch_results: List[SearchResult]
    bm25_results: List[SearchResult]
    random_results: List[SearchResult]

class APIResponse(BaseModel):
    results: List[SearchResult]
    page: Optional[PaginationModel]

class SearchResponse(BaseModel):
    id: int
    docid: int
    score: float

class BaseSearchEngine():
    def __init__(self, path: str) -> None:
        pass

    def index(self):
        pass

    def search(self, query: str) -> list[SearchResponse]:
        pass

class ExperimentResponse(BaseModel):
    ndcg: float
    query: str