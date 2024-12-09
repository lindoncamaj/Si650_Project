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