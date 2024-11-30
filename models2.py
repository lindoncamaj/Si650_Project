from pydantic import BaseModel
from typing import List, Optional

class QueryModel(BaseModel):
    query:str

class SearchResult(BaseModel):
    docid: str
    title: Optional[str]
    description: Optional[str]
    score: float

class PaginationModel(BaseModel):
    prev: str
    next: str

class APIResponse(BaseModel):
    results: List[SearchResult]
    page: PaginationModel | None
