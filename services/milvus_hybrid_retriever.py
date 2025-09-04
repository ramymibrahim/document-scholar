import asyncio
from itertools import chain
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.runnables import Runnable


class HybridRetrieverWithScores(Runnable):
    def __init__(
        self, vectorstore, k: int = 5, search_kwargs: Dict[str, Any] | None = None
    ):
        self.store = vectorstore
        self.k = k
        self.search_kwargs = search_kwargs or {}

    # sync
    def invoke(self, query: str, **kwargs) -> List[Document]:
        pairs = self.store.similarity_search_with_score(
            query, k=self.k, **self.search_kwargs
        )
        docs: List[Document] = []
        for doc, score in pairs:
            doc.metadata = {**doc.metadata, "score": float(score)}
            docs.append(doc)
        return docs

    # async
    async def ainvoke(self, query: str, **kwargs) -> List[Document]:
        pairs = await self.store.asimilarity_search_with_score(
            query, k=self.k, **self.search_kwargs
        )
        docs: List[Document] = []
        for doc, score in pairs:
            doc.metadata = {**doc.metadata, "score": float(score)}
            docs.append(doc)
        return docs

    # batches
    def batch(self, queries: List[str], **kwargs) -> List[List[Document]]:
        return [self.invoke(q, **kwargs) for q in queries]

    async def abatch(self, queries: List[str], **kwargs) -> List[List[Document]]:
        return await asyncio.gather(*[self.ainvoke(q, **kwargs) for q in queries])
