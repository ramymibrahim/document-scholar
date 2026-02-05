from __future__ import annotations
import os
from typing import List, Tuple
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

import uuid
import re
import itertools

from langchain_milvus import BM25BuiltInFunction, Milvus

from model.domain.core import UserFilter
from services.db import Db
import datetime
from itertools import chain
import json
from langchain_core.documents import Document
from langchain_core.runnables import chain

from services.milvus_hybrid_retriever import HybridRetrieverWithScores

DATE_FMT = "%Y-%m-%d"


# ---------------------------
# Helpers
# ---------------------------
def _parse_date(value: str):
    try:
        return datetime.datetime.strptime(value, DATE_FMT).date()
    except (TypeError, ValueError):
        return None


class VectorDbService:
    def __init__(
        self,
        mulvis_db_host,
        mulvis_db_port,
        mulvis_db,
        embedding_model,
        db: Db,
        categories,
    ):
        URI = f"http://{mulvis_db_host}:{mulvis_db_port}"
        self.vectorstore = Milvus(
            embedding_function=embedding_model,
            connection_args={"uri": URI, "token": "root:Milvus", "db_name": mulvis_db},
            consistency_level="Strong",
            drop_old=False,
            builtin_function=BM25BuiltInFunction(),
            vector_field=["dense", "sparse"],
        )
        self.db = db
        self.categories = categories

    ### File Management ###
    def save_meta_in_sql(self, meta):
        meta["id"] = meta.pop("file_id")
        cols = []
        for key in meta:
            cols.append(key)
        self.db.execute(
            f"""INSERT INTO files ({','.join(cols)}) 
                        values ({','.join([f":{col}" for col in cols])})""",
            meta,
        )

    def add_file(self, file: dict):
        file_path = file["file_path"]
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PDFPlumberLoader(file_path)
        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)
        elif ext == ".doc":
            loader = UnstructuredWordDocumentLoader(file_path)
        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError("Unsupported file type: " + ext)

        docs = loader.load()

        docs = [
            d
            for d in docs
            if isinstance(d, Document) and d.page_content and d.page_content.strip()
        ]
        if not docs:
            return False
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=90)
        chunks = splitter.split_documents(docs)

        chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

        if not chunks:
            return False

        original_file_name = file["original_file_name"]
        meta = {}

        for md in file:
            if md != "file_path":
                # Milvus varchar fields reject None; default to empty string
                meta[md] = file[md] if file[md] is not None else ""

        # Standard loader fields with defaults so every document type
        # produces the same Milvus schema regardless of loader used.
        _LOADER_DEFAULTS = {"page": 0, "total_pages": 1}

        for idx, doc in enumerate(chunks):
            text = doc.page_content
            text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
            doc.page_content = text
            loader_meta = {
                k: doc.metadata.get(k, default)
                for k, default in _LOADER_DEFAULTS.items()
            }
            doc.metadata = {**loader_meta, **meta, "chunk_index": idx}
            if idx == 0 and original_file_name:
                doc.page_content = f"Source: {original_file_name}\n\n{doc.page_content}"

        uuids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        self.vectorstore.add_documents(chunks, ids=uuids)

        self.save_meta_in_sql(meta)
        return True

    def delete_file(self, file_id):
        self.vectorstore.delete(expr=f"file_id=='{file_id}'")
        self.db.execute("DELETE FROM files WHERE id=:id", {"id": file_id})

    def get_file_content(self, file_id):

        results = self.vectorstore.similarity_search(
            query="", expr=f"file_id=='{file_id}'", k=200, fetch_k=200
        )
        documents = [doc.model_dump() for doc in results]
        documents = sorted(documents, key=lambda c: c["metadata"]["chunk_index"])
        return documents

    def get_all_files(self, user_filter: UserFilter, page, size, sort_field, sort_dir):
        where, sql_params = self._prepare_user_filter(user_filter)
        offset = (page - 1) * size
        sql = f"""SELECT * FROM files
        WHERE {where}
        ORDER  BY {sort_field} {sort_dir}
        LIMIT  :size
        OFFSET :offset
        """

        count_sql = f"""SELECT COUNT(0) as cnt FROM files f WHERE {where}"""
        count = self.db.get_row_or_default(count_sql, sql_params)

        sql_params["offset"] = offset
        sql_params["size"] = size
        rows = self.db.get_rows(sql, sql_params)

        return rows, count["cnt"]

    def get_file_data(self, file_id):
        return self.db.get_row_or_default(
            "SELECT * FROM files WHERE id=:id", {"id": file_id}
        )

    ### Searching ###
    def get_file_ids(self, filter: UserFilter):
        if not filter:
            return None
        where, sql_params = self._prepare_user_filter(filter)
        sql = f"""SELECT id FROM files WHERE {where}
        """
        rows = self.db.get_rows(sql, sql_params)
        return [row["id"] for row in rows] or None

    async def get_documents(
        self, queries, file_ids, k=5, fetch_k=30, alpha=0.7, beta=0.3
    ):
        search_kwargs = {
            "fetch_k": fetch_k,
            "ranker_type": "weighted",
            "ranker_params": {
                "alpha": alpha,
                "beta": beta,
            },
        }
        if file_ids:
            quoted = ",".join(f"'{id}'" for id in file_ids)
            expr = f"file_id in [{quoted}]"
            search_kwargs["expr"] = expr

        retriever = HybridRetrieverWithScores(
            self.vectorstore, k=k, search_kwargs=search_kwargs
        )

        lists = await retriever.abatch(queries)
        candidates = list(itertools.chain.from_iterable(lists))

        seen, deduped = set(), []
        for d in candidates:
            key = d.metadata.get("pk") or (
                d.page_content,
                frozenset(d.metadata.items()),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)

        scored = sorted(
            deduped,
            key=lambda d: abs(d.metadata.get("score", 0.0)),
            reverse=True,
        )

        top_docs = scored[:k]
        return top_docs

    def get_context(self, documents: list[Document]):
        parts = []
        for i, d in enumerate(documents, start=1):
            parts.append(f"{d.page_content}")
        return "\n\n".join(parts)

    ### Helper Functions ###
    def _prepare_user_filter(self, filter: UserFilter):
        created_from = (
            str(_parse_date(filter.created_from)) if filter.created_from else None
        )
        updated_from = (
            str(_parse_date(filter.updated_from)) if filter.updated_from else None
        )
        created_to = (
            str(_parse_date(filter.created_to) + datetime.timedelta(days=1))
            if filter.created_to
            else None
        )
        updated_to = (
            str(_parse_date(filter.updated_to) + datetime.timedelta(days=1))
            if filter.updated_to
            else None
        )

        where = """1=1
            AND (:file IS NULL OR original_file_name LIKE '%' || :file || '%')
            AND (:author IS NULL OR author LIKE '%' || :author || '%')
            AND (:folder IS NULL OR folder LIKE :folder || '%')    
            AND (:created_from IS NULL OR created_at>=:created_from)
            AND (:created_to IS NULL OR created_at<:created_to)    
            AND (:updated_from IS NULL OR updated_at>=:updated_from)
            AND (:updated_to IS NULL OR updated_at<:updated_to)
            """
        sql_params = {
            "file": filter.file,
            "author": filter.author,
            "folder": filter.folder,
            "created_from": created_from,
            "created_to": created_to,
            "updated_from": updated_from,
            "updated_to": updated_to,
        }
        if filter.category_ids:
            for category in filter.category_ids:
                if category.categories:
                    vals = [f"'{val}'" for val in category.categories]
                    where += f"\nAND {category.id} in ({','.join(vals)})"
        return where, sql_params

    ### Chain
    # Create a Runnable that returns Documents with score attached to metadata
    @chain
    async def retriever_with_scores(
        self, query: str, k: int = 5, **search_kwargs
    ) -> List[Document]:
        pairs: List[Tuple[Document, float]] = (
            await self.vectorstore.asimilarity_search_with_relevance_scores(
                query, k=k, **search_kwargs
            )
        )
        docs: List[Document] = []
        for doc, score in pairs:
            doc.metadata = {**doc.metadata, "score": float(score)}
            docs.append(doc)
        return docs
