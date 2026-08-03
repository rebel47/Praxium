"""Documents, provenance-preserving chunking, and hybrid in-memory retrieval."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Any, Protocol

from pydantic import Field, model_validator

from praxium.core import ChunkId, DocumentId, FrameworkModel


class Document(FrameworkModel):
    id: DocumentId = Field(default_factory=DocumentId.new)
    tenant_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    collection: str = Field(default="default", min_length=1)
    content: str
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)


class Chunk(FrameworkModel):
    id: ChunkId
    document_id: DocumentId
    tenant_id: str
    namespace: str
    collection: str
    text: str
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    source_uri: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Chunk:
        if self.end < self.start:
            raise ValueError("chunk end must be greater than or equal to start")
        return self


class Citation(FrameworkModel):
    document_id: DocumentId
    chunk_id: ChunkId
    source_uri: str | None = None
    start: int
    end: int
    title: str | None = None


class RetrievalQuery(FrameworkModel):
    tenant_id: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    collection: str = Field(default="default", min_length=1)
    text: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(default=5, ge=1, le=100)
    dense_weight: float = Field(default=0.5, ge=0, le=1)


class RetrievedChunk(FrameworkModel):
    chunk: Chunk
    score: float = Field(ge=0)
    lexical_score: float = Field(ge=0)
    dense_score: float = Field(ge=0)
    citation: Citation


class IndexResult(FrameworkModel):
    document_id: DocumentId
    version: int
    chunks_indexed: int = Field(ge=0)
    replaced_chunks: int = Field(ge=0)


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class TextChunker(FrameworkModel):
    """Character chunker that prefers whitespace while preserving exact offsets."""

    chunk_size: int = Field(default=1000, ge=50)
    overlap: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> TextChunker:
        if self.overlap >= self.chunk_size:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content:
            return []
        chunks: list[Chunk] = []
        start = 0
        index = 0
        while start < len(document.content):
            hard_end = min(start + self.chunk_size, len(document.content))
            end = hard_end
            if hard_end < len(document.content):
                boundary = document.content.rfind(" ", start + self.chunk_size // 2, hard_end)
                if boundary > start:
                    end = boundary
            text = document.content[start:end]
            digest = hashlib.sha256(
                f"{document.id}:{document.version}:{index}:{start}:{end}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    id=ChunkId(root=f"chkpart_{digest}"),
                    document_id=document.id,
                    tenant_id=document.tenant_id,
                    namespace=document.namespace,
                    collection=document.collection,
                    text=text,
                    index=index,
                    start=start,
                    end=end,
                    source_uri=document.source_uri,
                    metadata=document.metadata,
                )
            )
            if end >= len(document.content):
                break
            start = max(end - self.overlap, start + 1)
            index += 1
        return chunks


class InMemoryRagIndex:
    """Tenant-isolated, idempotent hybrid retrieval reference backend."""

    def __init__(self) -> None:
        self._documents: dict[DocumentId, Document] = {}
        self._chunks: dict[ChunkId, Chunk] = {}
        self._lock = asyncio.Lock()

    async def index(
        self,
        document: Document,
        *,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
    ) -> IndexResult:
        chunks = (chunker or TextChunker()).chunk(document)
        if embedder is not None and chunks:
            embeddings = await embedder.embed([chunk.text for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise ValueError("embedder returned a different number of embeddings than inputs")
            chunks = [
                chunk.model_copy(update={"embedding": embedding})
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]
        async with self._lock:
            existing = self._documents.get(document.id)
            if existing and existing.tenant_id != document.tenant_id:
                raise ValueError("document ID already belongs to another tenant")
            old_ids = [
                chunk_id
                for chunk_id, chunk in self._chunks.items()
                if chunk.document_id == document.id
            ]
            for chunk_id in old_ids:
                del self._chunks[chunk_id]
            self._documents[document.id] = document.model_copy(deep=True)
            for chunk in chunks:
                self._chunks[chunk.id] = chunk.model_copy(deep=True)
        return IndexResult(
            document_id=document.id,
            version=document.version,
            chunks_indexed=len(chunks),
            replaced_chunks=len(old_ids),
        )

    async def query(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        async with self._lock:
            chunks = [chunk.model_copy(deep=True) for chunk in self._chunks.values()]
        query_terms = _terms(query.text)
        results: list[RetrievedChunk] = []
        for chunk in chunks:
            if (
                chunk.tenant_id != query.tenant_id
                or chunk.namespace != query.namespace
                or chunk.collection != query.collection
            ):
                continue
            if any(chunk.metadata.get(key) != value for key, value in query.metadata.items()):
                continue
            lexical = _jaccard(query_terms, _terms(chunk.text))
            dense = _cosine(query.embedding, chunk.embedding)
            weight = (
                query.dense_weight
                if query.embedding is not None and chunk.embedding is not None
                else 0
            )
            score = (weight * dense) + ((1 - weight) * lexical)
            if score <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    lexical_score=lexical,
                    dense_score=dense,
                    citation=Citation(
                        document_id=chunk.document_id,
                        chunk_id=chunk.id,
                        source_uri=chunk.source_uri,
                        start=chunk.start,
                        end=chunk.end,
                        title=chunk.metadata.get("title"),
                    ),
                )
            )
        results.sort(key=lambda result: (-result.score, str(result.chunk.id)))
        return results[: query.top_k]

    async def delete(self, tenant_id: str, document_id: DocumentId) -> bool:
        async with self._lock:
            document = self._documents.get(document_id)
            if document is None or document.tenant_id != tenant_id:
                return False
            del self._documents[document_id]
            for chunk_id in [
                value.id for value in self._chunks.values() if value.document_id == document_id
            ]:
                del self._chunks[chunk_id]
            return True


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.casefold()))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _cosine(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None or len(left) != len(right) or not left:
        return 0.0
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm), 0.0)
