"""Vector store wrapper for Pinecone integration with LangChain."""

from functools import lru_cache
from typing import List

from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import get_settings


@lru_cache(maxsize=1)
def _get_vector_store() -> PineconeVectorStore:
    """Create and cache a PineconeVectorStore instance."""
    settings = get_settings()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    
    index = pc.Index(
        name=settings.pinecone_index_name,
        host=settings.pinecone_host,   # 👈 THIS IS THE NEW LINE
    )

    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model_name,
        api_key=settings.openai_api_key,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


def get_retriever(k: int | None = None):
    """Get a retriever for querying the vector store."""
    settings = get_settings()
    if k is None:
        k = settings.retrieval_k

    vector_store = _get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})


def retrieve(query: str, k: int | None = None) -> List[Document]:
    """Retrieve documents from Pinecone for a given query."""
    retriever = get_retriever(k=k)
    return retriever.invoke(query)


def index_documents(docs: List[Document]) -> int:
    """
    Index a list of already-loaded Document objects into Pinecone.

    Args:
        docs: Documents produced by a loader (e.g., PyPDFLoader).

    Returns:
        Number of document chunks indexed.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = text_splitter.split_documents(docs)

    vector_store = _get_vector_store()
    vector_store.add_documents(chunks)

    return len(chunks)
