import json

from pinecone import ServerlessSpec, Pinecone
from typing import List
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter

from utils.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    logger
)

##########################################
# Initialize Pinecone
##########################################
pc = Pinecone(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

def ensure_pinecone_index_exists(index_name: str, dim: int = 1536):
    # List existing Pinecone indexes
    index_list_response = pc.list_indexes()
    index_names = [index_info['name'] for index_info in index_list_response.get('indexes', [])]
    logger.info(f"Available indexes before creation: {index_names}")
    # Create Pinecone index if it doesn't exist
    if index_name not in index_names:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info(f"Created new Pinecone index: {index_name}")
    else:
        logger.info(f"Using existing Pinecone index: {index_name}")


def create_or_load_vector_store_index(index_name: str) -> VectorStoreIndex:
    """Create or load a Pinecone-based VectorStoreIndex."""
    ensure_pinecone_index_exists(index_name)
    pindex = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pindex)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create an empty index if none exists yet
    index = VectorStoreIndex([], storage_context=storage_context, embed_model=OpenAIEmbedding())
    return index


def process_json(json_path: str, index_name: str):
    """
    Process a JSON file, parse it into separate documents,
    and upsert them into Pinecone via llama_index.
    """
    # 1) Ensure Pinecone
    ensure_pinecone_index_exists(index_name)
    pindex = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pindex)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2) Parse JSON => multiple docs
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []
    for i, item in enumerate(data):
        item_str = json.dumps(item, indent=2)
        metadata = {
            "filename": json_path,
            "item_index": i
        }
        docs.append(Document(text=item_str, metadata=metadata))

    # 3) Build index from documents (no chunking for JSON in your example)
    VectorStoreIndex.from_documents(
        documents=docs,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding()
    )
    logger.info(f"Successfully processed JSON file: {json_path}")


def process_pdf(pdf_dir: str, index_name: str):
    """
    Process PDF files inside `pdf_dir`, chunk them,
    and upsert them into Pinecone via llama_index.
    """
    from llama_index.core import SimpleDirectoryReader

    # 1) Ensure Pinecone
    ensure_pinecone_index_exists(index_name)
    pindex = pc.Index(index_name)
    vector_store = PineconeVectorStore(pinecone_index=pindex)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 2) Use SimpleDirectoryReader to parse PDFs in that folder
    docs = SimpleDirectoryReader(pdf_dir).load_data()

    # 3) Create or update index
    VectorStoreIndex.from_documents(
        documents=docs,
        storage_context=storage_context,
        embed_model=OpenAIEmbedding(),
        transformations=[SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)]
    )
    logger.info(f"Successfully processed PDF(s) in: {pdf_dir}")


def clear_pinecone_index(index_name: str) -> str:
    """Deletes the entire Pinecone index."""
    pc.delete_index(index_name)
    return f"Pinecone index '{index_name}' cleared"
