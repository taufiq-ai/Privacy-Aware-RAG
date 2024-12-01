# https://docs.trychroma.com/
# embedding support: ["OpenAI", "Google Gemini", "Cohere", "Hugging Face", "Instructor", "Hugging Face Embedding Server", "Jina AI", "Roboflow", "Ollama Embeddings"]
# framework support: ["Langchain", "LlamaIndex", "Braintrust", "OpenLLMetry", "Streamlit", "Haystack", "OpenLIT"]
# hf supported embedding models: https://huggingface.co/models

import settings
from typing import Literal
import uuid
from ulid import ULID

# _ulid = ULID()


import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.types import IncludeEnum

# chroma_client = chromadb.Client()
client = chromadb.PersistentClient(
    path=".chroma",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)
# HttpClient = chromadb.HttpClient(
#     host="localhost",
#     port=8025,
#     ssl=False,
#     headers=None,
#     settings=Settings(),
#     tenant=DEFAULT_TENANT,
#     database=DEFAULT_DATABASE,
# )

def get_collection(name:str, ef_model_name:str=None, ef=None):
    if ef_model_name:
        ef = create_embedding_function_hf(model_name=ef_model_name)
    return client.get_collection(
        name=name, 
        embedding_function=ef
    )
    
def get_or_create_collection(
    name: str,
    metadata: dict = None,
    embedding_function=None,
    distance_metric: Literal[
        "l2", "cosine", "ip"
    ] = "cosine",  # Euclidean Distance Squared, cosine distance, inner product
):
    if not metadata:
        metadata = {}
    metadata["hnsw:space"] = distance_metric

    collection = client.get_or_create_collection(
        name=name, metadata=metadata, embedding_function=embedding_function
    )
    return collection


def create_embedding_function_hf(
    # hf supported embedding models: https://huggingface.co/models
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_api_key=settings.HF_TOKEN,
):
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=hf_api_key,
        model_name=model_name,
    )
    return huggingface_ef


def retrieve_knowledge(
    collection,
    query: str | list,
    retrieve_embeddings=False,
    k: int = 3,
    include: list = [
        IncludeEnum.distances,
        IncludeEnum.documents,
        IncludeEnum.metadatas,
    ],
):
    if type(query) == str:
        query = [query]
    
    results = collection.query(
        query_texts=query,  # Chroma will embed this for you
        n_results=k,
        include=(
            include
            if not retrieve_embeddings
            else include.append(IncludeEnum.embeddings)
        ),
    )
    retrieval = ["\n".join(doc) for doc in results["documents"]]
    return results, retrieval


def update_collection(collection, documents: list, **kwargs):
    """
    Args:
        collection: ChromaDB collection
        documents: List of text in chunks
        **kwargs [Optional parameters] -> embeddings:list, metadatas:list
    # switch `add` to `upsert` to avoid adding the same documents every time.
    """
    collection.upsert(
        documents=documents,
        ids=[f"{uuid.uuid4()}" for _ in range(len(documents))],
        # ids=[f"{_ulid.generate()}" for _ in range(len(documents))],
        # embeddings=embeddings,
        # metadatas = metadatas,
        **kwargs,
    )
    return


def delete_collection():
    return


"""
embedding models tried so far:
1. "BAAI/bge-m3"
2. X "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
"""
