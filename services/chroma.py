# https://docs.trychroma.com/
# embedding support: ["OpenAI", "Google Gemini", "Cohere", "Hugging Face", "Instructor", "Hugging Face Embedding Server", "Jina AI", "Roboflow", "Ollama Embeddings"]
# framework support: ["Langchain", "LlamaIndex", "Braintrust", "OpenLLMetry", "Streamlit", "Haystack", "OpenLIT"]
# hf supported embedding models: https://huggingface.co/models
# DB Filter Query: https://cookbook.chromadb.dev/core/filters/

import settings
from typing import Literal
import uuid
from ulid import ULID
import structlog
import time

import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.api.types import IncludeEnum

logger = structlog.get_logger(__name__)

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
    logger.info("[Getting] Collection", collection_name=name, ef_model_name=ef_model_name, ef=ef)
    if ef_model_name:
        ef = create_embedding_function_hf(model_name=ef_model_name)
    collection = client.get_collection(
        name=name, 
        embedding_function=ef
    )
    logger.info("[Fetched] Collection", collection=collection, collection_name=name, ef=ef, ef_model_name=ef_model_name)
    return collection
    
def get_or_create_collection(
    name: str,
    metadata: dict = None,
    embedding_function=None,
    distance_metric: Literal[
        "l2", "cosine", "ip"
    ] = "cosine",  # Euclidean Distance Squared, cosine distance, inner product
):
    logger.info("[Fetching] Get or Create Collection", collection_name=name, distance_metric=distance_metric)
    if not metadata:
        metadata = {}
    metadata["hnsw:space"] = distance_metric

    collection = client.get_or_create_collection(
        name=name, metadata=metadata, embedding_function=embedding_function
    )
    logger.info("[Fetched] Get or Create collection", collection_name=name)
    return collection


def create_embedding_function_hf(
    # hf supported embedding models: https://huggingface.co/models
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_api_key=settings.HF_TOKEN,
):
    logger.info("[CREATING] Embedding Function", model_name=model_name)
    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        api_key=hf_api_key,
        model_name=model_name,
    )
    logger.info("[CREATED] Embedding Function", ef=huggingface_ef, model_name=model_name)
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
    logger.info("[Retrieving] Knowledge", collection=collection, query=query, number_of_retrieval=k)
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
    logger.info("[Retrieval] Done", collection=collection, query=query, retrieval=retrieval)
    return results, retrieval


def update_collection(collection, documents: list, **kwargs):
    """
    Args:
        collection: ChromaDB collection
        documents: List of text in chunks
        **kwargs [Optional parameters] -> embeddings:list, metadatas:list
    # switch `add` to `upsert` to avoid adding the same documents every time.
    """
    logger.info("[Updating] Collection", collection=collection, documents_count=len(documents))
    collection.upsert(
        documents=documents,
        ids=[f"{uuid.uuid4()}" for _ in range(len(documents))],
        # embeddings=embeddings,
        # metadatas = metadatas,
        **kwargs,
    )
    logger.info("[Updated] Collection", collection=collection, documents_count=len(documents))
    return

def update_collection_in_batches(collection_name:str, documents: list, ef=None, batch_size:int=1, **kwargs):
    total_docs = len(documents)
    print("\n\n")
    logger.info("[UPSERT BATCH START]", total_docs=total_docs, batch_size=batch_size, collection_name=collection_name)
    for i in range(0, total_docs, batch_size):
        upsert_args = {}
        upper_threshold = i + batch_size
        batch_documents = documents[i:upper_threshold]
        upsert_args["documents"] = batch_documents
        if 'metadatas' in kwargs and kwargs['metadatas']:
            upsert_args["metadatas"] = kwargs['metadatas'][i:upper_threshold]
        if 'embeddings' in kwargs and kwargs['embeddings']:
            upsert_args["embeddings"] = kwargs['embeddings'][i:upper_threshold]
        try:
            collection = get_collection(name=collection_name, ef=ef)
            time.sleep(0.5)
            upsert_args["collection"] = collection
            update_collection(**upsert_args)
            logger.info(f"[updated batch] doc: {i}-{upper_threshold}", batch_size=batch_size)
        except Exception as e:
            logger.error(f"[UPSERT ERROR] in batch {i}-{upper_threshold}", exc=str(e))
            continue
    return


def delete_collection():
    return


"""
embedding models tried so far:
1. "BAAI/bge-m3"
2. X "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
"""
