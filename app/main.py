import time
import structlog
from typing import Literal

from data.data import company_details
from utils.langchain import (
    xl_to_chunks,
    text_to_chunks,
    convert_chunk_obj_into_lists,
)
from utils.chroma import (
    create_embedding_function_hf,
    get_or_create_collection,
    update_collection,
    update_collection_in_batches,
    retrieve_knowledge,
    get_collection,
)
from app.llm_interface import (
    answer_from_context,
)

logger = structlog.get_logger(__name__)


def populate_collection(
    collection_name: str,
    data: str,  # str data or filepath
    ef_model: str = None, # "BAAI/bge-m3",
    data_type: Literal["text", "xlsx", "str"] = "str",
):
    metadatas = []
    logger.info(
        "Populating collection",
        collection_name=collection_name,
        ef_model=ef_model,
        data_type=data_type,
    )
    ef = create_embedding_function_hf(model_name=ef_model)
    collection = get_or_create_collection(name=collection_name, embedding_function=ef)

    documents, metadatas = _get_documents_and_metadata_for_update(data=data, data_type=data_type)
    
    try:
        update_collection_in_batches(
            collection_name=collection_name, 
            ef=ef,
            documents=documents,
            metadatas=metadatas,
        )
    except Exception as exc:
        logger.error(
            "Can't fetch or update collection",
            collection_name=collection_name,
            ef_model=ef_model,
            exc=str(exc),
        )
        return None
    return collection


def _get_documents_and_metadata_for_update(data, data_type:Literal["text", "xlsx", "str"] = "str",):
    # 2. Read raw str data or xl file and convert into chunk of documents and metadata
    if data_type == "str":
        text_chunks = text_to_chunks(text=data)
        documents, metadatas = convert_chunk_obj_into_lists(chunks_obj=text_chunks)
    elif data_type == "xlsx":
        xl_chunks = xl_to_chunks(filepath=data)
        documents, metadatas = convert_chunk_obj_into_lists(chunks_obj=xl_chunks)
    return documents, metadatas


def main(collection_name, ef_model_name="BAAI/bge-m3"):
    """Main function to run the QA system."""
    logger.info(
        "Initializing RAG",
        collection_name=collection_name,
        ef_model_name=ef_model_name,
    )
    try:
        try:
            collection = get_collection(name=collection_name, ef_model_name=ef_model_name)
            time.sleep(0.5)
        except Exception as exc:
            logger.error(
                "Can not fetch collection",
                collection_name=collection_name,
                ef_model=ef_model_name,
            )
            return
        
        while True:
            try:
                query = input("Question: ").strip()
                if not query:
                    continue
                result, retrieval = retrieve_knowledge(
                    collection=collection, query=query
                )
                answer = answer_from_context(query=query, context=retrieval)
                print(f"\nAnswer: {answer}\n")
            except KeyboardInterrupt:
                print("\nExiting the chat...")
                break
            except Exception as e:
                print(f"\nError processing query: {str(e)}\n")
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")
        return


# if __name__ == "__main__":
#     main()
