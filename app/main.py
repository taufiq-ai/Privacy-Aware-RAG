import time
import structlog

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
    retrieve_knowledge,
    get_collection,
)
from app.llm_interface import (
    answer_from_context,
)

logger = structlog.get_logger(__name__)

# xl_chunks = xl_to_chunks(filepath="data/data.xlsx")
# documents_xl, metadatas_xl = convert_chunk_obj_into_doc_str(chunks_obj=xl_chunks)
def populate_collection():
    """Initialize once to populate DB"""
    """Initialize and setup the collection with embeddings."""
    ef_model = "BAAI/bge-m3"
    collection_name = "Company_Info"
    
    text_chunks = text_to_chunks(text=company_details)
    documents, metadatas = convert_chunk_obj_into_lists(chunks_obj=text_chunks)
    
    ef = create_embedding_function_hf(model_name=ef_model)
    collection = get_or_create_collection(name=collection_name, embedding_function=ef)
    update_collection(collection=collection, documents=documents)
    
    return collection

def main():
    """Main function to run the QA system."""
    try:
        collection_name = "Company_Info"
        ef_model_name = "BAAI/bge-m3"
        logger.info("Initializing RAG", collection_name=collection_name, ef_model_name=ef_model_name)
        collection = get_collection(name=collection_name, ef_model_name=ef_model_name)
        while True:
            try:
                query = input("Question: ").strip()
                if not query:
                    continue
                result, retrieval = retrieve_knowledge(collection=collection, query=query)
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

if __name__ == "__main__":
    main()