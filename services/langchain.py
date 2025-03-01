import structlog
import pandas as pd
import settings

# chunking
from langchain.text_splitter import (
    MarkdownTextSplitter,
    RecursiveJsonSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import DataFrameLoader
from langchain.schema import Document

# embedding
from transformers import AutoModel
from langchain_huggingface import HuggingFaceEmbeddings

## Vector DB
# from langchain.vectorstores import Chroma ## FIXME
from langchain_community.vectorstores import Chroma

# Retriever
from langchain_openai import ChatOpenAI 
# from langchain.llms.openai import OpenAI # FIXME
from langchain_community.llms import OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import SelfQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

logger = structlog.get_logger("__name__")

def create_splitter(
    Langchain_Splitter=RecursiveCharacterTextSplitter,
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
    **args,
):
    splitter = Langchain_Splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        is_separator_regex=is_separator_regex,
    )
    return splitter


def xl_to_chunks(filepath: str):
    df = pd.read_excel(filepath)
    logger.info("[STARTED] Excel to chunks", filepath=filepath, number_of_records=len(df))
    
    documents = []
    for idx, row in df.iterrows():
        product_id = row.get('product_code', f'product_{idx}')
        product_content = '; '.join(f"{col}: {value}" for col, value in row.items())
        doc = Document(
            page_content=product_content,
            ## FIXME: parsing metadata should be dynamic
            metadata={
                'product_id': product_id,
                'price': row.get('price'),
                'in_stock': row.get('in_stock'),
                'category': row.get('category'),
                'sub_category': row.get('sub_category'),
            }
        )
        documents.append(doc)
    logger.info("[DONE] Excel Chuning", total_chunks=len(documents))
    return documents


def text_to_chunks(text):
    splitter = create_splitter(RecursiveCharacterTextSplitter)
    chunks = splitter.split_text(text)
    chunks_in_docs = [Document(page_content=chunk) for chunk in chunks]
    return chunks_in_docs

def convert_chunk_obj_into_lists(chunks_obj):
    # After chunking we have langchain chunks document objects but chroma expects these as list
    # langchain splitter objects example: 
    # >>> text_chunks[:2]
    # [
    #    Document(metadata={}, page_content='# TechVision Electronics'), 
    #    Document(metadata={}, page_content='## About Us')
    # ]
    logger.info("Converting Chunk Obj into List")
    documents = [doc.page_content for doc in chunks_obj]
    metadatas = [doc.metadata for doc in chunks_obj]
    logger.info("[DONE] Convertion", documents=documents[:1], total_documents=len(documents), metadatas=metadatas[:1], total_metadata=len(metadatas))
    return documents, metadatas


def download_hf_model(model_name: str = "all-MiniLM-L6-v2"):
    # "Marqo/marqo-ecommerce-embeddings-B"
    try:
        model = AutoModel.from_pretrained(
            model_name, cache_dir="./models"  # Local cache
        )
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None


def load_local_model(
    cache_dir="models",
    model_name="all-MiniLM-L6-v2",
):
    # "Marqo/marqo-ecommerce-embeddings-B"
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=model_name,
        local_files_only=True,
        cache_dir=cache_dir,
    )
    return model


def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        cache_folder="./models",
    )
    return embeddings


def save_to_vectordb(chunks, embedder):
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embedder)
    return vectorstore


def retrieve_from_db_using_llm(
    query,
    vectorstore,
    retriever_type="muliquery",
):
    llm_be=ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY, temperature=0.5, model="gpt-4o")
    # llm_be = get_custom_llm()
    logger.info("Initializing Retrivar", retriever_type=retriever_type, llm_be=llm_be)
    llm = llm_be
    if retriever_type == "muliquery":
        # 1. MultiQueryRetriever
        # - Generates multiple queries for better coverage
        retriever = MultiQueryRetriever.from_llm(
            retriever=vectorstore.as_retriever(), 
            llm=llm,
        )
    elif retriever_type == "structured":
        # 2. SelfQueryRetriever
        # - LLM converts natural language to structured queries
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vectorstore,
            document_contents="Company information",
            metadata_field_info=[],
        )
    elif retriever_type == "contextual":
        # LLM reranks and filters retrieved chunks
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vectorstore.as_retriever()
        )
    else:
        logger.error("retriever_type is required", retriever_type=retriever_type)

    logger.info("Retrival Prepared", retriever_type=retriever_type)
    docs = retriever.get_relevant_documents(query)
    return docs


def get_custom_llm():
    client = OpenAI(
        api_key=settings.OPENAI_API_KEY,
    )
    logger.info("Using custom LLM")
    return ChatOpenAI(
        model_name="gpt-4o",
        temperature=0.5,
        max_tokens=1000,
    )


def retrieve_from_vectordb(
    query: str,
    vectorstore,
    k: int = 3,
    search_type: str = "similarity"
):
    if search_type == "similarity":
        docs = vectorstore.similarity_search(query, k=k)
    elif search_type == "mmr":
        docs = vectorstore.max_marginal_relevance_search(
            query, k=k, fetch_k=k*2
        )
    elif search_type == "similarity_score":
        docs = vectorstore.similarity_search_with_score(query, k=k)
    return docs

def prepare_retrieval_context(docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    return context
