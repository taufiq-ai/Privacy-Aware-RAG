import structlog

from app.llm_engine import (
    classify_text,
    extract_ner,
    optimize_filter_query,
    answer_from_context,
    answer_without_context,
)
from services.chroma import (
    get_collection,
    retrieve_knowledge,
    retrieve_knowledge_by_filter,
)

logger = structlog.get_logger(__name__)
FIELDS = [
    'product_code:int', 
    'price:int', 
    'in_stock:bool', 
    'category:str', 
    'sub_category:str'
]

def pipe_generation(text:str, collection):
    class_payload = get_query_class(text=text)
    call_rag = class_payload.get("call_rag")
    query_class = class_payload.get("query_class")
    if not call_rag:
        if query_class == "CHITCHAT":
            return answer_without_context(query=text)
        else:
            return "I am sorry, I am not able to help you with this. Please contact customer support."
    
    payload = understand_user_query(text=text)
    context = retrieve_context(text=text, collection=collection, payload=payload)
    return answer_from_context(context=context, query=text)


def retrieve_context(
    text: str|list,
    collection,
    k=3,
    payload={}
): 
    optimized_filter_query = payload.get("result")
    logger.info(
        "Retrieving Context",
        text=text,
        collection=collection,
        k=k,
        optimized_filter_query=optimized_filter_query,
    )
    
    if not optimized_filter_query:
        results, retrieval = retrieve_knowledge(
            collection=collection,
            query=text,
        )
        return retrieval
    
    results, retrieval = retrieve_knowledge_by_filter(
        collection=collection,
        query=text,
        where=optimized_filter_query,
        k=k,
    )
    return retrieval

def understand_user_query(
    text:str,
    classes = ["CHITCHAT", "PRESALES_ENQUIRY", "CUSTOMER_SUPPORT", "ACCOUNT_ACTION", "FLAGGED_CONTENT"],
    prohibited_classes: list = ["CHITCHAT","ACCOUNT_ACTION", "FLAGGED_CONTENT"],
    ):
    payload = {
        "ner_result": {},
        "filter_logic": {},
        "result": {},
    }
    ner_result = extract_ner(text=text, fileds=FIELDS)
    payload["ner_result"] = ner_result
    if not ner_result:
        return payload
    
    filter_logic = optimize_filter_query(text=text, ner=ner_result, fileds=FIELDS)
    payload["filter_logic"] = filter_logic
    payload["result"] = filter_logic
    
    return payload

def get_query_class(
    text,
    call_rag=True,
    classes = ["CHITCHAT", "PRESALES_ENQUIRY", "CUSTOMER_SUPPORT", "ACCOUNT_ACTION", "FLAGGED_CONTENT"],
    prohibited_classes: list = ["CHITCHAT","ACCOUNT_ACTION", "FLAGGED_CONTENT"],
):
    payload= {
        "call_rag": call_rag,
        "query_class": "",
        "classification_result": {},
    }
    classification_result = classify_text(text)
    query_class = classification_result.get("class")
    if query_class in prohibited_classes:
        payload["call_rag"] = False
    payload["query_class"] = query_class
    payload["classification_result"] = classification_result
    return payload

"""
Process Flow:
collection = chroma.get_collection(collection_name, "BAAI/bge-m3")
call_rag, payload = rag.understand_user_query(text)
context = rag.retrieve_context(text, collection, k=3, call_rag, payload)
content = rag.augment_generation(text, context)
"""
    
