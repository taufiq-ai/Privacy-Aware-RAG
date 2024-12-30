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

FIELDS = [
    'product_code:int', 
    'price:int', 
    'in_stock:bool', 
    'category:str', 
    'sub_category:str'
]

def augment_generation(
    text:str,
    context = None
):  
    # context = retrieve_context(text=text, collection=collection)
    if context:
        generation = answer_from_context(context=context, query=text)
    else:
        generation = answer_without_context(query=text)
        
    return generation

def retrieve_context(
    text: str|list,
    collection,
    k=3,
    call_rag=False, 
    payload={}
): 
    if not call_rag:
        return None
    if not payload.get("result"):
        results, retrieval = retrieve_knowledge(
            collection=collection,
            query=text,
        )
        return retrieval
    
    results, retrieval = retrieve_knowledge_by_filter(
        collection=collection,
        query=text,
        where=payload["result"],
        k=k,
    )
    return retrieval

def understand_user_query(
    text:str,
    classes = ["CHITCHAT", "PRESALES_ENQUIRY", "CUSTOMER_SUPPORT", "ACCOUNT_ACTION", "FLAGGED_CONTENT"],
    prohibited_classes: list = ["CHITCHAT","ACCOUNT_ACTION", "FLAGGED_CONTENT"],
    ):
    call_rag = True
    payload = {
        "call_rag": call_rag,
        "classification_result": {},
        "ner_result": {},
        "filter_logic": {},
        "result": {},
    }
    
    # Classify the text
    classification_result = classify_text(text)
    payload["classification_result"] = classification_result
    if classification_result.get("class") in prohibited_classes:
        payload["call_rag"] = False
        return False, payload
    
    # Extract Named Entity Recognition (NER)
    ner_result = extract_ner(text=text, fileds=FIELDS)
    payload["ner_result"] = ner_result
    if not ner_result:
        return call_rag, payload
    
    filter_logic = optimize_filter_query(text=text, ner=ner_result, fileds=FIELDS)
    payload["filter_logic"] = filter_logic
    payload["result"] = filter_logic
    
    return call_rag, payload


"""
Process Flow:
collection = chroma.get_collection(collection_name, "BAAI/bge-m3")
call_rag, payload = rag.understand_user_query(text)
context = rag.retrieve_context(text, collection, k=3, call_rag, payload)
content = rag.augment_generation(text, context)
"""
    
