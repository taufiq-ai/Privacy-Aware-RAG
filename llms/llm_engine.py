import time
import random
import structlog
import re
import ast
import json
import settings
from datetime import datetime
from importlib import import_module
from llms import prompts as template

logger = structlog.get_logger(__name__)
llm_be = import_module(settings.LLM_BACKEND)
model = llm_be.MODEL_SM
FIELDS = [
    'product_code:int', 
    'price:int', 
    'in_stock:bool', 
    'category:str', 
    'sub_category:str'
]

def answer_from_context(query:str, context: str|list, llm_be=llm_be, model=model):
    prompt = template.q_and_a.format(
        context=context,
        query=query,
    )
    logger.info(
        "[Chatbot] answring based on [RAG]",
        query=query,
        context=context,
        prompt=prompt,
    )
    completion, content = llm_be.api_complete(
        prompt=prompt,
        model=model,
    )
    return content


def recognize_named_entity(text, fileds=FIELDS, response_format="json_object", llm_be=llm_be, model=model):
    # fileds = ['product_code:int', 'product_name:str', 'brand:str', 'price:int', 'discount_percent:int', 'discounted_price:int', 'in_stock:bool', 'stock_quantity:int', 'category:str', 'sub_category:str', 'description:str', 'ratings:float', 'reviews_count:int', 'warranty_months:int', 'added_date:date', 'tags:list']
    prompt = template.ner.format(
        fileds=fileds,
        text=text,
        example_output=template.ner_example_output,
    )
    logger.info(
        "[NER] Analyzing Query",
        text=text,
        model=model,
    )
    completion, content = llm_be.api_complete(
        prompt=prompt,
        model=model,
        response_format=response_format,
    )
    logger.info(
        "[NER] DONE",
        content=content,
    )
    return content
    

def classify_text(
    text:str, 
    classes:str|list = "", 
    llm_be=llm_be, 
    model=model,
    response_format="json_object",
):
    prompt = template.classification.format(
        text=text,
        classes=template.classification_classes,
        example_output=template.classification_output,
    )
    logger.info("[LLM] Classifying Query", text=text, model=model)
    completion, content = llm_be.api_complete(
        prompt=prompt,
        model=model,
        response_format=response_format,
    )
    return content

def optimize_filter_query(
    text:str,
    ner,
    fileds:list = FIELDS,
    llm_be=llm_be,
    model=model,
):
    # TODO: Should provide rules of operator and formatting usage
    prompt = template.db_filter.format(
        text=text,
        fields=fileds,
        ner_result=ner,
        example_format=template.db_filter_example_format,
        example_output=template.db_filter_example_output,
    )
    logger.info("[LLM] Optimizing Filter Query", text=text, model=model)
    completion, content = llm_be.api_complete(
        prompt=prompt,
        model=model,
    )
    logger.info("[LLM] Optimized Filter Query", content=content)
    return content    
