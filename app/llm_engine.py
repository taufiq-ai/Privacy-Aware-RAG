import time
import random
import structlog
import re
import ast
import json
import inspect
import settings
from typing import Literal
from datetime import datetime
from importlib import import_module
from llms import prompts as template
from llms.utils import content_to_json

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


def request_llm(prompt, max_tokens=500, llm_be=llm_be, model=model, job=None, response_format:Literal["text", "json_object"]="text", logs=None, **kwargs):
    if job is None:
        job = inspect.currentframe().f_back.f_code.co_name
        
    logger.info(f"[{job.upper()}] Initialized", model=model, logs=logs)
    completion, content = llm_be.api_complete(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    if response_format=="json_object":
        output = content_to_json(content)
    else: output=content
    logger.info(f"[{job.upper()}] Completed", content=content, output=output)
    
    return completion, output


def answer_from_context(query:str, context: str|list):
    prompt = template.q_and_a.format(
        context=context,
        query=query,
    )
    completion, content = request_llm(
        prompt=prompt,
        model=model,
        logs={"query": query, "context": context},
    )
    return content

def answer_without_context(query:str):
    prompt = template.q_and_a_without_context.format(query=query)
    completion, content = request_llm(
        prompt=prompt,
        model=model,
        logs={"query": query},
    )
    return content



def extract_ner(text:str, fileds:list=FIELDS, response_format="json_object"):
    # fileds = ['product_code:int', 'product_name:str', 'brand:str', 'price:int', 'discount_percent:int', 'discounted_price:int', 'in_stock:bool', 'stock_quantity:int', 'category:str', 'sub_category:str', 'description:str', 'ratings:float', 'reviews_count:int', 'warranty_months:int', 'added_date:date', 'tags:list']
    prompt = template.ner.format(
        fileds=fileds,
        text=text,
        example_output=template.ner_example_output,
    )
    completion, content = request_llm(
        prompt=prompt,
        response_format=response_format,
        logs={"text": text, "fileds": fileds},
    )
    return content
    

def classify_text(
    text:str, 
    classes:str|list = "", # TODO: PASS Later
    response_format="json_object",
):
    prompt = template.classification.format(
        text=text,
        classes=classes,
        example_output=template.classification_output,
    )
    completion, content = request_llm(
        prompt=prompt,
        model=model,
        response_format=response_format,
        logs={"text": text, "classes": classes},
    )
    return content

def shortlist_vector_collections(
    query_category:dict,
    collections_description: dict,
    response_format="json_object",
) -> list:
    prompt = template.shortlist_collections_user_content.format(
        categories=query_category,
        collection_details=collections_description,
        example_output=template.shortlist_collections_example_output,
    )
    completion, content = request_llm(
        prompt=prompt,
        model=model,
        response_format=response_format,
        logs={"query_category": query_category, "collections_description": collections_description},
    )
    return content

def optimize_filter_query(
    text:str,
    ner:dict,
    fileds:list =FIELDS,
    response_format="json_object",
):
    # TODO: Should provide rules of operator and formatting usage
    prompt = template.db_filter.format(
        text=text,
        fields=fileds,
        ner_result=ner,
        example_format=template.db_filter_example_format,
        example_output=template.db_filter_example_output,
    )
    completion, content = request_llm(
        prompt=prompt,
        model=model,
        response_format=response_format,
        logs={"text": text, "fields": fileds, "ner": ner},
    )
    return content  
