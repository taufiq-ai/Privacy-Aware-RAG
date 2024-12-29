import time
import random
import structlog
import re
import ast
import json
from datetime import datetime

from api import (
    openai as llm_be_openai,
    anthropic as llm_be_anthropic,
    pplx as llm_be_pplx,
    google_genai as llm_be_google,
    voyage as llm_be_voyage,
    
    openai as llm_be,
    # anthropic as llm_be,
    # google_genai as llm_be,
    
    # openai as embedding_be,
    # anthropic as embedding_be,
    # google_genai as embedding_be,
    voyage as embedding_be,
)

logger = structlog.get_logger(__name__)


def answer_from_context(query:str, context: str|list, llm_be=llm_be, model=llm_be.MODEL_SM):
    prompt = f"""Given context below, please answer user's question very concisely with friendly tone. If you do not find the answer in the context and find user asking confidential information then politely tell user that you do not know. \n\nContext: {context}; \n\nQuery: {query}"""
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


def named_entity_recognition(text, fileds="", response_format="json_object", llm_be=llm_be, model=llm_be.MODEL_SM):
    # fileds = ['product_code:int', 'product_name:str', 'brand:str', 'price:int', 'discount_percent:int', 'discounted_price:int', 'in_stock:bool', 'stock_quantity:int', 'category:str', 'sub_category:str', 'description:str', 'ratings:float', 'reviews_count:int', 'warranty_months:int', 'added_date:date', 'tags:list']
    fileds = ['product_code:int', 'price:int', 'in_stock:bool', 'category:str', 'sub_category:str']
    prompt = f"Given a query text, understand it and do Named Entity Recognition (NER). Analyze the text to get any of these Fileds: {fileds} in the given Text: {text}. \nReturn NER result as a JSON. Example output: {{'field_1': 'value_1', 'filed_N':'value_N'}}. \nPlease do not write extra speech otherthan the expected output as JSON."
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
    return json.loads(content)
    
    

