import time
import random
import structlog
import re
import ast
import json
from datetime import datetime

from llms import (
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


