# https://cookbook.openai.com/

import os
import structlog
from openai import OpenAI
from enum import Enum
import settings

logger = structlog.get_logger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)


class OpenaiModel(Enum):
    GPT3T = "gpt-3.5-turbo"
    GPT4OM = "gpt-4o-mini" # 16K Output tokens
    GPT4O = "gpt-4o"
    GPT4T = "gpt-4-turbo"
    GPT4TP = "gpt-4-turbo-preview"
    GPT4 = "gpt-4"


MODEL_SM = OpenaiModel.GPT4OM.value
MODEL = OpenaiModel.GPT4O.value
MODEL_L = OpenaiModel.GPT4T.value
MODEL_XL = OpenaiModel.GPT4.value
MODEL_LARGE_CONTEXT = OpenaiModel.GPT4OM.value



def verify_seed(seed):
    response = client.moderations.create(input=seed)
    mod = response["results"][0]
    seed_count = len(seed.split())
    logger.info(
        f"Verified seed",
        flagged=mod["flagged"],
        **mod["category_scores"],
        metric_name=f"openai.moderation.create",
        seed_count=seed_count,
    )
    return not mod["flagged"]


def api_complete(
    prompt,
    model=OpenaiModel.GPT4O.value,
    frequency_penalty=0,
    top_p=1,
    max_tokens=2000,
    temperature=0.8,
    response_format="text",
    stop=None,
    **kwargs,
):

    if "response_format" in kwargs:
        model = MODEL if model.startswith("gpt-4") else MODEL_SM
    if not model.startswith("gpt") and not model.startswith("ft"):
        model = MODEL
    if type(prompt) == str:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        n=1,
        frequency_penalty=frequency_penalty,
        temperature=temperature,
        # top_p= top_p,
        stop=stop,
        response_format={"type": response_format},
        # **kwargs,
    )
    choice = completion.choices[0]
    content = choice.message.content

    logger.info(
        "Completion done",
        metric_name=f"openai.chat_completion.create.{model}",
        num_request=1,
        usage=completion.usage,
        model=model,
    )
    return completion, content