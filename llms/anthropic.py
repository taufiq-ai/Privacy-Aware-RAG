import os
import anthropic
import structlog
from enum import Enum
import settings

logger = structlog.get_logger(__name__)
client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)


class ClaudeModel(Enum):
    HAIKU = "claude-3-haiku-20240307"
    SONNET = "claude-3-sonnet-20240229"
    OPUS = "claude-3-opus-20240229"
    HAIKU35 = "claude-3-5-haiku-20241022"
    SONNET35 = "claude-3-5-sonnet-20240620"
    SONNET35_latest = "claude-3-5-sonnet-20241022"

MODEL_SM = ClaudeModel.HAIKU.value
MODEL = ClaudeModel.SONNET35.value
MODEL_L = ClaudeModel.OPUS.value
MODEL_XL = ClaudeModel.OPUS.value
MODEL_LARGE_CONTEXT = ClaudeModel.SONNET35.value


def api_complete(
    prompt, model=MODEL, max_tokens=3000, temperature=0.8, top_p=1, stop=None, **kwargs
):

    if "response_format" in kwargs:
        kwargs.pop("response_format")
    if not model.startswith("claude"):  ## TODO: Handle Finetuning Model later
        model = MODEL
    if type(prompt) == str:
        user_content = prompt
        system_content = ""
    else:
        ## TODO: what if we implement few shot in future?
        system_content = prompt[0]["content"]
        user_content = prompt[-1]["content"]

    completion = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        # top_p=top_p, ## NOTE: You should either alter temperature or top_p, but not both.
        system=system_content,
        messages=[{"role": "user", "content": user_content}],
        stop_sequences=stop,
        # **kwargs,
    )
    content = completion.content[0].text

    logger.info(
        "Completion done",
        metric_name=f"anthropic.messages.create.{model}",
        num_request=1,
        usage=completion.usage,
        model=model,
    )
    return completion, content