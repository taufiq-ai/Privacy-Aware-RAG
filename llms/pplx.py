import structlog
from openai import OpenAI
from enum import Enum
import settings

logger = structlog.get_logger(__name__)

client = OpenAI(api_key=settings.PPLX_API_KEY, base_url="https://api.perplexity.ai")

class PerplexityModel(Enum):
    SONAR_SMALL_ONLINE = "llama-3.1-sonar-small-128k-online"
    SONAR_LARGE_ONLINE = "llama-3.1-sonar-large-128k-online"
    SONAR_HUGE_ONLINE = "llama-3.1-sonar-huge-128k-online"
    SONAR_SMALL_CHAT = "llama-3.1-sonar-small-128k-chat"
    SONAR_LARGE_CHAT = "llama-3.1-sonar-large-128k-chat"

MODEL_SM = PerplexityModel.SONAR_SMALL_ONLINE.value
MODEL = PerplexityModel.SONAR_LARGE_ONLINE.value
MODEL_L = PerplexityModel.SONAR_HUGE_ONLINE.value


## TODO: pplx-aapi also works with openai library. integrate it to openai.api_comnplete later
## TODO: In June pplx-api will have json formatter. Till then format the JSON output with openai
def api_complete(prompt, model=MODEL, max_tokens=2500, temperature=1, **kwargs):
    if not model.startswith("llama"):       ## TODO: Handle Finetuning Model later
        model = MODEL 
    if type(prompt) == str:
        messages = [{"role": "user", "content": f"{prompt}"}]
    else:
        messages = prompt

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,

    )
    content = completion.choices[0].message.content

    logger.info(
        "Completion done",
        metric_name=f"perplexity.chat.completion.create.{model}",
        num_request=1,
        usage=completion.usage,
        model=model,
    )
    return completion, content

