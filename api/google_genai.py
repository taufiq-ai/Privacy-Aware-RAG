# https://ai.google.dev/gemini-api/docs
# https://ai.google.dev/pricing

import settings
import structlog
from enum import Enum

import google.generativeai as genai


genai.configure(api_key=settings.GEMINI_API_KEY)
logger = structlog.get_logger(__name__)

class GeminiModel(Enum):
    GEMINI_FLASH = "gemini-1.5-flash"
    GEMINI_FLASH_8B = "gemini-1.5-flash-8b"
    GEMINI_PRO = "gemini-1.5-pro"
    TEXT_EMBEDDING = "text-embedding-004"
    AQA = "aqa"

MODEL_SM = GeminiModel.GEMINI_FLASH.value
MODEL = GeminiModel.GEMINI_PRO.value
MODEL_L = GeminiModel.GEMINI_PRO.value
MODEL_LARGE_CONTEXT = GeminiModel.GEMINI_FLASH_8B.value
MODEL_EMBEDDING = GeminiModel.TEXT_EMBEDDING.value
MODEL_QA = GeminiModel.AQA.value

def api_complete(prompt, model=MODEL_SM, max_tokens=3000, temperature=0.8, top_p=1, stop=None, **kwargs):
	model = genai.GenerativeModel(model)
	response = model.generate_content(prompt)
	logger.info(
		"Completion done",
		metric_name=f"googelai.chat_completion.{model}",
		num_request=1,
		model=model,
	)
	return response, response.text


def embedding(model=MODEL_EMBEDDING):
    return


def question_answering(model=MODEL_QA):
    return