import settings
import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login as hf_login
from typing import Tuple, List, Dict, Union

hf_login(token=settings.HF_TOKEN) 
logger = structlog.get_logger(__name__)

def get_pretrained_model_and_tokenizer(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device: str = "cpu"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
    device_map = {"": 0} if device == "cuda" else None
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(
        "Setup HF Causal Model", 
        model_name=model_name,
        device_map=device_map, 
        device=device,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def api_complete(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer,
    prompt: str|list,
    max_tokens: int = 200
):
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    
    encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")
    model_inputs = encodeds.to(model.device)
    
    generated_ids = model.generate(
        model_inputs, 
        max_new_tokens=max_tokens, 
        do_sample=True
    )
    
    decoded = tokenizer.batch_decode(generated_ids)
    content = _extract_mistral_instruct_answer(decoded[0])
    logger.info(
        "API Complete", 
        messages=messages,
        max_tokens=max_tokens, 
        content=content,
    )
    return decoded, content

def _extract_mistral_instruct_answer(decoded: str) -> str:
    return decoded.split("[/INST] ")[-1].split("</s>")[0].strip()
