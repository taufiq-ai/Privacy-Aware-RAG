import os 
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
PPLX_API_KEY=os.getenv("PPLX_API_KEY")
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
VOYAGE_API_KEY=os.getenv("VOYAGE_API_KEY")