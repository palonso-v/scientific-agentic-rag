from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    max_retries=2,
    timeout=30.0,
)