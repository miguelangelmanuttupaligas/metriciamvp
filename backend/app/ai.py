import json
import re

from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI

from app.config import get_settings


settings = get_settings()


def require_openai_key() -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY no está configurada.")
    return settings.openai_api_key


def get_embedding_model() -> OpenAIEmbedding:
    kwargs = {
        "model": settings.embedding_model,
        "api_key": require_openai_key(),
        "timeout": settings.openai_timeout_seconds,
    }
    if settings.embedding_dimensions:
        kwargs["dimensions"] = settings.embedding_dimensions
    return OpenAIEmbedding(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedding_model().get_text_embedding_batch(texts)


def llm_text(system_prompt: str, user_prompt: str) -> str:
    client = OpenAI(api_key=require_openai_key(), timeout=settings.openai_timeout_seconds)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.create(model=settings.llm_model, input=messages)
    return response.output_text.strip()


def llm_json(system_prompt: str, user_prompt: str) -> dict:
    text = llm_text(system_prompt, user_prompt)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))
