from __future__ import annotations

from typing import List

import httpx

from app.core.config import Settings


class MistralClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = "https://api.mistral.ai/v1"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.settings.mistral_api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, inputs: List[str]) -> List[List[float]]:
        payload = {"model": self.settings.mistral_embed_model, "input": inputs}
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{self.base_url}/embeddings", headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings

    def chat(self, messages: List[dict], temperature: float = 0.2) -> str:
        payload = {
            "model": self.settings.mistral_chat_model,
            "messages": messages,
            "temperature": temperature,
        }
        with httpx.Client(timeout=45.0) as client:
            response = client.post(f"{self.base_url}/chat/completions", headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"].strip()