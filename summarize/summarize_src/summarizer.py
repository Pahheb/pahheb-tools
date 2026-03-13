"""AI summarization module supporting multiple providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional

import httpx


class SummarizerError(Exception):
    """Base exception for summarizer errors."""

    pass


class ProviderNotAvailableError(SummarizerError):
    """Raised when the requested provider is not available."""

    pass


class ModelNotFoundError(SummarizerError):
    """Raised when the requested model is not found."""

    pass


@dataclass
class SummaryResult:
    """Result of a summarization operation."""

    summary: str
    key_points: list[str]
    metadata: dict


class BaseProvider(ABC):
    """Abstract base class for AI providers."""

    @abstractmethod
    def summarize(self, text: str, summary_type: str = "standard") -> SummaryResult:
        """Summarize the given text."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass


class OllamaProvider(BaseProvider):
    """Ollama local provider."""

    def __init__(
        self, model: str = "llama3.2", base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url

    def is_available(self) -> bool:
        """Check if Ollama is running."""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    def _build_prompt(self, text: str, summary_type: str) -> str:
        """Build the prompt based on summary type."""
        length_instruction = {
            "brief": "Provide a very concise summary in 2-3 sentences.",
            "standard": "Provide a comprehensive summary in one or two paragraphs.",
            "detailed": "Provide a detailed summary with all important information preserved.",
        }.get(summary_type, "Provide a comprehensive summary.")

        return f"""You are an expert at summarizing transcriptions from videos and audio.
Your goal is to extract the main points and condense the content while preserving all important information.
Remove filler words, repetitions, and non-essential content.
{length_instruction}

First, provide a list of key points (bullet points).
Then, provide the detailed summary.

Format your response as follows:
KEYPOINTS:
- Point 1
- Point 2
- Point 3

SUMMARY:
[Your detailed summary here]

TRANSCRIPTION TO SUMMARIZE:
{text}

Remember: Preserve all factual information, names, dates, numbers, and technical details.
"""

    def summarize(self, text: str, summary_type: str = "standard") -> SummaryResult:
        """Summarize text using Ollama."""
        prompt = self._build_prompt(text, summary_type)

        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                        },
                    },
                )

                if response.status_code != 200:
                    raise SummarizerError(
                        f"Ollama API error: {response.status_code} - {response.text}"
                    )

                result = response.json()
                content = result.get("response", "")

                return self._parse_response(content)

        except httpx.ConnectError:
            raise ProviderNotAvailableError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running? Run 'ollama serve' to start it."
            )
        except httpx.TimeoutException:
            raise SummarizerError("Request timed out. The text may be too long.")

    def _parse_response(self, content: str) -> SummaryResult:
        """Parse the Ollama response into key points and summary."""
        key_points = []

        parts = content.split("SUMMARY:", 1)
        if len(parts) > 1:
            summary_part = parts[1].strip()
        else:
            summary_part = content.strip()

        points_part = parts[0].strip() if len(parts) > 1 else ""

        if points_part.startswith("KEYPOINTS:"):
            points_part = points_part[10:].strip()

        for line in points_part.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("-"):
                point = line[1:].strip()
            elif line[0].isdigit() and "." in line:
                point = line.split(".", 1)[1].strip()
            else:
                continue
            if point:
                key_points.append(point)

        if not key_points and summary_part:
            key_points = ["See summary for main points"]

        return SummaryResult(
            summary=summary_part,
            key_points=key_points,
            metadata={"provider": "ollama", "model": self.model},
        )


class HuggingFaceProvider(BaseProvider):
    """Hugging Face local provider using transformers."""

    def __init__(
        self,
        model: str = "microsoft/Phi-3.5-mini-instruct",
        device: str = "auto",
    ):
        self.model_name = model
        self.device = device
        self._model = None
        self._tokenizer = None

    def is_available(self) -> bool:
        """Check if transformers is available and model can be loaded."""
        try:
            import transformers

            return True
        except ImportError:
            return False

    def _ensure_model_loaded(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device = self.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device,
                    trust_remote_code=True,
                )
            except ImportError as e:
                raise ProviderNotAvailableError(
                    "transformers library not installed. "
                    "Install with: pip install transformers torch"
                )
            except Exception as e:
                raise ModelNotFoundError(f"Failed to load model {self.model_name}: {e}")

    def _build_prompt(self, text: str, summary_type: str) -> str:
        """Build the prompt based on summary type."""
        length_instruction = {
            "brief": "Provide a very concise summary in 2-3 sentences.",
            "standard": "Provide a comprehensive summary in one or two paragraphs.",
            "detailed": "Provide a detailed summary with all important information preserved.",
        }.get(summary_type, "Provide a comprehensive summary.")

        return f"""<s>[INST]
You are an expert at summarizing transcriptions from videos and audio.
Your goal is to extract the main points and condense the content while preserving all important information.
Remove filler words, repetitions, and non-essential content.
{length_instruction}

First, provide a list of key points (bullet points).
Then, provide the detailed summary.

Format your response as follows:
KEYPOINTS:
- Point 1
- Point 2
- Point 3

SUMMARY:
[Your detailed summary here]

TRANSCRIPTION TO SUMMARIZE:
{text}

Remember: Preserve all factual information, names, dates, numbers, and technical details.
[/INST]"""

    def summarize(self, text: str, summary_type: str = "standard") -> SummaryResult:
        """Summarize text using Hugging Face."""
        self._ensure_model_loaded()

        prompt = self._build_prompt(text, summary_type)

        try:
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                )

            response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)

            content = (
                response.split("[/INST]")[-1].strip()
                if "[/INST]" in response
                else response
            )

            return self._parse_response(content)

        except Exception as e:
            raise SummarizerError(f"Error during summarization: {e}")

    def _parse_response(self, content: str) -> SummaryResult:
        """Parse the response into key points and summary."""
        key_points = []

        parts = content.split("SUMMARY:", 1)
        if len(parts) > 1:
            summary_part = parts[1].strip()
        else:
            summary_part = content.strip()

        points_part = parts[0].strip() if len(parts) > 1 else ""

        if points_part.startswith("KEYPOINTS:"):
            points_part = points_part[10:].strip()

        for line in points_part.split("\n"):
            line = line.strip()
            if line.startswith("-"):
                point = line[1:].strip()
            elif line and line[0].isdigit() and "." in line:
                point = line.split(".", 1)[1].strip()
            else:
                continue
            if point:
                key_points.append(point)

        if not key_points and summary_part:
            key_points = ["See summary for main points"]

        return SummaryResult(
            summary=summary_part,
            key_points=key_points,
            metadata={"provider": "huggingface", "model": self.model_name},
        )


class WatsonxProvider(BaseProvider):
    """IBM watsonx.ai provider (cloud-based)."""

    def __init__(
        self,
        model: str = "ibm/granite-4-hulti",
        project_id: str | None = None,
        api_key: str | None = None,
        url: str = "https://us-south.ml.cloud.ibm.com",
    ):
        self.model = model
        self.project_id = project_id
        self.api_key = api_key
        self.url = url

    def is_available(self) -> bool:
        """Check if watsonx credentials are configured."""
        return bool(self.api_key and self.project_id)

    def summarize(self, text: str, summary_type: str = "standard") -> SummaryResult:
        """Summarize text using watsonx.ai (cloud)."""
        if not self.is_available():
            raise ProviderNotAvailableError(
                "watsonx.ai credentials not configured. "
                "Set WATSONX_API_KEY and WATSONX_PROJECT_ID environment variables."
            )

        raise NotImplementedError("watsonx provider is not yet implemented")


def get_provider(
    provider: str,
    model: str | None = None,
    **kwargs,
) -> BaseProvider:
    """Get a provider instance based on the provider name."""
    if provider == "ollama":
        return OllamaProvider(model=model or "llama3.2")
    elif provider == "huggingface":
        return HuggingFaceProvider(model=model or "microsoft/Phi-3.5-mini-instruct")
    elif provider == "watsonx":
        return WatsonxProvider(model=model or "ibm/granite-4-hulti")
    else:
        raise ValueError(f"Unknown provider: {provider}")
