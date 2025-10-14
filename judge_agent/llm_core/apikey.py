"""Local credentials for Judge agents.

Populate the module-level constants with your own API keys before running
scripts that call commercial LLM services. Consider loading them from
environment variables instead of checking secrets into source control.
"""

import os
from typing import Optional

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
"""API key for accessing OpenAI endpoints (e.g., GPT-4o)."""

HUGGINGFACE_API_KEY: Optional[str] = os.getenv("HUGGINGFACE_API_KEY", "")
"""Token for authenticated Hugging Face Hub downloads."""

__all__ = ["OPENAI_API_KEY", "HUGGINGFACE_API_KEY"]
