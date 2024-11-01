"""
API Integration module for ProteinFlex.
Provides unified interface for multiple LLM APIs and local models.
"""

import os
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests

class BaseModelAPI(ABC):
    """Abstract base class for model API integration."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from the model."""
        pass

    @abstractmethod
    def analyze_protein(self, sequence: str, **kwargs) -> Dict[str, Any]:
        """Analyze protein sequence."""
        pass

class OpenAIAPI(BaseModelAPI):
    """OpenAI API integration."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('OPENAI_APIKEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found")

    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation for OpenAI API calls
        return "OpenAI response"

    def analyze_protein(self, sequence: str, **kwargs) -> Dict[str, Any]:
        # Implementation for protein analysis using OpenAI
        return {"analysis": "OpenAI analysis"}

class ClaudeAPI(BaseModelAPI):
    """Anthropic Claude API integration."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('CLAUDE_API_KEY')
        if not self.api_key:
            raise ValueError("Claude API key not found")

    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation for Claude API calls
        return "Claude response"

    def analyze_protein(self, sequence: str, **kwargs) -> Dict[str, Any]:
        # Implementation for protein analysis using Claude
        return {"analysis": "Claude analysis"}

class GeminiAPI(BaseModelAPI):
    """Google Gemini API integration."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GEMINI_API')
        if not self.api_key:
            raise ValueError("Gemini API key not found")

    def generate(self, prompt: str, **kwargs) -> str:
        # Implementation for Gemini API calls
        return "Gemini response"

    def analyze_protein(self, sequence: str, **kwargs) -> Dict[str, Any]:
        # Implementation for protein analysis using Gemini
        return {"analysis": "Gemini analysis"}

class APIManager:
    """Manager class for handling multiple API integrations."""

    def __init__(self):
        self.apis = {}
        self._initialize_apis()

    def _initialize_apis(self):
        """Initialize available APIs."""
        try:
            self.apis['openai'] = OpenAIAPI()
        except ValueError:
            pass

        try:
            self.apis['claude'] = ClaudeAPI()
        except ValueError:
            pass

        try:
            self.apis['gemini'] = GeminiAPI()
        except ValueError:
            pass

    def get_api(self, name: str) -> BaseModelAPI:
        """Get specific API instance."""
        if name not in self.apis:
            raise ValueError(f"API {name} not found or not configured")
        return self.apis[name]

    def analyze_with_all(self, sequence: str, **kwargs) -> Dict[str, Dict[str, Any]]:
        """Analyze protein sequence using all available APIs."""
        results = {}
        for name, api in self.apis.items():
            try:
                results[name] = api.analyze_protein(sequence, **kwargs)
            except Exception as e:
                results[name] = {"error": str(e)}
        return results
