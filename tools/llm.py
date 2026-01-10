"""LLM abstraction using LiteLLM."""

import json
from typing import Any, Dict, List, Optional

from litellm import acompletion


class LLM:
    def __init__(
        self, model_name: str, max_tokens: int = 8192, temperature: float = 0.0
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.provider = self._get_provider(model_name)

    def _get_provider(self, model_name: str) -> str:
        if "claude" in model_name or "anthropic" in model_name:
            return "anthropic"
        if "gemini" in model_name or "google" in model_name:
            return "google"
        return "openai"

    async def chat(self, messages: List[Dict], tools: Optional[List] = None) -> Any:
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        return await acompletion(**kwargs)

    def get_tool_calls(self, response: Any) -> List[Dict]:
        """Extract tool calls from response."""
        tool_calls = []
        msg = response.choices[0].message
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    }
                )
        return tool_calls

    def get_text(self, response: Any) -> str:
        """Extract text content from response."""
        msg = response.choices[0].message
        return msg.content or ""

    def append_assistant(self, messages: List[Dict], response: Any):
        """Append assistant message to conversation."""
        msg = response.choices[0].message
        assistant_msg = {"role": "assistant", "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

    def append_tool_result(self, messages: List[Dict], tool_call_id: str, result: str):
        """Append tool result to conversation."""
        messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": result}
        )
