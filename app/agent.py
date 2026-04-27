"""QA agent built on Gemini with native function calling."""
import json
import logging
import time
from typing import Any, Callable, TypeVar

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from google.generativeai.types import content_types

from app.config import settings
from app.models import AskResponse
from app.tools import Tool, all_tools

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Gemini surfaces rate-limit errors as `ResourceExhausted` over gRPC and as
# `TooManyRequests` over REST — we treat both identically.
RateLimitError = (ResourceExhausted, TooManyRequests)

SYSTEM_PROMPT = """You are a research assistant that answers questions about scientific papers indexed in a local vector store. You do not know the corpus in advance — discover it through the tools.

Rules:
1. Use the available tools to gather evidence before answering. Call `search_documents` for general questions and `extract_section` when the user asks about a specific section (abstract, introduction, method, experiments, results, conclusion, related_work) of a specific paper.
2. Prefer a single, well-formed tool call. Only call a tool a second time if the first result is clearly empty or unrelated — never more than two tool calls per question.
3. Base your answer ONLY on the content returned by the tools. Do not rely on prior knowledge. If the tools return nothing relevant, state that the indexed papers do not cover the question.
4. Cite the paper(s) you used by their title (taken from the tool output).
5. Be concise, faithful to the source, and avoid speculation.
6. Reply in the same language as the question.
"""

MAX_TOOL_ITERATIONS = 5


def _suggested_retry_delay(exc: Exception) -> float | None:
    """Read Gemini's suggested retry delay (seconds) from the exception details, if present."""
    details = getattr(exc, "details", None)
    if callable(details):
        details = details()
    for detail in details or []:
        delay = getattr(detail, "retry_delay", None) or (
            detail.get("retryDelay") if isinstance(detail, dict) else None
        )
        if delay is None:
            continue
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                continue
        seconds = getattr(delay, "seconds", 0)
        if seconds:
            return float(seconds) + getattr(delay, "nanos", 0) / 1e9
    return None


def _with_retry(call: Callable[[], T], attempts: int = 4, base_delay: float = 15.0) -> T:
    """Retry a Gemini call on rate-limit errors, honoring the server's suggested delay when present."""
    for attempt in range(attempts):
        try:
            return call()
        except RateLimitError as exc:
            if attempt == attempts - 1:
                raise
            suggested = _suggested_retry_delay(exc)
            delay = max(suggested + 2.0 if suggested else base_delay * (2**attempt), 5.0)
            delay = min(delay, 90.0)
            logger.warning("Gemini rate-limited, sleeping %.1fs (attempt %d/%d)", delay, attempt + 1, attempts)
            time.sleep(delay)
    raise RuntimeError("unreachable")


class QAAgent:
    """Gemini-backed agent that answers questions by orchestrating tool calls."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        """Configure Gemini and register the tools available for function calling."""
        # REST transport is more portable than gRPC (works reliably inside containers
        # where gRPC can hit network/permission issues on Docker Desktop).
        genai.configure(api_key=settings.gemini_api_key, transport="rest")
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or all_tools())}
        gemini_tools = [{"function_declarations": [t.declaration() for t in self.tools.values()]}]
        self.model = genai.GenerativeModel(
            model_name=settings.gemini_model,
            tools=gemini_tools,
            system_instruction=SYSTEM_PROMPT,
        )

    def ask(self, question: str) -> AskResponse:
        """Run the function-calling loop until Gemini produces a final textual answer."""
        chat = self.model.start_chat(enable_automatic_function_calling=False)
        response = _with_retry(lambda: chat.send_message(question))
        sources: set[str] = set()

        for _ in range(MAX_TOOL_ITERATIONS):
            function_calls = [
                part.function_call
                for part in response.candidates[0].content.parts
                if getattr(part, "function_call", None) and part.function_call.name
            ]
            if not function_calls:
                break

            tool_responses: list[content_types.PartType] = []
            for fc in function_calls:
                args = {k: v for k, v in fc.args.items()} if fc.args else {}
                logger.info("Tool call: %s args=%s", fc.name, args)
                result = self._run_tool(fc.name, args)
                self._collect_sources(result, sources)
                tool_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fc.name,
                            response={"result": json.dumps(result, default=str)},
                        )
                    )
                )
            response = _with_retry(lambda: chat.send_message(tool_responses))

        answer_text = self._extract_text(response)
        return AskResponse(question=question, answer=answer_text, sources=sorted(sources))

    def _run_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call by name and return the serialized `ToolResult`."""
        tool = self.tools.get(name)
        if tool is None:
            return {"success": False, "error": f"Unknown tool: {name}"}
        result = tool.run(**args)
        return result.model_dump()

    @staticmethod
    def _extract_text(response: Any) -> str:
        """Pull the final text from a Gemini response, falling back to manual part traversal."""
        try:
            return response.text.strip()
        except (ValueError, AttributeError):
            parts = response.candidates[0].content.parts
            texts = [p.text for p in parts if getattr(p, "text", None)]
            return "\n".join(texts).strip() or "No answer produced."

    @staticmethod
    def _collect_sources(result: dict[str, Any], sources: set[str]) -> None:
        """Add any `paper_title` found in a tool result to the running sources set."""
        if not result.get("success"):
            return
        data = result.get("data")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("paper_title"):
                    sources.add(str(item["paper_title"]))
        elif isinstance(data, dict) and data.get("paper_title"):
            sources.add(str(data["paper_title"]))
