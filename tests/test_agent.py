from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.agent import QAAgent
from app.models import ToolResult
from app.tools import Tool


class FakeTool(Tool):
    name: str = "fake_search"
    description: str = "fake"

    def declaration(self):
        return {"name": self.name, "description": self.description, "parameters": {"type": "object", "properties": {}}}

    def run(self, **kwargs):
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=[{"paper_title": "Attention Is All You Need", "text": "x"}],
        )


def _part(text=None, function_call=None):
    return SimpleNamespace(text=text, function_call=function_call)


def _response(parts, text=""):
    candidate = SimpleNamespace(content=SimpleNamespace(parts=parts))
    return SimpleNamespace(candidates=[candidate], text=text)


def test_agent_calls_tool_then_returns_text():
    function_call = SimpleNamespace(name="fake_search", args={"query": "attention"})
    first_response = _response([_part(function_call=function_call)])
    final_response = _response([_part(text="The paper proposes self-attention.")], text="The paper proposes self-attention.")

    chat = MagicMock()
    chat.send_message.side_effect = [first_response, final_response]

    with patch("app.agent.genai") as mock_genai:
        model = MagicMock()
        model.start_chat.return_value = chat
        mock_genai.GenerativeModel.return_value = model
        mock_genai.protos.Part = MagicMock
        mock_genai.protos.FunctionResponse = MagicMock

        agent = QAAgent(tools=[FakeTool()])
        result = agent.ask("What is the central mechanism?")

    assert "self-attention" in result.answer.lower()
    assert "Attention Is All You Need" in result.sources
    assert chat.send_message.call_count == 2


def test_agent_returns_direct_answer_without_tool_call():
    final = _response([_part(text="Direct answer.")], text="Direct answer.")
    chat = MagicMock()
    chat.send_message.return_value = final

    with patch("app.agent.genai") as mock_genai:
        model = MagicMock()
        model.start_chat.return_value = chat
        mock_genai.GenerativeModel.return_value = model

        agent = QAAgent(tools=[FakeTool()])
        result = agent.ask("hi")

    assert result.answer == "Direct answer."
    assert result.sources == []
