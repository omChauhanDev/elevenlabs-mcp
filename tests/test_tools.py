import json
import pytest
from elevenlabs_mcp.server import list_tools, get_tool


class DummyObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def model_dump_json(self, indent=None):
        def to_plain(o):
            if isinstance(o, DummyObj):
                return {k: to_plain(v) for k, v in o.__dict__.items()}
            if isinstance(o, list):
                return [to_plain(x) for x in o]
            return o
        return json.dumps(to_plain(self), indent=indent)


@pytest.fixture(autouse=True)
def mock_client(monkeypatch):
    state = {"response": DummyObj(tools=[]), "get_response": None}

    class MockTools:
        def list(self):
            return state["response"]

        def get(self, tool_id: str):
            return state["get_response"]

    class MockConvAI:
        def __init__(self):
            self.tools = MockTools()

    def build_client():
        return DummyObj(conversational_ai=MockConvAI())

    import elevenlabs_mcp.server as srv
    monkeypatch.setattr(srv, "client", build_client())

    return state


def test_list_tools_empty_json_passthrough(mock_client):
    mock_client["response"] = DummyObj(tools=[])
    result = list_tools()
    data = json.loads(result.text)
    assert "tools" in data
    assert data["tools"] == []


def test_list_tools_mixed_types_json_passthrough(mock_client):
    # Minimal construction of three tools with distinct IDs
    webhook_tool = DummyObj(id="tool_webhook", tool_config=DummyObj(type="webhook", name="w"), access_info=DummyObj(role="admin"), usage_stats=DummyObj(total_calls=1))
    client_tool = DummyObj(id="tool_client", tool_config=DummyObj(type="client", name="c"), access_info=DummyObj(role="editor"), usage_stats=DummyObj(total_calls=2))
    system_tool = DummyObj(id="tool_system", tool_config=DummyObj(type="system", name="s"), access_info=DummyObj(role="viewer"), usage_stats=DummyObj(total_calls=3))

    mock_client["response"] = DummyObj(tools=[webhook_tool, client_tool, system_tool])

    result = list_tools()
    data = json.loads(result.text)

    assert "tools" in data
    assert isinstance(data["tools"], list)
    assert len(data["tools"]) == 3

    ids = {t["id"] for t in data["tools"]}
    assert ids == {"tool_webhook", "tool_client", "tool_system"}


def test_get_tool_json_passthrough(mock_client):
    api_schema = DummyObj(method="GET", url="https://example.com/test")
    webhook_cfg = DummyObj(type="webhook", name="first_tool", api_schema=api_schema)
    mock_client["get_response"] = DummyObj(
        id="tool_123",
        tool_config=webhook_cfg,
        access_info=DummyObj(role="admin"),
        usage_stats=DummyObj(total_calls=0),
    )

    result = get_tool("tool_123")
    data = json.loads(result.text)

    assert data["id"] == "tool_123"
    assert data["tool_config"]["type"] == "webhook"
    assert data["tool_config"]["name"] == "first_tool" 