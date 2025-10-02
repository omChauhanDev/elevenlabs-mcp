import json
import pytest
from elevenlabs_mcp.server import list_tools, get_tool, delete_tool, get_dependent_agents
from elevenlabs_mcp.utils import ElevenLabsMcpError


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
    state = {
        "response": DummyObj(tools=[]),
        "get_response": None,
        "get_raises": False,
        "get_error": "tool_not_found",
        "delete_response": None,
        "delete_raises": False,
        "delete_error": "tool_not_found",
        "dep_agents_response": DummyObj(agents=[], next_cursor=None, has_more=False),
        "dep_agents_raises": False,
        "dep_agents_error": "tool_not_found",
    }

    class MockTools:
        def list(self):
            return state["response"]

        def get(self, tool_id: str):
            if state["get_raises"]:
                raise RuntimeError(state["get_error"])
            return state["get_response"]

        def delete(self, tool_id: str):
            if state["delete_raises"]:
                raise RuntimeError(state["delete_error"])
            return state["delete_response"]

        def get_dependent_agents(self, tool_id: str, cursor=None, page_size=None):
            if state["dep_agents_raises"]:
                raise RuntimeError(state["dep_agents_error"])
            return state["dep_agents_response"]

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


def test_get_tool_not_found_error(mock_client):
    mock_client["get_raises"] = True
    mock_client["get_error"] = "body: {'detail': {'status': 'tool_not_found', 'message': 'Tool with id tool_123 not found.'}}"
    with pytest.raises(ElevenLabsMcpError):
        get_tool("tool_123")


def test_delete_tool_success_message(mock_client):
    mock_client["delete_raises"] = False
    mock_client["delete_response"] = None
    tool_id = "tool_999"
    result = delete_tool(tool_id)
    assert result.text == f"Tool deleted successfully: {tool_id}"


def test_delete_tool_not_found_error(mock_client):
    mock_client["delete_raises"] = True
    mock_client["delete_error"] = "body: {'detail': {'status': 'tool_not_found', 'message': 'Tool with id tool_999 not found.'}}"
    with pytest.raises(ElevenLabsMcpError):
        delete_tool("tool_999")


def test_get_dependent_agents_empty(mock_client):
    mock_client["dep_agents_response"] = DummyObj(agents=[], next_cursor=None, has_more=False)
    result = get_dependent_agents("tool_good")
    data = json.loads(result.text)
    assert data["agents"] == []
    assert data["next_cursor"] is None
    assert data["has_more"] is False


def test_get_dependent_agents_populated(mock_client):
    agents = [
        DummyObj(
            id="agent_1",
            name="Jarvis",
            type="available",
            created_at_unix_secs=1759419452,
            access_level="admin",
        )
    ]
    mock_client["dep_agents_response"] = DummyObj(agents=agents, next_cursor=None, has_more=False)
    result = get_dependent_agents("tool_good")
    data = json.loads(result.text)
    assert len(data["agents"]) == 1
    assert data["agents"][0]["id"] == "agent_1"
    assert data["has_more"] is False


def test_get_dependent_agents_not_found(mock_client):
    mock_client["dep_agents_raises"] = True
    mock_client["dep_agents_error"] = "body: {'detail': {'status': 'tool_not_found', 'message': 'Tool with id tool_bad not found.'}}"
    with pytest.raises(ElevenLabsMcpError):
        get_dependent_agents("tool_bad") 