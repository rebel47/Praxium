from __future__ import annotations

from fastapi.testclient import TestClient

from praxium import (
    Agent,
    AgentRunner,
    Application,
    DeterministicModelProvider,
    GraphBuilder,
    Model,
    ModelCapabilities,
    ModelProviderRegistry,
    State,
    Team,
    TeamMember,
    TeamPolicy,
    TeamRole,
    TeamRunner,
    text_response,
)
from praxium.api import create_fastapi_app


async def output(_state: State, _context: object) -> dict[str, str]:
    return {"output": "graph answer"}


def _application() -> Application:
    provider = DeterministicModelProvider(responses=[text_response("agent answer")])
    runner = AgentRunner(ModelProviderRegistry([provider]))
    agent = Agent(
        name="assistant",
        instructions="Answer concisely.",
        model=Model(name="fake", provider="deterministic"),
    )
    graph = (
        GraphBuilder("workflow")
        .add_node("output", output)
        .set_entrypoint("output")
        .set_finish_point("output")
        .build()
    )
    return (
        Application(agent_runner=runner)
        .register(agent)
        .register(graph)
        .register_embedding_model(
            "embedding",
            Model(
                name="fake-embedding",
                provider="deterministic",
                capabilities=ModelCapabilities(embeddings=True),
            ),
        )
    )


def test_fastapi_generated_and_openai_compatible_routes() -> None:
    client = TestClient(create_fastapi_app(_application()))

    assert client.get("/health").json() == {"status": "ok"}
    components = client.get("/components").json()
    assert {item["name"] for item in components} == {"assistant", "embedding", "workflow"}

    run = client.post("/runs/workflow", json={"input": {"question": "hello"}})
    assert run.status_code == 200
    assert run.json()["state"]["data"]["output"] == "graph answer"

    chat = client.post(
        "/v1/chat/completions",
        json={"model": "assistant", "messages": [{"role": "user", "content": "hello"}]},
    )
    assert chat.status_code == 200
    assert chat.json()["choices"][0]["message"]["content"] == "agent answer"

    embedding = client.post(
        "/v1/embeddings",
        json={"model": "embedding", "input": ["hello", "world"]},
    )
    assert embedding.status_code == 200
    assert len(embedding.json()["data"]) == 2


def test_fastapi_streaming_responses_and_explicit_errors() -> None:
    client = TestClient(create_fastapi_app(_application()))

    assert client.get("/ready").json()["status"] == "ready"
    assert len(client.get("/v1/models").json()["data"]) == 3
    assert client.post("/runs/missing", json={"input": {}}).status_code == 404

    graph_stream = client.post("/runs/workflow/stream", json={"input": {}})
    assert graph_stream.status_code == 200
    assert "event: run.completed" in graph_stream.text

    chat_stream = client.post(
        "/v1/chat/completions",
        json={
            "model": "assistant",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )
    assert chat_stream.status_code == 200
    assert "chat.completion.chunk" in chat_stream.text
    assert "data: [DONE]" in chat_stream.text

    unsupported_tools = client.post(
        "/v1/chat/completions",
        json={
            "model": "assistant",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function"}],
        },
    )
    assert unsupported_tools.status_code == 400

    response = client.post(
        "/v1/responses",
        json={"model": "workflow", "input": "hello"},
    )
    assert response.status_code == 200
    assert response.json()["output"][0]["content"][0]["text"] == "graph answer"

    response_stream = client.post(
        "/v1/responses",
        json={"model": "workflow", "input": "hello", "stream": True},
    )
    assert "response.output_text.delta" in response_stream.text

    dimensions = client.post(
        "/v1/embeddings",
        json={"model": "embedding", "input": "hello", "dimensions": 3},
    )
    assert len(dimensions.json()["data"][0]["embedding"]) == 3


async def test_multi_agent_runner_enforces_explicit_team_order() -> None:
    provider = DeterministicModelProvider(
        responses=[text_response("research"), text_response("review")]
    )
    runner = AgentRunner(ModelProviderRegistry([provider]))
    model = Model(name="fake", provider="deterministic")
    team = Team(
        name="editorial",
        policy=TeamPolicy(max_rounds=1, max_delegations=2),
        members=[
            TeamMember(
                name="researcher",
                role=TeamRole.RESEARCHER,
                agent=Agent(name="researcher", instructions="Research.", model=model),
            ),
            TeamMember(
                name="reviewer",
                role=TeamRole.REVIEWER,
                agent=Agent(name="reviewer", instructions="Review.", model=model),
            ),
        ],
    )

    result = await TeamRunner(runner).run(team, "Write a note")

    assert [turn.member for turn in result.turns] == ["researcher", "reviewer"]
    assert result.final_response.text_content == "review"
