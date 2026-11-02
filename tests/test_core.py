import asyncio
import json

import httpx
import pytest

from prophet_voice import core as core_
from prophet_voice.core import Core, CoreError

KEY = "pk_" + "a" * 43


def sse(*events) -> bytes:
    out = ": keepalive\n\n"
    for event_id, name, data in events:
        out += f"id: {event_id}\nevent: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    return out.encode()


def make(handler) -> Core:
    return Core("http://core:8770/", KEY, transport=httpx.MockTransport(handler))


def test_calls_carry_the_key_and_the_body():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"run_id": 7})

    run_id = asyncio.run(make(handler).chat("Привет", ""))
    assert run_id == 7
    r = seen[0]
    assert r.url == "http://core:8770/api/v1/chat"
    assert r.headers["Authorization"] == f"Bearer {KEY}"
    assert json.loads(r.content) == {"text": "Привет", "project": None}


def test_a_refusal_is_a_core_error_with_the_cores_words():
    async def go(status, body):
        return await make(lambda r: httpx.Response(status, json=body)).me()

    with pytest.raises(CoreError, match="does not know this key") as e:
        asyncio.run(go(401, {"detail": "client_key_required"}))
    assert e.value.status == 401
    with pytest.raises(CoreError, match='no "chat" scope') as e:
        asyncio.run(go(403, {"detail": 'this key has no "chat" scope'}))
    assert e.value.status == 403


def test_no_core_is_a_core_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(CoreError, match="does not answer"):
        asyncio.run(make(handler).me())


def test_events_follow_a_run_and_reconnect_where_they_were(monkeypatch):
    monkeypatch.setattr(core_, "RECONNECT_WAIT", 0)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:  # the connection breaks after one event
            return httpx.Response(
                200, content=sse(("1.0.3", "step", {"id": 3})) + b"id: 2.0.3\nevent: mess"
            )
        return httpx.Response(
            200,
            content=sse(
                ("2.0.3", "message", {"run_id": 5, "role": "assistant", "content": "Да."}),
                ("2.0.3", "run", {"run_id": 5, "status": "done"}),
            ),
        )

    async def collect():
        return [e async for e in make(handler).events(5)]

    events = asyncio.run(collect())
    assert [name for name, _ in events] == ["step", "message", "run"]
    assert calls[0].url.params["run"] == "5"
    assert "Last-Event-ID" not in calls[0].headers
    assert calls[1].headers["Last-Event-ID"] == "1.0.3"


def test_events_give_up_after_some_tries(monkeypatch):
    monkeypatch.setattr(core_, "RECONNECT_WAIT", 0)

    def handler(request):
        raise httpx.ReadError("broken")

    async def collect():
        return [e async for e in make(handler).events(5)]

    with pytest.raises(CoreError, match="lost the core"):
        asyncio.run(collect())


def test_a_stream_that_ends_before_the_run_is_followed_again(monkeypatch):
    monkeypatch.setattr(core_, "RECONNECT_WAIT", 0)
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:  # the core restarted: the stream closed, the run went on
            return httpx.Response(200, content=sse(("1.0.0", "run", {"status": "running"})))
        return httpx.Response(200, content=sse(("1.0.0", "run", {"status": "done"})))

    async def collect():
        return [d["status"] async for _, d in make(handler).events(5)]

    assert asyncio.run(collect()) == ["running", "done"]
    assert len(calls) == 2


def test_parse_joins_data_lines_and_skips_comments():
    async def lines():
        for line in [": hi", "id: 9", "event: question", 'data: {"id": 1,', 'data: "q": "?"}', ""]:
            yield line

    async def collect():
        return [e async for e in core_.parse(lines())]

    assert asyncio.run(collect()) == [("9", "question", {"id": 1, "q": "?"})]
