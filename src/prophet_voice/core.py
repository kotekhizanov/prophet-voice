"""Prophet's core, as the voice sees it: /api/v1 with the client key.

Every request carries `Authorization: Bearer <key>`. A refusal is a
CoreError with the core's own words ({"detail"}) and its status: 401 a wrong
or revoked key, 403 a missing scope, 404, 409, 422.

`events(run)` follows one run as Server-Sent Events: its steps, the
questions and messages that come meanwhile, and its status, until it ends.
When the connection breaks it reconnects with the last event's id
(Last-Event-ID), so nothing is missed or heard twice.
"""

import asyncio
import json
import logging

import httpx

log = logging.getLogger("prophet_voice.core")
RECONNECTS = 5  # tries in a row before a run is given up
RECONNECT_WAIT = 2  # seconds between them
FINAL = ("done", "failed", "cancelled", "paused")  # a run's statuses that end its events


class CoreError(Exception):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class Core:
    def __init__(self, url: str, key: str, transport: httpx.AsyncBaseTransport | None = None):
        self.http = httpx.AsyncClient(
            base_url=url.rstrip("/") + "/api/v1",
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(30, read=150),  # a read waits up to the core's 120 s
            transport=transport,
        )

    async def close(self) -> None:
        await self.http.aclose()

    async def _call(self, method: str, path: str, **kwargs):
        try:
            r = await self.http.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise CoreError(f"the core does not answer: {exc}") from exc
        if r.status_code >= 400:
            raise CoreError(_detail(r), r.status_code)
        return r.json()

    async def me(self) -> dict:
        """The key's name and scopes."""
        return await self._call("GET", "/me")

    async def plugin(self) -> dict | None:
        """{"plugin", "enabled", "settings"} when the key is a plugin's, else None
        (a key made by hand in Settings → Clients, or a core before plugins)."""
        try:
            return await self._call("GET", "/plugin/settings")
        except CoreError as exc:
            if exc.status == 404:
                return None
            raise

    async def chat(self, text: str, project: str | int | None = "") -> int:
        # The core takes a project as text, its name or number; Omniscient gives a number.
        body = {"text": text, "project": str(project) if project else None}
        return (await self._call("POST", "/chat", json=body))["run_id"]

    async def run(self, run_id: int, wait: int = 0) -> dict:
        return await self._call("GET", f"/runs/{run_id}", params={"wait": wait})

    async def cancel(self, run_id: int) -> None:
        await self._call("POST", f"/runs/{run_id}/cancel")

    async def answer(self, question_id: int, text: str) -> None:
        await self._call("POST", f"/questions/{question_id}/answer", json={"text": text})

    async def events(self, run_id: int):
        """(name, data) of the run's events until it ends."""
        last, tries = "", 0
        while True:
            headers = {"Last-Event-ID": last} if last else {}
            try:
                async with self.http.stream(
                    "GET", "/events", params={"run": run_id}, headers=headers
                ) as r:
                    if r.status_code >= 400:
                        await r.aread()
                        raise CoreError(_detail(r), r.status_code)
                    async for event_id, name, data in parse(r.aiter_lines()):
                        last, tries = event_id or last, 0
                        yield name, data
                        if name == "run" and data.get("status") in FINAL:
                            return  # the core ends the stream with the run
                raise httpx.ReadError("the stream ended before the run")  # the core restarted
            except (httpx.HTTPError, CoreError) as exc:
                if isinstance(exc, CoreError) and exc.status:  # a refusal, not a break
                    raise
                tries += 1
                if tries > RECONNECTS:
                    raise CoreError(f"lost the core while following run #{run_id}") from exc
                log.warning("events of run #%s broke (%s); reconnecting", run_id, exc)
                await asyncio.sleep(RECONNECT_WAIT)


async def parse(lines):
    """Server-Sent Events from lines: (id, name, data) for each event."""
    event_id, name, data = "", "message", []
    async for line in lines:
        if not line:
            if data:
                try:
                    yield event_id, name, json.loads("\n".join(data))
                except ValueError:
                    log.warning("an event that is not JSON: %r", data[:1])
            name, data = "message", []
            continue
        if line.startswith(":"):  # a keepalive
            continue
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "id":
            event_id = value
        elif field == "event":
            name = value
        elif field == "data":
            data.append(value)


def _detail(r: httpx.Response) -> str:
    try:
        detail = r.json().get("detail")
    except ValueError:
        detail = None
    if r.status_code == 401:
        return "the core does not know this key: make one in Settings → Clients"
    if isinstance(detail, list):  # FastAPI's validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or f"HTTP {r.status_code}")
