"""ASGI request-body size guard for API writes."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

ASGIMessage = dict[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]


class ApiBodySizeLimitMiddleware:
    """Reject API write bodies over a fixed limit before application parsing.

    The middleware buffers at most ``max_bytes`` and replays the accepted ASGI
    request messages unchanged to the application. Both declared and streamed
    body sizes are checked; Content-Length is an optimization, not the only
    enforcement mechanism.
    """

    def __init__(self, app, max_bytes: int = 16 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Receive, send: Send) -> None:
        if (scope.get("type") != "http" or not scope.get("path", "").startswith("/api/")
                or scope.get("method") not in {"POST", "PUT", "PATCH"}):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        raw_length = headers.get(b"content-length", b"")
        try:
            declared_length = int(raw_length)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > self.max_bytes:
            await self._too_large(send)
            return

        messages: list[ASGIMessage] = []
        body_size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            messages.append(message)
            if message["type"] == "http.request":
                body_size += len(message.get("body", b""))
                if body_size > self.max_bytes:
                    await self._too_large(send)
                    return
                if not message.get("more_body", False):
                    break

        replay_index = 0

        async def replay_receive() -> ASGIMessage:
            nonlocal replay_index
            if replay_index < len(messages):
                message = messages[replay_index]
                replay_index += 1
                return message
            return await receive()

        await self.app(scope, replay_receive, send)

    async def _too_large(self, send: Send) -> None:
        limit_mib = self.max_bytes / (1024 * 1024)
        label = f"{limit_mib:g} MiB" if limit_mib >= 1 else f"{self.max_bytes} bytes"
        body = json.dumps({"detail": f"Request body exceeds the {label} API limit."}).encode("utf-8")
        await send({"type": "http.response.start", "status": 413, "headers": [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]})
        await send({"type": "http.response.body", "body": body, "more_body": False})
