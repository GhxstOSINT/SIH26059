import asyncio
import json

from service.body_limit import ApiBodySizeLimitMiddleware


def run_asgi(app, scope, incoming):
    sent = []
    messages = iter(incoming)

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    asyncio.run(app(scope, receive, send))
    return sent


def request_scope(path="/api/v1/forecast", method="POST", headers=None):
    return {"type": "http", "path": path, "method": method,
            "headers": [(key.lower().encode(), value.encode()) for key, value in (headers or {}).items()]}


def test_body_limit_rejects_declared_oversize_without_reading_body():
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    response = run_asgi(ApiBodySizeLimitMiddleware(app, max_bytes=4),
                        request_scope(headers={"Content-Length": "5"}), [])

    assert not called
    assert response[0]["status"] == 413
    assert json.loads(response[1]["body"]) == {"detail": "Request body exceeds the 4 bytes API limit."}


def test_body_limit_rejects_oversize_chunked_body_before_dispatch():
    called = False

    async def app(scope, receive, send):
        nonlocal called
        called = True

    response = run_asgi(ApiBodySizeLimitMiddleware(app, max_bytes=4), request_scope(), [
        {"type": "http.request", "body": b"123", "more_body": True},
        {"type": "http.request", "body": b"45", "more_body": False},
    ])

    assert not called
    assert response[0]["status"] == 413


def test_body_limit_replays_allowed_body_unchanged():
    seen = bytearray()

    async def app(scope, receive, send):
        while True:
            message = await receive()
            seen.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    response = run_asgi(ApiBodySizeLimitMiddleware(app, max_bytes=4), request_scope(), [
        {"type": "http.request", "body": b"12", "more_body": True},
        {"type": "http.request", "body": b"34", "more_body": False},
    ])

    assert bytes(seen) == b"1234"
    assert response[0]["status"] == 204


def test_body_limit_does_not_restrict_non_api_or_read_only_requests():
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    huge = [{"type": "http.request", "body": b"12345", "more_body": False}]
    middleware = ApiBodySizeLimitMiddleware(app, max_bytes=4)
    assert run_asgi(middleware, request_scope(path="/api/v1/model", method="GET"), huge)[0]["status"] == 200
    assert run_asgi(middleware, request_scope(path="/healthz", method="POST"), huge)[0]["status"] == 200
