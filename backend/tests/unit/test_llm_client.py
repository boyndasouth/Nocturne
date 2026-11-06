import json

import httpx
import pytest

from app.llm.client import (
    AsyncRateLimiter,
    LLMError,
    OpenAICompatClient,
    RetryableLLMError,
    parse_retry_after,
)


def ok_body(text="hi", prompt_tokens=7, completion_tokens=3):
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class Sleeps:
    def __init__(self):
        self.calls: list[float] = []

    async def __call__(self, s: float) -> None:
        self.calls.append(s)


def make_client(responses, sleeps=None, max_retries=3):
    """`responses` is a list of httpx.Response or exceptions, returned in order."""
    seen: list[httpx.Request] = []
    it = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    client = OpenAICompatClient(
        name="test",
        base_url="https://llm.example/v1/",
        api_key="sk-test",
        model="m",
        rpm=0,
        max_retries=max_retries,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        sleep=sleeps or Sleeps(),
    )
    return client, seen


async def test_success_parses_text_and_usage_and_sends_auth():
    client, seen = make_client([httpx.Response(200, json=ok_body("hello"))])
    resp = await client.chat([{"role": "user", "content": "x"}], max_tokens=50, json_mode=True)
    assert resp.text == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
    req = seen[0]
    assert str(req.url) == "https://llm.example/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sk-test"
    body = json.loads(req.content)
    assert body["max_tokens"] == 50 and body["response_format"] == {"type": "json_object"}


async def test_429_honours_retry_after_then_succeeds():
    sleeps = Sleeps()
    client, seen = make_client(
        [
            httpx.Response(429, headers={"retry-after": "2"}, text="slow down"),
            httpx.Response(200, json=ok_body()),
        ],
        sleeps=sleeps,
    )
    resp = await client.chat([{"role": "user", "content": "x"}])
    assert resp.text == "hi" and len(seen) == 2
    assert len(sleeps.calls) == 1 and 2.0 <= sleeps.calls[0] <= 2.5


async def test_transport_errors_and_5xx_are_retried():
    client, seen = make_client(
        [
            httpx.ConnectError("boom"),
            httpx.Response(503, text="unavailable"),
            httpx.Response(200, json=ok_body("finally")),
        ]
    )
    resp = await client.chat([{"role": "user", "content": "x"}])
    assert resp.text == "finally" and len(seen) == 3


async def test_gives_up_after_max_retries():
    client, seen = make_client([httpx.Response(500)] * 10, max_retries=2)
    with pytest.raises(RetryableLLMError):
        await client.chat([{"role": "user", "content": "x"}])
    assert len(seen) == 3  # 1 attempt + 2 retries


async def test_4xx_is_not_retried():
    client, seen = make_client([httpx.Response(401, text="bad key")])
    with pytest.raises(LLMError) as exc_info:
        await client.chat([{"role": "user", "content": "x"}])
    assert not isinstance(exc_info.value, RetryableLLMError)
    assert len(seen) == 1


async def test_malformed_response_raises_llm_error():
    client, _ = make_client([httpx.Response(200, json={"choices": []})])
    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "x"}])


def test_parse_retry_after_reads_gemini_retry_delay_from_body():
    body = {"error": {"code": 429, "details": [{"retryDelay": "23s"}]}}
    assert parse_retry_after(httpx.Response(429, json=body)) == 23.0


def test_parse_retry_after_ignores_http_date_and_missing():
    assert parse_retry_after(httpx.Response(429, headers={"retry-after": "Wed, 21 Oct"})) is None
    assert parse_retry_after(httpx.Response(429)) is None


async def test_rate_limiter_spaces_requests():
    sleeps = Sleeps()
    limiter = AsyncRateLimiter(60, clock=lambda: 100.0, sleep=sleeps)  # 1 request / second
    waits = [await limiter.acquire() for _ in range(3)]
    assert waits == pytest.approx([0.0, 1.0, 2.0])
    assert sleeps.calls == pytest.approx([1.0, 2.0])


async def test_rate_limiter_disabled_when_rpm_is_zero():
    assert await AsyncRateLimiter(0).acquire() == 0.0


async def test_extra_params_are_sent_in_every_request():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=ok_body())

    client = OpenAICompatClient(
        name="groq", base_url="https://llm.example/v1", api_key="k", model="m", rpm=0,
        http=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        extra_params={"reasoning_effort": "low", "include_reasoning": False},
    )  # fmt: skip
    await client.chat([{"role": "user", "content": "x"}])
    body = json.loads(seen[0].content)
    assert body["reasoning_effort"] == "low" and body["include_reasoning"] is False


async def test_empty_answer_cut_off_by_max_tokens_is_an_error():
    body = ok_body(text="")
    body["choices"][0]["finish_reason"] = "length"
    client, seen = make_client([httpx.Response(200, json=body)])
    with pytest.raises(LLMError, match="max_tokens"):
        await client.chat([{"role": "user", "content": "x"}])
    assert len(seen) == 1  # not retried: the same request would fail the same way


def test_online_client_sends_reasoning_params_only_when_configured():
    from app.config import Settings
    from app.llm.providers import build_online_client
    from app.llm.tracing import Tracer

    base = {"database_url": "postgresql://x", "groq_api_key": "k", "_env_file": None}
    gpt_oss = build_online_client(Settings(**base), Tracer())
    assert gpt_oss.model == "openai/gpt-oss-120b"
    assert gpt_oss._extra == {"reasoning_effort": "low", "include_reasoning": False}

    llama = build_online_client(
        Settings(**base, online_model="some-llama", online_reasoning_effort=""), Tracer()
    )
    assert llama._extra == {}


GEMINI_DAILY = {
    "error": {
        "code": 429,
        "message": "You exceeded your current quota, please check your plan and billing details.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {
                "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                "violations": [
                    {
                        "quotaMetric": "generativelanguage.googleapis.com/"
                        "generate_content_free_tier_requests",
                        "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
                        "quotaValue": "20",
                    }
                ],
            },
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "43s"},
        ],
    }
}
GEMINI_MINUTE = json.loads(
    json.dumps(GEMINI_DAILY).replace("PerDay", "PerMinute").replace('"20"', '"10"')
)


def test_daily_quota_detection():
    from app.llm.client import daily_quota_detail

    detail = daily_quota_detail(json.dumps(GEMINI_DAILY))
    assert detail == "quotaId=GenerateRequestsPerDayPerProjectPerModel-FreeTier, quotaValue=20"
    assert daily_quota_detail(json.dumps(GEMINI_MINUTE)) is None
    groq = (
        "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000"
    )
    assert "tokens per day (TPD)" in daily_quota_detail(groq)
    assert daily_quota_detail("Rate limit exceeded: free-models-per-day") is not None
    assert daily_quota_detail("Rate limit reached on requests per minute (RPM)") is None


async def test_daily_quota_429_fails_fast_without_retrying():
    from app.llm.client import QuotaExhaustedError

    sleeps = Sleeps()
    client, seen = make_client(
        [httpx.Response(429, json=GEMINI_DAILY), httpx.Response(200, json=ok_body())], sleeps
    )
    with pytest.raises(QuotaExhaustedError, match="quotaValue=20"):
        await client.chat([{"role": "user", "content": "x"}])
    assert len(seen) == 1 and sleeps.calls == []


async def test_per_minute_429_is_still_retried():
    client, seen = make_client(
        [httpx.Response(429, json=GEMINI_MINUTE), httpx.Response(200, json=ok_body())], Sleeps()
    )
    assert (await client.chat([{"role": "user", "content": "x"}])).text == "hi"
    assert len(seen) == 2
