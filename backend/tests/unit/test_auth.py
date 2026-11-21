"""Supabase token verification: fetching the project's signing keys (and refetching on
rotation), and the legacy shared-secret (HS256) projects."""

from __future__ import annotations

import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from app.auth import AuthError, SupabaseVerifier

URL = "https://proj.supabase.co"


def _jwk(key, kid: str) -> dict:
    return {**jwt.algorithms.ECAlgorithm.to_jwk(key.public_key(), as_dict=True), "kid": kid,
            "alg": "ES256", "use": "sig"}  # fmt: skip


def _token(key, kid: str, alg: str = "ES256", sub: str = "u1") -> str:
    now = int(time.time())
    claims = {"sub": sub, "aud": "authenticated", "iss": URL + "/auth/v1", "exp": now + 600}
    return jwt.encode(claims, key, algorithm=alg, headers={"kid": kid})


async def test_keys_are_fetched_once_and_refetched_when_rotated():
    old, new = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    served = {"keys": [_jwk(old, "old")]}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=served)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        v = SupabaseVerifier(URL, http=http)
        assert (await v.verify(_token(old, "old"))).owner == "u1"
        assert (await v.verify(_token(old, "old", sub="u2"))).owner == "u2"
        assert calls == [URL + "/auth/v1/.well-known/jwks.json"]  # cached

        served = {"keys": [_jwk(old, "old"), _jwk(new, "new")]}  # rotation
        assert (await v.verify(_token(new, "new"))).owner == "u1"
        assert len(calls) == 2
        with pytest.raises(AuthError):  # an unknown kid right after a refetch: no hammering
            await v.verify(_token(new, "other"))
        assert len(calls) == 2


async def test_unreachable_key_endpoint_is_an_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        v = SupabaseVerifier(URL, http=http)
        key = ec.generate_private_key(ec.SECP256R1())
        with pytest.raises(AuthError, match="signing keys"):
            await v.verify(_token(key, "k"))


async def test_legacy_shared_secret_projects():
    secret = "super-secret-jwt-token-with-at-least-32-characters"
    v = SupabaseVerifier(URL, jwt_secret=secret, jwks={"keys": []})
    assert (await v.verify(_token(secret, "x", alg="HS256"))).owner == "u1"
    with pytest.raises(AuthError):
        await v.verify(_token("another-secret-that-is-long-enough-32ch", "x", alg="HS256"))
    without = SupabaseVerifier(URL, jwks={"keys": []})
    with pytest.raises(AuthError, match="SUPABASE_JWT_SECRET"):
        await without.verify(_token(secret, "x", alg="HS256"))
