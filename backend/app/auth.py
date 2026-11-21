"""Supabase Auth: verify the access token the frontend sends, and say whose memory the
request is for (the token's `sub`, the Supabase user id).

Supabase projects sign tokens with asymmetric keys (ES256/RS256) published at
`<SUPABASE_URL>/auth/v1/.well-known/jwks.json`; those are fetched and cached for ten minutes,
and fetched again when a token names a key we haven't seen (key rotation). Projects still on
the legacy shared secret set SUPABASE_JWT_SECRET instead (HS256).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx
import jwt

JWKS_TTL_SECONDS = 600.0
JWKS_MIN_REFETCH_SECONDS = 30.0


class AuthError(Exception):
    """The token is missing, malformed, expired, or not signed by this Supabase project."""


@dataclass(frozen=True)
class Principal:
    owner: str  # Supabase user id, or 'local' with accounts off
    email: str | None = None
    service: bool = False  # the cron's service key: may act across accounts


class SupabaseVerifier:
    def __init__(
        self,
        supabase_url: str,
        *,
        audience: str = "authenticated",
        jwt_secret: str | None = None,
        jwks: dict[str, Any] | None = None,  # tests: a fixed key set instead of fetching
        http: httpx.AsyncClient | None = None,
    ) -> None:
        self.issuer = supabase_url.rstrip("/") + "/auth/v1"
        self.jwks_url = self.issuer + "/.well-known/jwks.json"
        self.audience = audience
        self._secret = jwt_secret
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at = 0.0
        self._forced_at = float("-inf")  # last refetch for an unknown key
        self._fixed = jwks is not None
        if jwks is not None:
            self._keys = self._parse(jwks)
            self._fetched_at = float("inf")
        self._http = http
        self._lock = asyncio.Lock()

    @staticmethod
    def _parse(jwks: dict[str, Any]) -> dict[str, jwt.PyJWK]:
        keys: dict[str, jwt.PyJWK] = {}
        for k in jwks.get("keys", []):
            if k.get("use", "sig") == "sig" and "kid" in k:
                keys[k["kid"]] = jwt.PyJWK(k)
        return keys

    async def _refresh(self, *, force: bool) -> None:
        if self._fixed:
            return
        async with self._lock:
            now = time.monotonic()
            if not force and now - self._fetched_at < JWKS_TTL_SECONDS:
                return
            if force:
                if now - self._forced_at < JWKS_MIN_REFETCH_SECONDS:
                    return  # an unknown key again so soon: don't hammer Supabase
                self._forced_at = now
            client = self._http or httpx.AsyncClient(timeout=10)
            try:
                r = await client.get(self.jwks_url)
                r.raise_for_status()
                self._keys = self._parse(r.json())
                self._fetched_at = time.monotonic()
            except (httpx.HTTPError, ValueError) as exc:
                raise AuthError(f"couldn't fetch Supabase signing keys: {exc}") from exc
            finally:
                if self._http is None:
                    await client.aclose()

    async def _key_for(self, header: dict[str, Any]) -> tuple[Any, str]:
        alg = header.get("alg")
        if alg == "HS256":
            if not self._secret:
                raise AuthError("HS256 token but SUPABASE_JWT_SECRET isn't set")
            return self._secret, alg
        if alg not in ("ES256", "RS256", "EdDSA"):
            raise AuthError(f"unsupported token algorithm {alg!r}")
        kid = header.get("kid")
        await self._refresh(force=False)
        if kid not in self._keys:
            await self._refresh(force=True)  # rotated key
        if kid not in self._keys:
            raise AuthError("token signed by an unknown key")
        return self._keys[kid].key, alg

    async def verify(self, token: str) -> Principal:
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthError("malformed token") from exc
        key, alg = await self._key_for(header)
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=[alg],
                audience=self.audience,
                issuer=self.issuer,
                options={"require": ["exp", "sub", "aud", "iss"]},
                leeway=30,
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"invalid token: {exc}") from exc
        if claims.get("role") == "anon" or not claims.get("sub"):
            raise AuthError("sign in first: this is an anonymous token")
        return Principal(owner=str(claims["sub"]), email=claims.get("email"))
