"""Centralised LLM client wrapping OpenAI, Anthropic, and optional Ollama (local).

Cloud OpenAI calls use the ``langfuse.openai`` wrapper for Langfuse tracing.
Anthropic uses the standard SDK with manual ``@observe()`` tracing.
Ollama uses the stock ``openai.AsyncOpenAI`` against a local ``/v1`` base URL
(no Helicone/Langfuse wrapper).

Structured output uses `instructor <https://python.useinstructor.com/>`_
which sends Pydantic validation errors back to the LLM for automatic
correction, dramatically reducing parse failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from typing import Any, TypeVar

import httpx
import instructor
from anthropic import AsyncAnthropic
from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI
from openai import AsyncOpenAI as OpenAICompatAsyncOpenAI
from pydantic import BaseModel

from brain_os.config import Settings, get_settings
from brain_os.services.resilience import CircuitBreaker, RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_MAX_RETRIES = 3
_RETRY_BACKOFF = 2.0
_MAX_DELAY_SECONDS = 10.0
_STRUCTURED_DEFAULT_USER_CHARS = 12_000


def _status_code(exc: Exception) -> int | None:
    return getattr(getattr(exc, "response", None), "status_code", None)


def _clip_user_for_structured(user: str, max_user_chars: int) -> str:
    """Bound structured-call user payloads (stability vs provider limits)."""
    if max_user_chars <= 0 or len(user) <= max_user_chars:
        return user
    logger.warning(
        "Structured LLM user message truncated (%d → %d chars); increase max_user_chars or batch "
        "the input (e.g. Dream mode episodic JSON).",
        len(user),
        max_user_chars,
    )
    return user[:max_user_chars]


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Return True for rate limits (429), server errors (5xx), connection errors; False for 401/400."""
    status = _status_code(exc)
    if status is not None and status in (401, 400):
        return False
    return status in (402, 429) or status is None or (status is not None and status >= 500)


_MAX_HELICONE_HEADER_VALUE_LEN = 256


def _normalize_ollama_base_url(raw: str) -> str:
    """Ensure Ollama OpenAI-compatible base URL ends with ``/v1``.

    Replaces ``localhost`` / ``[::1]`` with ``127.0.0.1`` so we hit the IPv4
    listener: many Ollama builds bind ``0.0.0.0:11434`` only, while ``localhost``
    can resolve to ``::1`` and fail with connection refused / connection errors.
    """
    u = raw.strip().rstrip("/")
    if not u:
        return ""
    if "://localhost" in u:
        u = u.replace("://localhost", "://127.0.0.1", 1)
    u = re.sub(r"(?i)://\[::1\]", "://127.0.0.1", u)
    if not u.endswith("/v1"):
        return f"{u}/v1"
    return u


def _helicone_session_user_headers(
    *,
    session_id: str | None,
    user_id: str | None,
) -> dict[str, str]:
    """Build Helicone session/user headers (values clipped for safe HTTP headers)."""
    headers: dict[str, str] = {}
    if session_id is not None and (sid := session_id.strip()):
        headers["Helicone-Session-Id"] = sid[:_MAX_HELICONE_HEADER_VALUE_LEN]
    if user_id is not None and (uid := user_id.strip()):
        headers["Helicone-User-Id"] = uid[:_MAX_HELICONE_HEADER_VALUE_LEN]
    return headers


class LLMClient:
    """Unified async client for OpenAI, Anthropic, and optional local Ollama."""

    def __init__(self, settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        openai_key = cfg.llm.openai_api_key.get_secret_value()
        anthropic_key = cfg.llm.anthropic_api_key.get_secret_value()
        helicone_key = cfg.helicone.api_key.get_secret_value()

        openai_kwargs: dict[str, Any] = {"api_key": openai_key}
        anthropic_kwargs: dict[str, Any] = {"api_key": anthropic_key}

        self._helicone = bool(helicone_key.strip())
        proxy_anthropic = cfg.app.helicone_proxy_anthropic
        if helicone_key:
            _h_auth = {"Helicone-Auth": f"Bearer {helicone_key}"}
            _hc = cfg.helicone
            if pe := str(_hc.property_environment).strip():
                _h_auth["Helicone-Property-Environment"] = pe[:_MAX_HELICONE_HEADER_VALUE_LEN]
            if pa := str(_hc.property_app).strip():
                _h_auth["Helicone-Property-App"] = pa[:_MAX_HELICONE_HEADER_VALUE_LEN]
            openai_kwargs["base_url"] = "https://oai.helicone.ai/v1"
            openai_kwargs["default_headers"] = _h_auth
            if proxy_anthropic:
                anthropic_kwargs["base_url"] = "https://anthropic.helicone.ai/v1"
                anthropic_kwargs["default_headers"] = dict(_h_auth)
            logger.info(
                "Helicone proxy enabled for OpenAI%s",
                " and Anthropic" if proxy_anthropic else "; Anthropic uses direct API",
            )

        self._openai: AsyncOpenAI | None = AsyncOpenAI(**openai_kwargs) if openai_key else None
        self._anthropic: AsyncAnthropic | None = (
            AsyncAnthropic(**anthropic_kwargs) if anthropic_key else None
        )

        self._openai_instructor = instructor.from_openai(self._openai) if self._openai else None
        self._anthropic_instructor = (
            instructor.from_anthropic(self._anthropic) if self._anthropic else None
        )

        ollama_raw = (cfg.llm.ollama_base_url or "").strip()
        ollama_base = _normalize_ollama_base_url(ollama_raw) if ollama_raw else ""
        ollama_key = cfg.llm.ollama_api_key.get_secret_value().strip()
        if not ollama_key:
            ollama_key = "ollama"
        self._ollama_http_client: httpx.AsyncClient | None = None
        if ollama_base:
            read_s = float(cfg.llm.ollama_http_timeout_seconds)
            conn_s = float(cfg.llm.ollama_connect_timeout_seconds)
            _httpx_timeout = httpx.Timeout(
                connect=conn_s,
                read=read_s,
                write=min(120.0, read_s),
                pool=15.0,
            )
            self._ollama_http_client = httpx.AsyncClient(
                timeout=_httpx_timeout,
                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
            )
            if "localhost" in ollama_raw.lower() or "[::1]" in ollama_raw.lower():
                logger.info(
                    "Ollama base URL normalized for IPv4 loopback: %s -> %s",
                    ollama_raw,
                    ollama_base,
                )
        self._ollama_openai: OpenAICompatAsyncOpenAI | None = (
            OpenAICompatAsyncOpenAI(
                base_url=ollama_base,
                api_key=ollama_key,
                http_client=self._ollama_http_client,
            )
            if ollama_base and self._ollama_http_client is not None
            else None
        )
        self._ollama_instructor = (
            instructor.from_openai(self._ollama_openai) if self._ollama_openai else None
        )
        self._ollama_model = cfg.llm.ollama_model

        self._openai_model = cfg.llm.openai_model
        self._anthropic_model = cfg.llm.anthropic_model
        _am = cfg.llm.anthropic_model
        _af = (cfg.llm.brain_anthropic_model_fast or "").strip() or _am
        _ar = (cfg.llm.brain_anthropic_model_reasoning or "").strip() or _am
        _aw = (cfg.llm.brain_anthropic_model_writing or "").strip() or _am
        _av = (cfg.llm.brain_anthropic_model_verifier or "").strip() or _am
        self._anthropic_model_profiles = {
            "default": _am,
            "fast": _af,
            "extract": _af,
            "extraction": _af,
            "classification": _af,
            "reasoning": _ar,
            "synthesis": _ar,
            "writing": _aw,
            "drafting": _aw,
            "verifier": _av,
            "verification": _av,
            "faithfulness": _av,
        }
        self._model_profiles = {
            "default": cfg.llm.openai_model,
            "fast": cfg.llm.brain_model_fast,
            "extract": cfg.llm.brain_model_fast,
            "extraction": cfg.llm.brain_model_fast,
            "classification": cfg.llm.brain_model_fast,
            "reasoning": cfg.llm.brain_model_reasoning,
            "synthesis": cfg.llm.brain_model_reasoning,
            "writing": cfg.llm.brain_model_writing,
            "drafting": cfg.llm.brain_model_writing,
            "verifier": cfg.llm.brain_model_verifier,
            "verification": cfg.llm.brain_model_verifier,
            "faithfulness": cfg.llm.brain_model_verifier,
        }

        self._semaphore = asyncio.Semaphore(10)
        self._retry_policy = RetryPolicy(
            max_attempts=_MAX_RETRIES,
            base_delay_seconds=_RETRY_BACKOFF,
            max_delay_seconds=_MAX_DELAY_SECONDS,
        )
        self._openai_breaker = CircuitBreaker(threshold=10, window_seconds=180)
        self._anthropic_breaker = CircuitBreaker(threshold=10, window_seconds=180)
        self._ollama_breaker = CircuitBreaker(threshold=10, window_seconds=180)
        self._redis_cache: Any = None

    def is_provider_configured(self, provider: str) -> bool:
        """Return True when *provider* has a live client (keys / URL set)."""
        p = provider.strip().lower()
        if p == "openai":
            return self._openai is not None
        if p == "anthropic":
            return self._anthropic is not None
        if p == "ollama":
            return self._ollama_openai is not None
        return False

    def _provider_fallback_chain(self, primary: str) -> list[str]:
        """Rotate ``ollama`` → ``openai`` → ``anthropic`` so *primary* is first; drop unavailable."""
        base = ["ollama", "openai", "anthropic"]
        p = primary.strip().lower()
        if p not in base:
            p = "openai"
        idx = base.index(p)
        rotated = base[idx:] + base[:idx]
        return [x for x in rotated if self.is_provider_configured(x)]

    def set_redis_cache(self, cache: Any) -> None:
        """Inject Redis cache for semantic response caching (optional)."""
        self._redis_cache = cache

    def _resolve_model(
        self,
        provider: str,
        model: str | None,
        model_profile: str | None,
    ) -> str:
        """Resolve explicit model/profile settings into a concrete model name."""
        if model:
            return model
        prov = provider.strip().lower()
        if prov == "ollama":
            return self._ollama_model
        if prov != "openai":
            if not model_profile:
                return self._anthropic_model
            normalized = model_profile.strip().lower().replace("-", "_")
            return self._anthropic_model_profiles.get(normalized, self._anthropic_model)
        if not model_profile:
            return self._openai_model
        normalized = model_profile.strip().lower().replace("-", "_")
        return self._model_profiles.get(normalized, self._openai_model)

    @staticmethod
    def _openai_token_kwargs(model: str, max_tokens: int) -> dict[str, int]:
        """OpenAI newer reasoning models use max_completion_tokens."""
        if model.startswith("gpt-5"):
            return {"max_completion_tokens": max_tokens}
        return {"max_tokens": max_tokens}

    @staticmethod
    def _openai_temperature_kwargs(model: str, temperature: float) -> dict[str, float]:
        """OpenAI GPT-5-class models only accept the default temperature."""
        if model.startswith("gpt-5"):
            return {}
        return {"temperature": temperature}

    @staticmethod
    def _llm_cache_key(
        system: str,
        user: str,
        model: str,
        temperature: float,
        extra: str = "",
    ) -> str:
        raw = f"{system[:2000]}|{user[:4000]}|{model}|{temperature:.2f}|{extra}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    _LLM_CACHE_TTL = 86400  # 24 hours

    # ── structured output (JSON → Pydantic) ──────────────────────────────

    async def generate_structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        provider: str = "openai",
        model: str | None = None,
        model_profile: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> T:
        """Call an LLM and parse the response into a Pydantic model.

        Uses Instructor to handle structured output with automatic
        validation-retry feedback for both OpenAI and Anthropic.

        ``max_user_chars`` truncates the *user* message before sending (default matches
        historic behaviour). Heavy callers such as Dream mode should pass a larger value
        so JSON / episode payloads are not cut mid-structure.
        """
        chain = self._provider_fallback_chain(provider)
        if not chain:
            return response_model()
        empty = response_model()
        async with self._semaphore:
            for i, prov in enumerate(chain):
                try:
                    result = await self._generate_structured_inner(
                        system,
                        user,
                        response_model,
                        provider=prov,
                        model=model if i == 0 else None,
                        model_profile=model_profile if i == 0 else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_user_chars=max_user_chars,
                        name=name,
                        session_id=session_id,
                        user_id=user_id,
                    )
                except (
                    Exception
                ):  # intentional broad catch — provider fallback chain; raises if last
                    if i + 1 < len(chain):
                        logger.info("Falling back to %s for structured output", chain[i + 1])
                        continue
                    raise
                if result != empty:
                    return result
                if i + 1 < len(chain):
                    logger.info("Falling back to %s for structured output", chain[i + 1])
        return empty

    async def _generate_structured_inner(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        provider: str = "openai",
        model: str | None = None,
        model_profile: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> T:
        resolved_model = self._resolve_model(provider, model, model_profile)
        if self._redis_cache and temperature <= 0.2:
            try:
                cache_key = self._llm_cache_key(
                    system,
                    user,
                    resolved_model,
                    temperature,
                    extra=f"structured:{response_model.__name__}",
                )
                cached = await self._redis_cache.get_llm_cache(cache_key)
                if cached is not None:
                    data = json.loads(cached)
                    return response_model.model_validate(data)
            except (json.JSONDecodeError, Exception):
                pass
        prov = provider.strip().lower()
        if prov == "openai":
            result = await self._openai_structured(
                system,
                user,
                response_model,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_user_chars=max_user_chars,
                name=name,
                session_id=session_id,
                user_id=user_id,
            )
        elif prov == "ollama":
            result = await self._ollama_structured(
                system,
                user,
                response_model,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_user_chars=max_user_chars,
                name=name,
            )
        else:
            result = await self._anthropic_structured(
                system,
                user,
                response_model,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_user_chars=max_user_chars,
                name=name,
                session_id=session_id,
                user_id=user_id,
            )
        if self._redis_cache and temperature <= 0.2:
            try:
                cache_key = self._llm_cache_key(
                    system,
                    user,
                    resolved_model,
                    temperature,
                    extra=f"structured:{response_model.__name__}",
                )
                await self._redis_cache.set_llm_cache(
                    cache_key,
                    json.dumps(result.model_dump(), default=str),
                    ttl_seconds=self._LLM_CACHE_TTL,
                )
            except Exception:  # intentional broad catch — best-effort LLM cache I/O
                pass
        return result

    async def _openai_structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> T:
        if self._openai_instructor is None:
            return response_model()

        resolved_model = model or self._openai_model
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id

        helicone_headers = (
            _helicone_session_user_headers(session_id=session_id, user_id=user_id)
            if self._helicone
            else {}
        )

        clipped_user = _clip_user_for_structured(user, max_user_chars)

        async def _operation() -> T:
            try:
                return await self._openai_instructor.chat.completions.create(
                    model=resolved_model,
                    response_model=response_model,
                    max_retries=2,
                    **self._openai_temperature_kwargs(resolved_model, temperature),
                    **self._openai_token_kwargs(resolved_model, max_tokens),
                    timeout=120.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": clipped_user},
                    ],
                    **({"name": name} if name else {}),
                    **({"metadata": metadata} if metadata else {}),
                    **({"extra_headers": helicone_headers} if helicone_headers else {}),
                )
            except Exception as exc:  # intentional broad catch — re-raised after log
                logger.warning("OpenAI structured attempt failed: %s", exc)
                raise

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._openai_breaker,
            )
        except Exception as exc:  # intentional broad catch — re-raised after log
            logger.warning(
                "OpenAI structured call failed after retries: %s",
                exc,
                exc_info=True,
            )
            raise

    async def _ollama_structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        name: str | None = None,
    ) -> T:
        if self._ollama_instructor is None:
            return response_model()

        _ = name
        resolved_model = model or self._ollama_model
        clipped_user = _clip_user_for_structured(user, max_user_chars)

        async def _operation() -> T:
            try:
                return await self._ollama_instructor.chat.completions.create(
                    model=resolved_model,
                    response_model=response_model,
                    max_retries=2,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=120.0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": clipped_user},
                    ],
                )
            except Exception as exc:  # intentional broad catch — re-raised after log
                logger.warning("Ollama structured attempt failed: %s", exc)
                raise

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._ollama_breaker,
            )
        except Exception as exc:  # intentional broad catch — re-raised after log
            logger.warning(
                "Ollama structured call failed after retries: %s",
                exc,
                exc_info=True,
            )
            raise

    @observe(name="anthropic_structured")
    async def _anthropic_structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        model: str | None = None,
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> T:
        if self._anthropic_instructor is None:
            return response_model()

        resolved_model = model or self._anthropic_model

        helicone_headers = (
            _helicone_session_user_headers(session_id=session_id, user_id=user_id)
            if self._helicone
            else {}
        )

        clipped_user = _clip_user_for_structured(user, max_user_chars)

        async def _operation() -> T:
            try:
                return await self._anthropic_instructor.messages.create(
                    model=resolved_model,
                    response_model=response_model,
                    max_retries=2,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=120.0,
                    system=system,
                    messages=[{"role": "user", "content": clipped_user}],
                    **({"extra_headers": helicone_headers} if helicone_headers else {}),
                )
            except Exception as exc:  # intentional broad catch — re-raised after log
                logger.warning("Anthropic structured attempt failed: %s", exc)
                raise

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._anthropic_breaker,
            )
        except Exception as exc:  # intentional broad catch — re-raised after log
            logger.warning(
                "Anthropic structured call failed after retries: %s",
                exc,
                exc_info=True,
            )
            raise

    # ── plain text output ─────────────────────────────────────────────────

    async def generate_text(
        self,
        system: str,
        user: str,
        *,
        provider: str = "openai",
        model: str | None = None,
        model_profile: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Call an LLM and return the raw text response."""
        chain = self._provider_fallback_chain(provider)
        if not chain:
            return "(No LLM provider configured)"
        last = ""
        async with self._semaphore:
            for i, prov in enumerate(chain):
                result = await self._generate_text_inner(
                    system,
                    user,
                    provider=prov,
                    model=model if i == 0 else None,
                    model_profile=model_profile if i == 0 else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                )
                if not result.startswith("("):
                    return result
                last = result
                if i + 1 < len(chain):
                    logger.info("Falling back to %s after %s failure", chain[i + 1], prov)
        return last

    async def _generate_text_inner(
        self,
        system: str,
        user: str,
        *,
        provider: str = "openai",
        model: str | None = None,
        model_profile: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        resolved_model = self._resolve_model(provider, model, model_profile)
        if self._redis_cache and temperature <= 0.2:
            try:
                cache_key = self._llm_cache_key(
                    system, user, resolved_model, temperature, extra="text"
                )
                cached = await self._redis_cache.get_llm_cache(cache_key)
                if cached is not None:
                    return cached
            except Exception:  # intentional broad catch — best-effort LLM cache I/O
                pass
        prov = provider.strip().lower()
        if prov == "openai":
            result = await self._openai_text(
                system,
                user,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                name=name,
                session_id=session_id,
                user_id=user_id,
            )
        elif prov == "ollama":
            result = await self._ollama_text(
                system,
                user,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        else:
            result = await self._anthropic_text(
                system,
                user,
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                name=name,
                session_id=session_id,
                user_id=user_id,
            )
        if self._redis_cache and temperature <= 0.2 and not result.startswith("("):
            try:
                cache_key = self._llm_cache_key(
                    system, user, resolved_model, temperature, extra="text"
                )
                await self._redis_cache.set_llm_cache(
                    cache_key,
                    result,
                    ttl_seconds=self._LLM_CACHE_TTL,
                )
            except Exception:  # intentional broad catch — best-effort LLM cache I/O
                pass
        return result

    async def _openai_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        if self._openai is None:
            return "(No OpenAI key configured)"

        resolved_model = model or self._openai_model
        metadata: dict[str, Any] = {}
        if session_id:
            metadata["langfuse_session_id"] = session_id
        if user_id:
            metadata["langfuse_user_id"] = user_id

        helicone_headers = (
            _helicone_session_user_headers(session_id=session_id, user_id=user_id)
            if self._helicone
            else {}
        )

        async def _operation() -> str:
            resp = await self._openai.chat.completions.create(
                model=resolved_model,
                **self._openai_temperature_kwargs(resolved_model, temperature),
                **self._openai_token_kwargs(resolved_model, max_tokens),
                timeout=120.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:12_000]},
                ],
                **({"name": name} if name else {}),
                **({"metadata": metadata} if metadata else {}),
                **({"extra_headers": helicone_headers} if helicone_headers else {}),
            )
            return resp.choices[0].message.content or ""

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._openai_breaker,
            )
        except Exception as exc:  # intentional broad catch — duck-typed status across SDKs + httpx
            status = _status_code(exc)
            if status in (429, 402):
                logger.warning("OpenAI %d — quota/rate limit", status)
                return "(OpenAI quota/rate limit exceeded)"
            if status and status < 500:
                logger.warning("OpenAI %d error: %s", status, exc)
                return "(LLM call failed)"
            logger.warning("OpenAI text retry exhausted: %s", exc)
            return "(LLM call failed after 3 retries)"

    async def _ollama_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        if self._ollama_openai is None:
            return "(No Ollama server configured)"

        resolved_model = model or self._ollama_model

        async def _operation() -> str:
            resp = await self._ollama_openai.chat.completions.create(
                model=resolved_model,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=120.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user[:12_000]},
                ],
            )
            return resp.choices[0].message.content or ""

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._ollama_breaker,
            )
        except Exception as exc:  # intentional broad catch — duck-typed status across SDKs + httpx
            status = _status_code(exc)
            if status in (429, 402):
                logger.warning("Ollama %d — rate limit", status)
                return "(Ollama rate limit exceeded)"
            if status and status < 500:
                logger.warning("Ollama %d error: %s", status, exc)
                return "(LLM call failed)"
            logger.warning("Ollama text retry exhausted: %s", exc)
            return "(LLM call failed after 3 retries)"

    @observe(name="anthropic_text")
    async def _anthropic_text(
        self,
        system: str,
        user: str,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        if self._anthropic is None:
            return "(No Anthropic key configured)"

        resolved_model = model or self._anthropic_model

        helicone_headers = (
            _helicone_session_user_headers(session_id=session_id, user_id=user_id)
            if self._helicone
            else {}
        )

        async def _operation() -> str:
            resp = await self._anthropic.messages.create(
                model=resolved_model,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=120.0,
                system=system,
                messages=[{"role": "user", "content": user[:12_000]}],
                **({"extra_headers": helicone_headers} if helicone_headers else {}),
            )
            return resp.content[0].text

        try:
            return await run_with_retry(
                _operation,
                policy=self._retry_policy,
                is_retryable=_is_retryable_llm_error,
                circuit_breaker=self._anthropic_breaker,
            )
        except Exception as exc:  # intentional broad catch — duck-typed status across SDKs + httpx
            status = _status_code(exc)
            if status in (429, 402):
                logger.warning("Anthropic %d — quota/rate limit", status)
                return "(Anthropic quota/rate limit exceeded)"
            if status and status < 500:
                logger.warning("Anthropic %d error: %s", status, exc)
                return "(LLM call failed)"
            logger.warning("Anthropic text retry exhausted: %s", exc)
            return "(LLM call failed after 3 retries)"

    # ── fallback wrappers ─────────────────────────────────────────────────

    async def generate_text_with_fallback(
        self,
        system: str,
        user: str,
        *,
        primary: str = "openai",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        model: str | None = None,
        model_profile: str | None = None,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> str:
        """Try *primary* then other configured providers (Ollama → OpenAI → Anthropic order)."""
        chain = self._provider_fallback_chain(primary)
        if not chain:
            return "(No LLM provider configured)"
        last = ""
        async with self._semaphore:
            for i, prov in enumerate(chain):
                result = await self._generate_text_inner(
                    system,
                    user,
                    provider=prov,
                    model=model if i == 0 else None,
                    model_profile=model_profile if i == 0 else None,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    name=name,
                    session_id=session_id,
                    user_id=user_id,
                )
                if not result.startswith("("):
                    return result
                last = result
                if i + 1 < len(chain):
                    logger.info("Falling back to %s after %s failure", chain[i + 1], prov)
        return last

    async def generate_structured_with_fallback(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        primary: str = "openai",
        temperature: float = 0,
        max_tokens: int = 4096,
        max_user_chars: int = _STRUCTURED_DEFAULT_USER_CHARS,
        model: str | None = None,
        model_profile: str | None = None,
        name: str | None = None,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> T:
        """Try *primary* then other configured providers until structured output validates."""
        chain = self._provider_fallback_chain(primary)
        if not chain:
            return response_model()
        empty = response_model()
        async with self._semaphore:
            for i, prov in enumerate(chain):
                try:
                    result = await self._generate_structured_inner(
                        system,
                        user,
                        response_model,
                        provider=prov,
                        model=model if i == 0 else None,
                        model_profile=model_profile if i == 0 else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        max_user_chars=max_user_chars,
                        name=name,
                        session_id=session_id,
                        user_id=user_id,
                    )
                except (
                    Exception
                ):  # intentional broad catch — provider fallback chain; raises if last
                    if i + 1 < len(chain):
                        logger.info("Falling back to %s for structured output", chain[i + 1])
                        continue
                    raise
                if result != empty:
                    return result
                if i + 1 < len(chain):
                    logger.info("Falling back to %s for structured output", chain[i + 1])
        return empty


# ── singleton ─────────────────────────────────────────────────────────────

_client: LLMClient | None = None
_client_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """Return the global LLMClient singleton (created on first call)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = LLMClient()
    return _client


def reset_llm_client() -> None:
    """Reset the singleton — useful for tests."""
    global _client
    with _client_lock:
        _client = None


def reset_llm_circuit_breakers() -> None:
    """Clear LLM circuit breakers so the next request is attempted.
    Use after reloading the OpenAI wallet or when starting Brain OS so prior 429s don't block."""
    client = get_llm_client()
    client._openai_breaker.reset()
    client._anthropic_breaker.reset()
    client._ollama_breaker.reset()


def get_llm_circuit_breaker_snapshots() -> list[dict[str, Any]]:
    """Return non-secret state for the three LLM circuit breakers.

    Phase-2 hardening: surfaces in ``GET /api/deep-health`` under
    ``services.llm_circuit_breakers`` so operators can tell at a glance
    whether OpenAI / Anthropic / Ollama is currently blocked, why, and
    how long ago the last failure landed. See
    ``docs/PHASE2_HARDENING_AUDIT.md`` §4.4 and
    ``docs/TROUBLESHOOTING.md`` "LLM circuit breaker open".
    """
    try:
        client = get_llm_client()
    except Exception:  # intentional broad catch — return [] if client unconfigured
        return []
    snapshots: list[dict[str, Any]] = []
    for name, breaker in (
        ("openai", getattr(client, "_openai_breaker", None)),
        ("anthropic", getattr(client, "_anthropic_breaker", None)),
        ("ollama", getattr(client, "_ollama_breaker", None)),
    ):
        if breaker is None:
            continue
        try:
            snap = breaker.public_snapshot()
        except Exception:  # pragma: no cover — defensive
            continue
        snap["name"] = name
        snapshots.append(snap)
    return snapshots
