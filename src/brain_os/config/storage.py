"""Qdrant / Neo4j / Postgres / Redis / Mem0 connection settings."""

from __future__ import annotations

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from brain_os.config.common import _COMMON


class QdrantConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", **_COMMON)

    url: str = "http://localhost:6333"
    api_key: SecretStr = SecretStr("")
    # Authoritative KB (dense + sparse). Dense-only ``brain_knowledge_v3`` retired 2026-08-02.
    collection: str = "brain_knowledge_hybrid"
    # Request timeout in seconds (avoids indefinite hang when Qdrant is slow or unreachable)
    timeout: float = 30.0
    # Optional: second cluster used when the primary (url) is unreachable (connect/timeout).
    # Typical: QDRANT_URL=https://…cloud.qdrant.io, QDRANT_FALLBACK_URL=http://localhost:6333
    fallback_url: str = ""
    fallback_api_key: SecretStr = SecretStr("")
    # Optional: when set, every upsert and ensure_collection is mirrored to this
    # cluster so local and cloud stay in sync (dual-write).
    cloud_url: str = ""
    cloud_api_key: SecretStr = SecretStr("")

    # Alias of authoritative collection (kept for APP__USE_SPARSE_HYBRID wiring).
    collection_hybrid: str = "brain_knowledge_hybrid"
    #: Pitch media bank Phase 2 — Voyage multimodal vectors for image pick/place.
    collection_media: str = "brain_media_v1"


class Neo4jConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NEO4J_", **_COMMON)

    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: SecretStr = SecretStr("")
    auth: str = ""
    # Optional: mirror every graph write to this database (e.g. Neo4j Aura) while
    # keeping NEO4J_URI on local Docker. Set URI + cloud password or NEO4J_CLOUD_AUTH.
    cloud_uri: str = ""
    cloud_user: str = ""
    cloud_password: SecretStr = SecretStr("")
    cloud_auth: str = ""

    def resolved_auth(self) -> tuple[str, str]:
        """Resolve Neo4j credentials from explicit password or NEO4J_AUTH."""
        user = self.user.strip() or "neo4j"
        password = self.password.get_secret_value().strip()
        if password:
            return user, password

        auth = self.auth.strip()
        if "/" in auth:
            auth_user, auth_password = auth.split("/", 1)
            auth_user = auth_user.strip() or user
            auth_password = auth_password.strip()
            if auth_password:
                return auth_user, auth_password
        if "localhost" in self.uri or "127.0.0.1" in self.uri:
            # Local docker-compose default (safe dev fallback).
            return user, "brain_knowledge_graph"
        return user, ""

    def resolved_cloud_auth(self) -> tuple[str, str] | None:
        """Credentials for NEO4J_CLOUD_URI when dual-write is enabled.

        Returns ``None`` if ``cloud_uri`` is unset or no cloud password/auth is configured.
        """
        uri = self.cloud_uri.strip()
        if not uri:
            return None
        primary_user = self.user.strip() or "neo4j"
        user = self.cloud_user.strip() or primary_user
        pw = self.cloud_password.get_secret_value().strip()
        if pw:
            return user, pw
        ca = self.cloud_auth.strip()
        if "/" in ca:
            cu, cpw = ca.split("/", 1)
            cu = cu.strip() or user
            cpw = cpw.strip()
            if cpw:
                return cu, cpw
        return None


class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DATABASE_", **_COMMON)

    url: str = "postgresql+asyncpg://brain:brain@localhost:5432/brain_crm"


class RedisConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", **_COMMON)

    url: str = ""


class MemoryConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEM0_", **_COMMON)

    api_key: SecretStr = SecretStr("")
