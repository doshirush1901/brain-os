"""Canonical string constants for the service locator dictionaries.

Using these constants instead of bare strings prevents silent
misconfiguration from typos and enables IDE autocompletion.
"""

from __future__ import annotations


class ServiceKey:
    """Known service keys used across inject_services / pipeline context."""

    CRM = "crm"
    QUOTES = "quotes"
    PRICING_ENGINE = "pricing_engine"
    RETRIEVER = "retriever"
    QDRANT = "qdrant"
    GRAPH = "graph"
    LONG_TERM_MEMORY = "long_term_memory"
    MEMORY_BLOCK_STORE = "memory_block_store"
    EPISODIC_MEMORY = "episodic_memory"
    CONVERSATION_MEMORY = "conversation_memory"
    RELATIONSHIP_MEMORY = "relationship_memory"
    GOAL_MANAGER = "goal_manager"
    EMOTIONAL_INTELLIGENCE = "emotional_intelligence"
    PROCEDURAL_MEMORY = "procedural_memory"
    LEARNING_HUB = "learning_hub"
    DATA_EVENT_BUS = "data_event_bus"
    PANTHEON = "pantheon"
    ENDOCRINE = "endocrine"
    CIRCULATORY = "circulatory"
    DIGESTIVE = "digestive"
    IMMUNE = "immune"
    SENSORY = "sensory"
    VOICE = "voice"
    MUSCULOSKELETAL = "musculoskeletal"
    RESPIRATORY = "respiratory"
    DREAM_MODE = "dream_mode"
    DRIP_ENGINE = "drip_engine"
    EMAIL_PROCESSOR = "email_processor"
    MEM0_CLIENT = "mem0_client"
    REDIS = "redis"
    GOOGLE_DOCS = "google_docs"
    GOOGLE_CALENDAR = "google_calendar"
    DOCUMENT_AI = "document_ai"
    PDFCO = "pdfco"
    DLP = "dlp"
    VENDOR_DB = "vendor_db"
    OUTBOUND_APPROVALS = "outbound_approvals"
    OPERATOR_INBOX = "operator_inbox"
    SCHEDULING = "scheduling"
    AGENT_JOURNAL = "agent_journal"
    POWER_LEVEL_TRACKER = "power_level_tracker"
    EMAIL_INTERCONNECTIONS = "email_interconnections"


ALL_SERVICE_KEYS = frozenset(v for k, v in vars(ServiceKey).items() if not k.startswith("_"))
