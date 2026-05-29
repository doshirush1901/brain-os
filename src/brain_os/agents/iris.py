"""Iris — External Intelligence agent.

Gathers real-time information from the web, news APIs, and external
sources to enrich internal knowledge.
Now operates via the ReAct loop with web-search, news-fetch, scrape,
and internal-knowledge tools.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from brain_os.agents.base_agent import AgentTool, BaseAgent
from brain_os.config import get_settings
from brain_os.prompt_loader import load_prompt
from brain_os.services.tool_runner import run_tool
from brain_os.systems import google_maps as google_maps_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = load_prompt("iris_system")


class Iris(BaseAgent):
    name = "iris"
    role = "External Intelligence"
    description = "Web search, news monitoring, and external research"
    knowledge_categories = [
        "market_research_and_analysis",
        "industry_knowledge",
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._newsdata_key = get_settings().external_apis.api_key.get_secret_value()

    def _register_default_tools(self) -> None:
        super()._register_default_tools()

        self.register_tool(
            AgentTool(
                name="fetch_news",
                description=(
                    "Fetch recent news articles via NewsData.io (/latest, business/tech categories). "
                    "Optional country: two-letter code (e.g. us, de) to bias headlines."
                ),
                parameters={
                    "query": "News search query (include company + industry terms for tighter hits)",
                    "country": "Optional two-letter country filter (e.g. us, ca); leave empty for global",
                },
                handler=self._tool_fetch_news,
            )
        )

        self.register_tool(
            AgentTool(
                name="search_internal_knowledge",
                description="Search Iris's domain knowledge categories (market research, industry knowledge).",
                parameters={"query": "Internal knowledge search query"},
                handler=self._tool_search_internal_knowledge,
            )
        )
        self.register_tool(
            AgentTool(
                name="searchapi_search",
                description=(
                    "Search via SearchAPI.io using SEARCHAPI_API_KEY; bypasses Tavily/Serper. "
                    "Optional engine (e.g. google, bing); default from SEARCHAPI_ENGINE."
                ),
                parameters={
                    "query": "Search query",
                    "engine": "SearchAPI engine id; leave empty for SEARCHAPI_ENGINE default",
                    "max_results": "Max results 1–50 (default 5)",
                },
                handler=self._tool_searchapi_search,
            )
        )
        self.register_tool(
            AgentTool(
                name="search_knowledge_base_skill",
                description="Search the broader internal knowledge base through canonical skill routing.",
                parameters={"query": "Search query"},
                handler=self._tool_search_knowledge_base_skill,
            )
        )
        self.register_tool(
            AgentTool(
                name="fetch_youtube_transcript",
                description=(
                    "Fetch captions for a public YouTube watch/embed/shorts URL (or 11-char video id). "
                    "Returns timestamped plain text when captions exist; reports languages if missing."
                ),
                parameters={
                    "url": "YouTube URL or video id",
                    "languages": "Optional comma-separated language codes (default en)",
                    "include_timestamps": "Optional true/false — prefix lines with [MM:SS] (default true)",
                },
                handler=self._tool_fetch_youtube_transcript,
            )
        )
        self.register_tool(
            AgentTool(
                name="maps_geocode",
                description=(
                    "Resolve a postal address or place name to coordinates using Google Geocoding API "
                    "(requires GOOGLE_MAPS_API_KEY). Optional region is a ccTLD hint (e.g. in, de)."
                ),
                parameters={
                    "address": "Street, city, region, or place to geocode",
                    "region": "Optional two-letter region bias (e.g. in)",
                },
                handler=self._tool_maps_geocode,
            )
        )
        self.register_tool(
            AgentTool(
                name="maps_driving_route",
                description=(
                    "Get driving distance and duration between two places (addresses or lat,lng) "
                    "via Google Directions API. Optional mode: driving, walking, bicycling, transit."
                ),
                parameters={
                    "origin": "Start address or lat,lng",
                    "destination": "End address or lat,lng",
                    "mode": "Optional: driving (default), walking, bicycling, transit",
                },
                handler=self._tool_maps_driving_route,
            )
        )
        self.register_tool(
            AgentTool(
                name="maps_places_text_search",
                description=(
                    "Google Places API (New) Text Search: find businesses/places from a natural-language "
                    "query (e.g. 'custom industrial forming manufacturer near Cleveland Ohio'). "
                    "Optional location_bias 'lat,lng,radius_m' (radius meters, max 50000) to bias results; "
                    "optional included_type (Table A type, e.g. establishment). "
                    "Results are discovery hints only — verify with web_search and internal KB; expect false positives."
                ),
                parameters={
                    "text_query": "What to search for (include geography in the query or use location_bias)",
                    "max_results": "Optional 1–20 (default 10)",
                    "location_bias": "Optional 'lat,lng' or 'lat,lng,radius_meters'",
                    "language_code": "Optional BCP-47 code (default en)",
                    "included_type": "Optional Google place type filter (often empty)",
                    "page_token": "Optional next_page_token from a previous Text Search response for pagination",
                },
                handler=self._tool_maps_places_text_search,
            )
        )
        self.register_tool(
            AgentTool(
                name="maps_places_nearby_search",
                description=(
                    "Google Places Nearby Search (New): list places inside a circle (lat, lng, radius_m). "
                    "Optional included_types comma-separated Table A types (empty = all types, noisier). "
                    "rank_preference: DISTANCE (default), POPULARITY, or empty for API default. "
                    "Use after geocoding a metro; verify industrial forming fit with web_search / site review."
                ),
                parameters={
                    "latitude": "Center latitude (decimal)",
                    "longitude": "Center longitude (decimal)",
                    "radius_meters": "Search radius in meters (1–50000)",
                    "included_types": "Optional comma-separated place types (e.g. establishment)",
                    "max_results": "Optional 1–20 (default 10)",
                    "language_code": "Optional BCP-47 (default en)",
                    "rank_preference": "Optional DISTANCE, POPULARITY, or empty",
                },
                handler=self._tool_maps_places_nearby_search,
            )
        )
        self.register_tool(
            AgentTool(
                name="maps_place_details",
                description=(
                    "Google Places Place Details (New) for one place id (ChIJ… or places/ChIJ… from Text/Nearby). "
                    "Returns address, types, website, phone, rating when available. Billed per request."
                ),
                parameters={
                    "place_id": "Place id from maps_places_text_search / maps_places_nearby_search",
                    "language_code": "Optional BCP-47 (default en)",
                },
                handler=self._tool_maps_place_details,
            )
        )

    async def handle(self, query: str, context: dict[str, Any] | None = None) -> str:
        return await self.run(query, context, system_prompt=_SYSTEM_PROMPT)

    async def _tool_fetch_news(self, query: str, country: str = "") -> str:
        articles = await self._fetch_news(query, country=country)
        if not articles:
            if not self._newsdata_key:
                return "News unavailable — NEWSDATA_API_KEY not configured."
            return "No news articles found."
        return "\n".join(f"- {a}" for a in articles)

    async def _tool_search_internal_knowledge(self, query: str) -> str:
        results = await self.search_domain_knowledge(query)
        return self._format_context(results)

    async def _tool_searchapi_search(
        self, query: str, engine: str = "", max_results: str = "5"
    ) -> str:
        try:
            n = int(max_results) if max_results else 5
        except ValueError:
            n = 5
        n = max(1, min(n, 50))
        eng = engine.strip() or None

        async def _do_searchapi() -> list[dict[str, Any]]:
            return await self.searchapi_search(query, engine=eng, max_results=n)

        sa_result = await run_tool(
            f"{self.name}.searchapi_search",
            _do_searchapi,
            retries=1,
            backoff_seconds=0.25,
        )
        if not sa_result.ok:
            return f"SearchAPI request failed: {sa_result.error}"
        rows = sa_result.value or []
        if not rows:
            if not get_settings().search.searchapi_api_key.get_secret_value().strip():
                return "SearchAPI unavailable — SEARCHAPI_API_KEY not configured."
            return "No SearchAPI results (quota, blocked query, or engine returned no organic/news rows)."
        return json.dumps(rows, indent=2)

    async def _tool_search_knowledge_base_skill(self, query: str) -> str:
        return await self.use_skill("search_knowledge_base", query=query)

    async def _tool_fetch_youtube_transcript(
        self,
        url: str,
        languages: str = "",
        include_timestamps: str = "true",
    ) -> str:
        return await self.use_skill(
            "fetch_youtube_transcript",
            url=url,
            languages=languages or None,
            include_timestamps=include_timestamps,
        )

    async def _tool_maps_geocode(self, address: str, region: str = "") -> str:
        key = get_settings().google.maps_api_key.get_secret_value().strip()
        if not key:
            return "Maps geocoding unavailable — GOOGLE_MAPS_API_KEY not configured."
        result = await google_maps_client.geocode_address(address, key, region=region or None)
        if not result.get("ok"):
            return f"Geocoding failed: {result.get('error', 'unknown error')}"
        rows = result.get("results") or []
        if not rows:
            return "No geocoding results for that query."
        lines = []
        for r in rows:
            lines.append(
                f"{r.get('formatted_address')} — ({r.get('lat')}, {r.get('lng')}) "
                f"[{r.get('location_type')}] place_id={r.get('place_id')}"
            )
        return "\n".join(lines)

    async def _tool_maps_driving_route(
        self, origin: str, destination: str, mode: str = "driving"
    ) -> str:
        key = get_settings().google.maps_api_key.get_secret_value().strip()
        if not key:
            return "Maps directions unavailable — GOOGLE_MAPS_API_KEY not configured."
        result = await google_maps_client.driving_route(origin, destination, key, mode=mode)
        if not result.get("ok"):
            return f"Directions failed: {result.get('error', 'unknown error')}"
        routes = result.get("routes") or []
        if not routes:
            return "No routes returned for that origin/destination."
        parts: list[str] = []
        for i, route in enumerate(routes, start=1):
            summary = route.get("summary") or f"Route {i}"
            warn = route.get("warnings") or []
            parts.append(f"**{summary}**" + (f" — warnings: {warn}" if warn else ""))
            for leg in route.get("legs") or []:
                parts.append(
                    f"  {leg.get('start_address')} → {leg.get('end_address')}: "
                    f"{leg.get('distance_text')} in {leg.get('duration_text')}"
                )
        return "\n".join(parts)

    async def _tool_maps_places_text_search(
        self,
        text_query: str,
        max_results: str = "10",
        location_bias: str = "",
        language_code: str = "en",
        included_type: str = "",
        page_token: str = "",
    ) -> str:
        key = get_settings().google.maps_api_key.get_secret_value().strip()
        if not key:
            return "Places search unavailable — GOOGLE_MAPS_API_KEY not configured."
        try:
            n = int(max_results) if max_results else 10
        except ValueError:
            n = 10
        lat, lng, rad = google_maps_client.parse_places_location_bias(location_bias)
        result = await google_maps_client.places_text_search(
            text_query,
            key,
            max_results=n,
            latitude=lat,
            longitude=lng,
            radius_meters=rad,
            language_code=language_code or "en",
            included_type=included_type.strip() or None,
            page_token=page_token.strip() or None,
        )
        if not result.get("ok"):
            return f"Places search failed: {result.get('error', 'unknown error')}"
        rows = result.get("places") or []
        if not rows:
            return "No places returned — broaden the query or adjust location_bias."
        lines: list[str] = []
        for p in rows:
            nm = p.get("name") or "(no name)"
            addr = p.get("formatted_address") or ""
            web = p.get("website") or ""
            phone = p.get("phone") or ""
            types = ", ".join((p.get("types") or [])[:6])
            pid = p.get("id") or ""
            lines.append(
                f"- {nm} | id={pid} | {addr} | {p.get('lat')}, {p.get('lng')} | {web} | {phone} | types: {types}"
            )
        npt = result.get("next_page_token")
        if npt:
            lines.append(f"\nnext_page_token (pass to maps_places_text_search page_token): {npt}")
        return "\n".join(lines)

    async def _tool_maps_places_nearby_search(
        self,
        latitude: str,
        longitude: str,
        radius_meters: str,
        included_types: str = "",
        max_results: str = "10",
        language_code: str = "en",
        rank_preference: str = "DISTANCE",
    ) -> str:
        key = get_settings().google.maps_api_key.get_secret_value().strip()
        if not key:
            return "Places nearby unavailable — GOOGLE_MAPS_API_KEY not configured."
        try:
            lat = float(latitude)
            lng = float(longitude)
            rad = float(radius_meters)
        except ValueError:
            return "Invalid latitude, longitude, or radius_meters (numbers required)."
        try:
            n = int(max_results) if max_results else 10
        except ValueError:
            n = 10
        types_list: list[str] | None = None
        raw_types = (included_types or "").strip()
        if raw_types:
            types_list = [t.strip() for t in raw_types.split(",") if t.strip()][:50]
        rp = (rank_preference or "").strip().upper()
        rp_out: str | None = rp if rp in {"DISTANCE", "POPULARITY"} else None
        result = await google_maps_client.places_nearby_search(
            lat,
            lng,
            rad,
            key,
            included_types=types_list,
            max_results=n,
            language_code=language_code or "en",
            rank_preference=rp_out,
        )
        if not result.get("ok"):
            return f"Places nearby failed: {result.get('error', 'unknown error')}"
        rows = result.get("places") or []
        if not rows:
            return "No places in that circle — widen radius or change included_types."
        lines: list[str] = []
        for p in rows:
            nm = p.get("name") or "(no name)"
            pid = p.get("id") or ""
            addr = p.get("formatted_address") or ""
            types = ", ".join((p.get("types") or [])[:6])
            lines.append(f"- {nm} | id={pid} | {addr} | types: {types}")
        return "\n".join(lines)

    async def _tool_maps_place_details(self, place_id: str, language_code: str = "en") -> str:
        key = get_settings().google.maps_api_key.get_secret_value().strip()
        if not key:
            return "Place details unavailable — GOOGLE_MAPS_API_KEY not configured."
        lc = (language_code or "").strip()
        result = await google_maps_client.place_details(
            place_id,
            key,
            language_code=lc or None,
        )
        if not result.get("ok"):
            return f"Place details failed: {result.get('error', 'unknown error')}"
        p = result.get("place") or {}
        parts = [
            f"name: {p.get('name')}",
            f"id: {p.get('id')}",
            f"address: {p.get('formatted_address')}",
            f"lat,lng: {p.get('lat')}, {p.get('lng')}",
            f"website: {p.get('website')}",
            f"phone: {p.get('phone')}",
            f"business_status: {p.get('business_status')}",
            f"primary_type: {p.get('primary_type')} ({p.get('primary_type_label')})",
            f"rating: {p.get('rating')} ({p.get('user_rating_count')} ratings)",
            f"types: {', '.join((p.get('types') or [])[:12])}",
        ]
        return "\n".join(parts)

    async def _fetch_news(self, query: str, country: str = "") -> list[str]:
        if not self._newsdata_key:
            return []
        from brain_os.tools.newsdata_client import fetch_headline_lines

        return await fetch_headline_lines(
            (query or "").strip(),
            country=country,
            max_items=15,
        )
