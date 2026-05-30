#!/usr/bin/env python3
"""Fail CI if `git ls-files` contains commercial/customer leakage or forbidden paths.

Conservative checks: substring bans, path bans, emails outside demo domains,
purchase-order shaped tokens, invoice/proforma lines with money-like amounts,
and currency literals outside explicit demo knowledge files.

Heavier entropy scanning stays in gitleaks (run separately on `git archive`)."""

from __future__ import annotations

import fnmatch
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST_PATH = ROOT / ".public_repo_allowlist.yml"

_THIS_FILE = "scripts/public_repo_guard.py"

# Path-shaped bans (exact legacy filenames / globs relative to repo root).
_FORBIDDEN_PATH_GLOBS = (
    "scripts/proforma_*",
    "scripts/invoice_*",
    "scripts/*naffco*",
    "scripts/*crm_update*",
    "scripts/send_*",
    "scripts/*_lead*.py",
    "data/imports/**",
    "data/knowledge/sales_playbook.md",
    "data/knowledge/one_pager_uae_fire_safety.md",
    "data/knowledge/european_sales_cycle_analysis.md",
    "data/knowledge/verified_sales_cycles.md",
    "data/knowledge/lead_heat_formula.md",
    "data/knowledge/lead_ranker_formula.md",
    "data/knowledge/pf1_specs_and_options.md",
    "data/knowledge/plain_text_email_sources.md",
)

_ORG_EMAIL_DOMAIN_FRAGMENTS = (
    "machinecraft.org",
    "machinecraft.in",
    "rushabh@",
    "@machinecraft.",
)

_CAMPAIGN_SUBSTRINGS = (
    "naffco",
    "dutchtides",
    "joplast",
    "soehner",
    "batelaan",
    "ridat",
    "bd-plastindustri",
    "donite",
    "plastochim",
    "dezet",
    "bermaq",
    "cozwei",
    "christian.philippen",
    "ruslan.didenko",
    "emad@naffco",
    "kraussmaffei",
    "vuteq",
)

_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

_SAFE_EMAIL_DOMAINS = frozenset(
    {
        "example.com",
        "example.org",
        "example.net",
        "invalid",
        "localhost",
        "test",
        "github.com",
        "example-company.org",
        "example.de",
        "acme.com",
        "acme.de",
        "acme-corp.com",
        "company.com",
        "domain.tld",
        "northwind-demo.example",
        "summit-demo.example",
        "imports.placeholder",
        "sentry.io",
        "wikimedia.org",
        "python.org",
        "pypi.org",
        "readthedocs.io",
    }
)

# Avoid matching "postgresql" / "postpone": require a separator after PO / P.O.
_PO_LINE_RE = re.compile(
    r"(?i)\b(?:P\.O\.|PO)\b(?:\s*[#:.:-]\s*|\s+)([A-Z0-9][A-Z0-9_-]{2,}\d|\d{5,})\b",
)

_COMMERCIAL_KW = re.compile(
    r"(?i)\b(invoice|proforma|pro forma|purchase\s+order|payment\s+received|down\s+payment|balance\s+payment)\b",
)
_MONEYISH_RE = re.compile(
    r"(?i)(€|£|\$|inr\b|usd\b|eur\b|lakhs?\b|\d[\d,.]{2,}\s*(€|\$|£)?|(€|\$|£)\s*[\d,.]{3,})",
)

_CURRENCY_LINE_RE = re.compile(
    r"(?i)(€|£|\$\s*\d|\d[\d,.]{2,}\s*(€|\$|£)|\binr\b\s*\d|\busd\b\s*\d|\blur\b\s*\d|\bear\b\s*\d|\blakhs?\b)",
)


def _git_ls_files() -> list[str]:
    if (ROOT / ".git").is_dir():
        out = subprocess.run(
            ["git", "-c", "safe.directory=*", "ls-files"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    files: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and ".git" not in path.parts:
            files.append(path.relative_to(ROOT).as_posix())
    return files


def _load_allowlist() -> dict:
    if yaml is None or not ALLOWLIST_PATH.is_file():
        return {}
    try:
        return yaml.safe_load(ALLOWLIST_PATH.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError):
        return {}


def _path_allowed(rel: str, allow: dict) -> bool:
    for prefix in allow.get("skip_paths_prefix", []) or []:
        p = prefix.rstrip("/")
        if rel == p or rel.startswith(p + "/"):
            return True
    return False


def _line_exempt(line: str, rel: str, allow: dict) -> bool:
    for row in allow.get("exempt_line_contains", []) or []:
        if row.get("path_fragment") and row["path_fragment"] not in rel:
            continue
        if row.get("substring") and row["substring"] in line:
            return True
    return False


def _path_forbidden(rel: str) -> bool | str:
    for pat in _FORBIDDEN_PATH_GLOBS:
        if "**" in pat:
            prefix, suffix = pat.split("**", 1)
            prefix = prefix.rstrip("/")
            if rel.startswith(prefix):
                return pat
        elif "*" in pat or "?" in pat:
            if fnmatch.fnmatch(rel, pat):
                return pat
        elif rel == pat:
            return pat
    return False


def _knowledge_dir_violation(rel: str) -> str | None:
    if not rel.startswith("data/knowledge/"):
        return None
    name = Path(rel).name
    allowed = (
        name == ".gitkeep"
        or name == "README.md"
        or (name.startswith("demo_") and name.endswith(".md"))
    )
    if allowed:
        return None
    return f"{rel}: only .gitkeep, README.md, demo_*.md allowed under data/knowledge/"


def _is_demo_knowledge(rel: str) -> bool:
    return rel.startswith("data/knowledge/demo_") and rel.endswith(".md")


def _demo_banner_ok(text: str) -> bool:
    first = text.lstrip("\ufeff").splitlines()[0] if text.strip() else ""
    want = (
        "PUBLIC DEMO DATA — fully synthetic. This file does not contain Machinecraft "
        "customer, pricing, quote, invoice, CRM, Gmail, or email data."
    )
    return first.strip() == want


# PO / invoice+currency / bare currency heuristics: **scripts/** only (historical HTML/ops payloads).
# Application source (`src/`, `memory/`, `crm/`, …) legitimately mentions USD/EUR in guardrails and tests.
_COMMERCIAL_SCAN_PREFIXES = ("scripts/",)

# Maintainer machine paths and identity — never in public brain-os (src/ + prompts/).
_IDENTITY_SCAN_PREFIXES = ("src/", "prompts/")
_HOME_PATH_FRAGMENTS = ("/users/", "/home/", "desktop/ira-v3", "desktop/ira-v3")
_IDENTITY_LITERALS = ("rushabh@", "rushabh doshi")
_DEIRA_FRAGMENTS = (
    "ira-v3",
    "ira-universe",
    "ira_universe",
    "query_ira",
    "ira_segment",
    "ira-pimp",
    "iraerror",
    "ira-network",
    "ira:ira@",
    "src/ira",
    "poetry run ira",
    "``ira ",
    "`ira ",
    "ira brief",
    "ira ask",
    "ira tinder",
    "ira_universe",
    "republish_ira",
    "export_ira_universe",
)
_CUSTOMER_FRAGMENTS = (
    "faure france",
    "faure ",
    "tvs motor",
    "dashmesh",
    "pattison sign",
    "naffco",
    "mikhail",
    "gerwin",
    "machinecraft",
    "rushabh@",
    "plastindia",
    "formpack.in",
    "data/imports/",
)
_DEIRA_BRAND_PREFIXES = (
    "src/",
    "prompts/",
    "docs/",
    "examples/",
    "alembic/",
    "docker-compose",
    "export_manifest.json",
)
_VERTICAL_FRAGMENTS = (
    "machinecraft.org",
    "machinecraft.in",
    "@machinecraft.",
    "quotemachinecraft",
    "naffco",
    "formpack.in",
    "plastindia",
    "active-21",
)
_SAFE_PUBLIC_DOMAIN_FRAGMENTS = (
    "example-company.org",
    "example-company.in",
    "partnerpack.example",
    "acme-corp.",
    "example.com",
    "acme.com",
)


def _scan_public_identity(rel: str, text: str, hits: list[str]) -> None:
    lower = text.lower()
    if rel == "export_manifest.json":
        for frag in _HOME_PATH_FRAGMENTS:
            if frag in lower:
                hits.append(f"{rel}: forbidden home/path fragment `{frag}`")
        if "/users/" in lower and "rdd0101" in lower:
            hits.append(f"{rel}: maintainer home directory path leaked")
    if not rel.startswith(_IDENTITY_SCAN_PREFIXES):
        return
    for lit in _IDENTITY_LITERALS:
        if lit in lower:
            hits.append(f"{rel}: forbidden identity literal `{lit}`")
    for frag in _VERTICAL_FRAGMENTS:
        if frag in lower:
            hits.append(f"{rel}: forbidden vertical fragment `{frag}`")
    if "data/imports/" in lower and "examples/acme/docs/" not in lower:
        hits.append(f"{rel}: forbidden path `data/imports/`")
    if re.search(r"\bmachinecraft\b", lower):
        hits.append(f"{rel}: forbidden token `machinecraft`")


def _scan_deira_branding(rel: str, text: str, hits: list[str]) -> None:
    """Public brain-os must not reference the Ira operator product by name."""
    if rel == "scripts/public_repo_guard.py":
        return
    if not (
        rel.startswith(_DEIRA_BRAND_PREFIXES)
        or rel == "export_manifest.json"
        or rel.startswith("docker-compose")
    ):
        return
    lower = text.lower()
    for frag in _DEIRA_FRAGMENTS:
        if frag in lower:
            hits.append(f"{rel}: forbidden Ira-brand fragment `{frag}`")
    for frag in _CUSTOMER_FRAGMENTS:
        if frag in lower:
            hits.append(f"{rel}: forbidden customer/vertical fragment `{frag}`")
    if re.search(r"\bIra\b", text):
        hits.append(f"{rel}: forbidden brand token `Ira`")
    if re.search(r"\bPF1\b", text) and "public_repo_guard" not in rel:
        hits.append(f"{rel}: forbidden vertical model token `PF1`")


def _scan_text(rel: str, text: str, allow: dict, hits: list[str]) -> None:
    _scan_public_identity(rel, text, hits)
    _scan_deira_branding(rel, text, hits)
    lower = text.lower()
    for frag in _ORG_EMAIL_DOMAIN_FRAGMENTS:
        if frag.lower() in lower:
            hits.append(f"{rel}: forbidden org/domain fragment `{frag}`")
    for tok in _CAMPAIGN_SUBSTRINGS:
        if tok in lower:
            hits.append(f"{rel}: forbidden token `{tok}`")

    skip_email = rel.startswith("tests/") or rel.endswith(".png") or rel.endswith(".jpg")
    if not skip_email:
        for m in _EMAIL_RE.finditer(text):
            dom = m.group(2).lower()
            addr = m.group(0)
            if dom in _SAFE_EMAIL_DOMAINS or dom.endswith(".example.com"):
                continue
            if dom in ("example-company.org", "example-company.in", "partnerpack.example"):
                continue
            if dom.endswith(".example.org") or dom.endswith(".example.net"):
                continue
            local = m.group(1).lower()
            if local in ("test", "no-reply", "noreply", "you") or local.startswith("your_"):
                continue
            line_ctx = next((ln for ln in text.splitlines() if addr in ln), addr)
            if _line_exempt(line_ctx, rel, allow):
                continue
            hits.append(f"{rel}: non-demo email `{addr}`")

    scan_commercial = rel.startswith(_COMMERCIAL_SCAN_PREFIXES) and not rel.startswith("tests/")
    if not scan_commercial:
        return
    # Shell helpers use timestamped paths / ports — not invoice payloads.
    if rel.endswith((".sh", ".command")):
        return

    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if _line_exempt(raw_line, rel, allow):
            continue

        if _PO_LINE_RE.search(raw_line):
            hits.append(f"{rel}:{i}: suspicious PO-shaped token")

        if _COMMERCIAL_KW.search(raw_line) and _MONEYISH_RE.search(raw_line):
            hits.append(f"{rel}:{i}: commercial keyword with amount-like token")

        if _is_demo_knowledge(rel):
            continue

        # Currency / amounts outside demos — reduces lakhs / EUR leaking from ops code.
        if "XX" in raw_line or "YY" in raw_line or "placeholder" in raw_line.lower():
            continue
        if _CURRENCY_LINE_RE.search(raw_line):
            hits.append(f"{rel}:{i}: currency / amount literal outside demo knowledge file")


def main() -> int:
    import argparse

    global ROOT, ALLOWLIST_PATH
    parser = argparse.ArgumentParser(description="Scan a repo tree for public-export leaks.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root to scan (default: parent of this script). Uses git ls-files or walk.",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=None,
        help="Optional .public_repo_allowlist.yml (default: <root>/.public_repo_allowlist.yml).",
    )
    args = parser.parse_args()
    if args.root is not None:
        ROOT = args.root.resolve()
    if args.allowlist is not None:
        ALLOWLIST_PATH = args.allowlist.resolve()
    elif (ROOT / ".public_repo_allowlist.yml").is_file():
        ALLOWLIST_PATH = ROOT / ".public_repo_allowlist.yml"

    allow = _load_allowlist()
    if yaml is None:
        print(
            "public_repo_guard: PyYAML not installed — pip install pyyaml / Poetry env",
            file=sys.stderr,
        )
        return 1

    hits: list[str] = []
    for rel in _git_ls_files():
        if rel == _THIS_FILE:
            continue
        if rel.endswith(".public_repo_allowlist.yml"):
            continue
        if _path_allowed(rel, allow):
            continue

        bad_pat = _path_forbidden(rel)
        if bad_pat:
            hits.append(f"{rel}: forbidden path pattern `{bad_pat}`")

        viol = _knowledge_dir_violation(rel)
        if viol:
            hits.append(viol)

        path = ROOT / rel
        if not path.is_file():
            continue

        # Demo markdown must carry the canonical banner (first line).
        if _is_demo_knowledge(rel):
            try:
                demo_txt = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not _demo_banner_ok(demo_txt):
                hits.append(f"{rel}: missing/wrong PUBLIC DEMO banner (first line)")

        try:
            binary_suffix = (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".webp",
                ".ico",
                ".woff",
                ".woff2",
                ".parquet",
                ".pickle",
                ".pkl",
                ".gz",
                ".zip",
            )
            if rel.endswith(binary_suffix):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        _scan_text(rel, text, allow, hits)

    if hits:
        print("public_repo_guard: FAILED — fix or allowlist with justification:\n", file=sys.stderr)
        for h in hits[:250]:
            print(h, file=sys.stderr)
        if len(hits) > 250:
            print(f"... and {len(hits) - 250} more", file=sys.stderr)
        return 1
    print("public_repo_guard: OK (tracked tree passes destructive public guard)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
