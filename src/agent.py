#!/usr/bin/env python3
"""
Agent enrichment — uses Google Gemini to read each fetched paper, extract its
key scientific content (finding / method / dataset), and score how relevant it
is to earthquake & seismology research. Papers below the relevance threshold
are dropped, and the rest are sorted most-relevant first.

Runs only when GEMINI_API_KEY (or GOOGLE_API_KEY) is set; otherwise the digest
is built from the raw CrossRef/arXiv data exactly as before (no hard dependency
at import time). Get a free key at https://aistudio.google.com/apikey

Uses Gemini function calling (tool declarations) for structured output.
"""

import json
import os
import time
from typing import Dict, List

try:
    from google import genai
    from google.genai import types
except ImportError:  # SDK not installed — enrichment simply no-ops
    genai = None

# Free-tier friendly default. Override with GEMINI_MODEL, e.g.
# gemini-2.5-flash-lite (cheaper/faster) or gemini-2.5-pro (stronger).
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Keep papers scoring at/above this (0-100). Lower it to be more inclusive.
RELEVANCE_THRESHOLD = int(os.environ.get("RELEVANCE_THRESHOLD", "50"))

# Seconds to pause between calls, to stay under the free-tier per-minute limit
# (gemini-2.5-flash free tier is ~10-15 requests/min). Raise if you see 429s.
DELAY = float(os.environ.get("GEMINI_DELAY", "4.0"))

# ── Function declaration for structured paper analysis ──────────────────

_ANALYZE_FN = types.FunctionDeclaration(
    name="analyze_paper",
    description=(
        "Analyze a scientific paper's relevance to earthquake/seismology "
        "research and extract its key scientific content (finding, method, dataset)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "relevance": {
                "type": "integer",
                "description": "Relevance to earthquake/seismology research, 0-100. 100 = squarely on-topic; a paper that only mentions earthquakes in passing or comes from an unrelated field should score low.",
            },
            "finding": {
                "type": "string",
                "description": "One plain-language sentence stating the paper's main result.",
            },
            "method": {
                "type": "string",
                "description": "Main method/approach used (e.g. 'ETAS model', 'graph neural network'). Write 'N/A' if unclear.",
            },
            "dataset": {
                "type": "string",
                "description": "Data or study region used (e.g. 'Southern California catalog'). Write 'N/A' if unclear.",
            },
        },
        "required": ["relevance", "finding", "method", "dataset"],
    },
)

# Force the model to always call the function — no text-only responses.
_TOOL = types.Tool(function_declarations=[_ANALYZE_FN])

_TOOL_CONFIG = types.ToolConfig(
    function_calling_config=types.FunctionCallingConfig(
        mode=types.FunctionCallingConfigMode.ANY,
        allowed_function_names=["analyze_paper"],
    ),
)

SYSTEM = (
    "You are a research assistant for a scientist who studies earthquakes, "
    "seismology, and the statistical and machine-learning methods applied to "
    "them. For each paper you are given, judge how relevant it is to that field "
    "and extract its key scientific content. Be strict about relevance: a paper "
    "that only mentions earthquakes in passing, or comes from an unrelated "
    "field, should score low. Ground every field in the title and abstract "
    "provided — do not invent details."
)


def _analyze(client, paper: Dict) -> Dict:
    """One function-call per paper. Returns dict with relevance/finding/method/dataset."""
    content = (
        f"Title: {paper['title']}\n\n"
        f"Abstract: {paper['abstract'] or '(no abstract available)'}"
    )
    resp = client.models.generate_content(
        model=MODEL,
        contents=content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            temperature=0,
            tools=[_TOOL],
            tool_config=_TOOL_CONFIG,
        ),
    )
    # With mode=ANY the model always calls the function; parse its arguments.
    fc = resp.candidates[0].content.parts[0].function_call
    return dict(fc.args)


def enrich(results: List[Dict]) -> List[Dict]:
    """Add agent fields to each paper, drop low-relevance ones, and sort.

    `results` is the list of section dicts built by tracker.py, each shaped as
    {"title": str, "journals": [{"name": str, "papers": [paper, ...]}, ...]}.
    Mutates in place and returns it. A no-op (returns unchanged) when the SDK
    or API key is absent, so the tracker keeps working without Gemini.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if genai is None or not api_key:
        print("  [agent] GEMINI_API_KEY not set — skipping agent enrichment.")
        return results

    client = genai.Client(api_key=api_key)
    analyzed = 0

    for sec in results:
        for jr in sec["journals"]:
            kept: List[Dict] = []
            for p in jr["papers"]:
                try:
                    a = _analyze(client, p)
                    p["relevance"] = int(a.get("relevance", 0))
                    p["finding"] = str(a.get("finding", ""))
                    p["method"] = str(a.get("method", ""))
                    p["dataset"] = str(a.get("dataset", ""))
                    analyzed += 1
                except Exception as e:  # keep the paper on any failure
                    print(f"  [agent] WARN {p['title'][:60]!r}: {e}")
                    p.setdefault("relevance", RELEVANCE_THRESHOLD)

                if p.get("relevance", 0) >= RELEVANCE_THRESHOLD:
                    kept.append(p)
                if DELAY:
                    time.sleep(DELAY)  # respect free-tier rate limits

            kept.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            jr["papers"] = kept

    print(f"  [agent] analyzed {analyzed} paper(s); model={MODEL}")
    return results
