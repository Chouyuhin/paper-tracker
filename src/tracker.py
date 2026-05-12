#!/usr/bin/env python3
"""
Paper Tracker — Daily digest of earthquake-related articles from
multiple journals via the CrossRef API.
"""

import os
import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict, Optional

import requests

# ── Journal definitions ──────────────────────────────────────────────

CORE_JOURNALS = [
    {"name": "Physical Review Letters",           "issns": ["0031-9007", "1079-7114"]},
    {"name": "Physical Review E",                 "issns": ["1539-3755", "2470-0053"]},
    {"name": "J. American Statistical Association","issns": ["0162-1459", "1537-274X"]},
    {"name": "BSSA",                              "issns": ["0037-1106", "1943-3573"]},
]

NEW_JOURNALS = [
    {"name": "Nature",                            "issns": ["0028-0836", "1476-4687"]},
    {"name": "Science",                           "issns": ["0036-8075", "1095-9203"]},
    {"name": "Nature Geoscience",                 "issns": ["1752-0894", "1752-0908"]},
    {"name": "Geophysical Research Letters",       "issns": ["0094-8276", "1944-8007"]},
    {"name": "Nature Communications",             "issns": ["2041-1723"]},
    {"name": "Seismological Research Letters",    "issns": ["0895-0695", "1938-2057"]},
    {"name": "Geophysical Journal International", "issns": ["0956-540X", "1365-246X"]},
    {"name": "JGR Solid Earth",                  "issns": ["2169-9313", "2169-9356"]},
    {"name": "Communications Earth & Environment","issns": ["2662-4435"]},
]

SECTIONS = [
    {
        "tag":      "Earthquake",
        "title":    "Earthquake Research",
        "query":    "earthquake",
        "journals": CORE_JOURNALS,
        "max":      None,
        "filter":   None,
    },
    {
        "tag":      "KnowledgeGraph",
        "title":    "Knowledge Graph in Seismology",
        "query":    "knowledge graph earthquake",
        "journals": NEW_JOURNALS,
        "max":      3,
        "filter":   "knowledge_graph",
    },
    {
        "tag":      "StatSeismo",
        "title":    "Statistical Seismology",
        "query":    "statistical seismology earthquake",
        "journals": NEW_JOURNALS,
        "max":      3,
        "filter":   "statistical_seismology",
    },
]

CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper"
PER_PAGE = 30
LAST_RUN_FILE = ".last_run_date"


# ── Date tracking ────────────────────────────────────────────────────

def get_since_date() -> Optional[str]:
    """Read the last successful run date. None = first run (get last 7 days)."""
    try:
        with open(LAST_RUN_FILE) as f:
            val = f.read().strip()
            return val if val else None
    except FileNotFoundError:
        return None


def save_run_date():
    """Write today's date as the last successful run date."""
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))


# ── Fetching ─────────────────────────────────────────────────────────

def fetch_papers(journal: Dict, query: str, filter_type: Optional[str] = None,
                 since_date: Optional[str] = None) -> List[Dict]:
    papers: List[Dict] = []
    for issn in journal["issns"]:
        filter_parts = [f"issn:{issn}"]
        if since_date:
            filter_parts.append(f"from-index-date:{since_date}")
        params = {
            "query": query,
            "filter": ",".join(filter_parts),
            "sort": "published",
            "order": "desc",
            "rows": PER_PAGE,
        }
        try:
            r = requests.get(CROSSREF_URL, params=params, timeout=30)
            r.raise_for_status()
            for item in r.json().get("message", {}).get("items", []):
                papers.append(_extract(item, journal["name"]))
        except Exception as e:
            print(f"  [WARN] {journal['name']} (ISSN {issn}): {e}")

    seen: set = set()
    unique: List[Dict] = []
    for p in papers:
        if p["doi"] and p["doi"] not in seen:
            seen.add(p["doi"])
            unique.append(p)

    enrich_abstracts(unique)
    unique = filter_papers(unique, filter_type)
    return unique


def enrich_abstracts(papers: List[Dict]):
    dois = [p["doi"] for p in papers if p["doi"] and (not p["abstract"] or len(p["abstract"]) < 50)]
    if not dois:
        return

    for i in range(0, len(dois), 20):
        batch = dois[i : i + 20]
        try:
            r = requests.post(
                f"{SEMANTIC_URL}/batch",
                params=[("fields", "abstract")],
                json={"ids": [f"DOI:{d}" for d in batch]},
                timeout=30,
            )
            if not r.ok:
                continue
            results = r.json()
            for j, doi in enumerate(batch):
                if j < len(results) and results[j] and results[j].get("abstract"):
                    abs_text = results[j]["abstract"]
                    for p in papers:
                        if p["doi"] == doi:
                            p["abstract"] = abs_text
                            break
        except Exception as e:
            print(f"  [WARN] Semantic Scholar batch: {e}")


def _extract(item: Dict, journal: str) -> Dict:
    title = (item.get("title") or [""])[0]

    authors = []
    for a in item.get("author", []):
        g, f = a.get("given", ""), a.get("family", "")
        authors.append(f"{g} {f}" if g and f else (f or ""))
    author_str = "; ".join(a for a in authors if a) or "N/A"

    doi = item.get("DOI", "")
    url = f"https://doi.org/{doi}" if doi else ""

    published = ""
    for key in ("published-print", "published-online", "created"):
        date = item.get(key)
        if date and "date-parts" in date and date["date-parts"]:
            published = "-".join(str(x) for x in date["date-parts"][0])
            break

    abstract = re.sub(r"<[^>]+>", "", item.get("abstract") or "")
    abstract = abstract.strip()
    if len(abstract) > 300:
        abstract = abstract[:297] + "…"

    return {"title": title, "authors": author_str, "journal": journal,
            "doi": doi, "url": url, "published": published, "abstract": abstract}


def filter_papers(papers: List[Dict], filter_type: Optional[str]) -> List[Dict]:
    """Post-filter papers for topic-specific relevance."""
    if filter_type is None:
        return papers

    if filter_type == "knowledge_graph":
        kg = ["knowledge graph", "knowledge base", "ontology", "semantic network",
              "knowledge representation", "knowledge-driven", "knowledge-guided",
              "knowledge embedding", "knowledge-aware", "semantic model",
              "knowledge-enhanced", "graph-based knowledge", "knowledge retrieval",
              "knowledge-aware", "entity alignment", "relation extraction"]
        ctx = ["earthquake", "seismic", "seismolog", "fault", "tsunami",
               "geoscience", "geophysic", "tectonic", "earth", "geolog",
               "subduction", "magnitude", "hypocenter", "epicenter"]
        kept = []
        for p in papers:
            t = (p["title"] + " " + p["abstract"]).lower()
            if any(k in t for k in kg) and any(c in t for c in ctx):
                kept.append(p)
        return kept

    if filter_type == "statistical_seismology":
        terms = ["statistical seismology", "earthquake statistics",
                 "seismicity statistics", "statistical model", "statistical analysis",
                 "stochastic model", "point process", "etas model", "clustering model",
                 "earthquake clustering", "seismic hazard", "recurrence interval",
                 "interevent time", "magnitude distribution", "b value", "b-value",
                 "gutenberg-richter", "aftershock decay", "omori"]
        kept = []
        for p in papers:
            t = (p["title"] + " " + p["abstract"]).lower()
            if any(term in t for term in terms):
                kept.append(p)
        return kept

    return papers


# ── Email HTML ───────────────────────────────────────────────────────

def build_html(results: List[Dict]) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    total = sum(len(jr["papers"]) for sec in results for jr in sec["journals"])

    lines = [f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:750px;margin:0 auto;padding:24px;">
<h1 style="color:#1a1a2e;font-size:22px;">&#127757; Earthquake &amp; Seismology Paper Digest</h1>
<p style="color:#666;font-size:13px;">{date_str} &middot; {total} article{'s' if total != 1 else ''}</p>
<hr style="border:none;border-top:1px solid #eee;">"""]

    for sec in results:
        section_total = sum(len(jr["papers"]) for jr in sec["journals"])
        if section_total == 0:
            continue

        lines.append(f"""
<h2 style="color:#16213e;font-size:18px;margin-top:28px;">&#128220; {sec['title']}</h2>""")

        for jr in sec["journals"]:
            papers = jr["papers"]
            if not papers:
                continue
            lines.append(f"""
<h3 style="font-size:15px;color:#333;margin:16px 0 6px;">{jr['name']} <span style="font-weight:400;color:#888;font-size:13px;">({len(papers)} paper{'s' if len(papers) != 1 else ''})</span></h3>
<table style="width:100%;border-collapse:collapse;">""")
            for i, p in enumerate(papers):
                summary = p["abstract"] or "No abstract available."
                lines.append(f"""<tr><td style="padding:10px 0;border-bottom:1px solid #f0f0f0;">
<div style="font-size:13px;font-weight:600;"><a href="{p['url']}" style="color:#0066cc;text-decoration:none;">{i+1}. {p['title']}</a></div>
<div style="font-size:11px;color:#888;margin-top:2px;">{p['authors']} &middot; {p['published']}</div>
<div style="font-size:12px;color:#444;margin-top:4px;line-height:1.5;">{summary}</div>
</td></tr>""")
            lines.append("</table>")

    lines.append(f"""<hr style="border:none;border-top:1px solid #eee;margin-top:24px;">
<p style="color:#999;font-size:11px;">Generated by Paper Tracker &middot; <a href="https://api.crossref.org" style="color:#999;">CrossRef</a> + <a href="https://www.semanticscholar.org" style="color:#999;">Semantic Scholar</a></p>
</body></html>""")
    return "\n".join(lines)


# ── Email sending ────────────────────────────────────────────────────

def send_email(html: str):
    ctx = ssl.create_default_context()
    with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"])) as s:
        s.starttls(context=ctx)
        s.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"&#127757; Paper Digest — {datetime.now().strftime('%Y-%m-%d')}"
        msg["From"] = os.environ["EMAIL_FROM"]
        msg["To"] = os.environ["EMAIL_TO"]
        msg.attach(MIMEText(html, "html"))
        s.sendmail(os.environ["EMAIL_FROM"], os.environ["EMAIL_TO"], msg.as_string())


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(f"Paper Tracker — {datetime.now().isoformat()}")
    print("=" * 60)

    since_date = get_since_date()
    if since_date:
        print(f"Since last run: {since_date}")
    else:
        print("First run — fetching recent papers only")

    results = []
    for sec in SECTIONS:
        print(f"\n── {sec['title']} ──")
        sec_results = {"title": sec["title"], "journals": []}
        for j in sec["journals"]:
            print(f"  {j['name']} …", end=" ", flush=True)
            papers = fetch_papers(j, sec["query"], sec.get("filter"), since_date)
            if sec["max"] is not None:
                papers = papers[: sec["max"]]
            sec_results["journals"].append({"name": j["name"], "papers": papers})
            print(f"{len(papers)} paper{'s' if len(papers) != 1 else ''}")
        results.append(sec_results)

    total = sum(len(jr["papers"]) for sec in results for jr in sec["journals"])
    print(f"\nTotal: {total} paper(s)")

    if not total:
        print("No new papers — skip email.")
        return

    html = build_html(results)
    send_email(html)
    save_run_date()
    print("Last run date saved.")
    print("Email sent successfully!")


if __name__ == "__main__":
    main()
