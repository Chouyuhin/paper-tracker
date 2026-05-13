#!/usr/bin/env python3
"""
Paper Tracker — Daily digest of earthquake-related articles from
multiple journals via the CrossRef API.
"""

import os
import re
import smtplib
import ssl
import xml.etree.ElementTree as ET
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
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
    {
        "tag":      "arXiv",
        "title":    "arXiv Preprints",
        "source":   "arxiv",
        "query":    "earthquake",
        "categories": ["physics.geo-ph", "stat.AP"],
        "max":      5,
        "filter":   None,
    },
    {
        "tag":      "Clustering",
        "title":    "Earthquake Clustering & Declustering",
        "query":    "earthquake clustering declustering",
        "journals": NEW_JOURNALS,
        "max":      3,
        "filter":   "clustering",
    },
    {
        "tag":      "EQChars",
        "title":    "Earthquake Source & Characteristics",
        "query":    "earthquake source rupture characteristics",
        "journals": NEW_JOURNALS,
        "max":      3,
        "filter":   "characteristics",
    },
]

CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper"
ARXIV_URL = "http://export.arxiv.org/api/query"
PER_PAGE = 30
LAST_RUN_FILE = ".last_run_date"


# ── Date tracking ────────────────────────────────────────────────────

def get_since_date() -> Optional[str]:
    """Read the last successful run date. None = first run."""
    try:
        with open(LAST_RUN_FILE) as f:
            val = f.read().strip()
            return val if val else None
    except FileNotFoundError:
        return None


def save_run_date():
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.now().strftime("%Y-%m-%d"))


def pub_date_after(pub_str: str, since: str) -> bool:
    """Check if a publication date string is after the cutoff."""
    if not pub_str:
        return False
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            pub = datetime.strptime(pub_str, fmt)
            cutoff = datetime.strptime(since, "%Y-%m-%d")
            return pub >= cutoff
        except ValueError:
            continue
    return False


# ── Fetching ─────────────────────────────────────────────────────────

def fetch_papers(journal: Dict, query: str, filter_type: Optional[str] = None,
                 since_date: Optional[str] = None) -> List[Dict]:
    papers: List[Dict] = []
    for issn in journal["issns"]:
        params = {
            "query": query,
            "filter": f"issn:{issn}",
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


# ── arXiv API ────────────────────────────────────────────────────────

ARXIV_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_arxiv(query: str, categories: List[str], max_results: int = 20) -> List[Dict]:
    """Fetch papers from arXiv API."""
    cat_q = "+OR+".join(f"cat:{c}" for c in categories)
    full_query = f"({cat_q})+AND+all:{query}"
    url = f"{ARXIV_URL}?search_query={full_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"

    papers: List[Dict] = []
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        root = ET.fromstring(r.content)

        for entry in root.findall("atom:entry", ARXIV_NS):
            title = re.sub(r"\s+", " ", (entry.find("atom:title", ARXIV_NS).text or "").strip())
            authors = [a.find("atom:name", ARXIV_NS).text for a in entry.findall("atom:author", ARXIV_NS)]
            author_str = "; ".join(authors[:5]) + ("; et al." if len(authors) > 5 else "")

            doi_el = entry.find("arxiv:doi", ARXIV_NS)
            doi = doi_el.text if doi_el is not None else ""
            arxiv_id = re.sub(r"^https?://arxiv\.org/abs/", "", entry.find("atom:id", ARXIV_NS).text.strip())
            url_link = f"https://arxiv.org/abs/{arxiv_id}"

            published = (entry.find("atom:published", ARXIV_NS).text or "")[:10]

            abstract = re.sub(r"\s+", " ", (entry.find("atom:summary", ARXIV_NS).text or "").strip())
            if len(abstract) > 300:
                abstract = abstract[:297] + "…"

            papers.append({
                "title": title, "authors": author_str, "journal": "arXiv",
                "doi": doi, "url": url_link, "published": published,
                "abstract": abstract,
            })
    except Exception as e:
        print(f"  [WARN] arXiv API: {e}")

    return papers


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

    if filter_type == "clustering":
        terms = ["clustering", "declustering", "etas", "epidemic type",
                 "space-time cluster", "spatiotemporal cluster", "earthquake swarm",
                 "cluster analysis", "nearest-neighbor", "nearest neighbor",
                 "epidemic-type", "branching model", "triggering"]
        ctx = ["earthquake", "seismic", "seismolog", "fault", "aftershock",
               "tectonic", "hypocenter", "epicenter", "magnitude"]
        kept = []
        for p in papers:
            t = (p["title"] + " " + p["abstract"]).lower()
            if any(term in t for term in terms) and any(c in t for c in ctx):
                kept.append(p)
        return kept

    if filter_type == "characteristics":
        terms = ["source parameter", "rupture", "magnitude distribution",
                 "stress drop", "source characteristic", "fault slip",
                 "rupture propagation", "seismic moment", "slip distribution",
                 "moment magnitude", "corner frequency", "radiation pattern",
                 "source time function", "rupture velocity", "slip rate",
                 "magnitude scaling", "ground motion", "attenuation",
                 "site effect", "directivity"]
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
        print(f"Papers published since: {since_date}")
    else:
        since_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        print(f"First run — fetching papers since {since_date}")

    results = []
    for sec in SECTIONS:
        print(f"\n── {sec['title']} ──")
        sec_results = {"title": sec["title"], "journals": []}

        if sec.get("source") == "arxiv":
            papers = fetch_arxiv(sec["query"], sec["categories"])
            papers = [p for p in papers if pub_date_after(p["published"], since_date)]
            if sec["max"] is not None:
                papers = papers[: sec["max"]]
            sec_results["journals"].append({"name": "arXiv", "papers": papers})
            print(f"  arXiv … {len(papers)} paper{'s' if len(papers) != 1 else ''}")
        else:
            for j in sec["journals"]:
                print(f"  {j['name']} …", end=" ", flush=True)
                papers = fetch_papers(j, sec["query"], sec.get("filter"))
                papers = [p for p in papers if pub_date_after(p["published"], since_date)]
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
