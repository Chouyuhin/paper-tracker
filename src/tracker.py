#!/usr/bin/env python3
"""
Paper Tracker — Daily digest of earthquake-related articles from
PRL, PRE, and JASA via the CrossRef API.
"""

import os
import re
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Dict

import requests

JOURNALS = [
    {"name": "Physical Review Letters", "issns": ["0031-9007", "1079-7114"]},
    {"name": "Physical Review E",       "issns": ["1539-3755", "2470-0053"]},
    {"name": "J. American Statistical Association", "issns": ["0162-1459", "1537-274X"]},
]

CROSSREF_URL = "https://api.crossref.org/works"
SEMANTIC_URL = "https://api.semanticscholar.org/graph/v1/paper"
QUERY = "earthquake"
PER_PAGE = 30


def fetch_papers(journal_name: str, issns: List[str]) -> List[Dict]:
    papers: List[Dict] = []
    for issn in issns:
        params = {
            "query": QUERY,
            "filter": f"issn:{issn}",
            "sort": "published",
            "order": "desc",
            "rows": PER_PAGE,
        }
        try:
            r = requests.get(CROSSREF_URL, params=params, timeout=30)
            r.raise_for_status()
            for item in r.json().get("message", {}).get("items", []):
                papers.append(_extract(item, journal_name))
        except Exception as e:
            print(f"  [WARN] ISSN {issn}: {e}")

    seen: set = set()
    unique: List[Dict] = []
    for p in papers:
        if p["doi"] and p["doi"] not in seen:
            seen.add(p["doi"])
            unique.append(p)

    enrich_abstracts(unique)
    return unique


def enrich_abstracts(papers: List[Dict]):
    """Fill missing/weak abstracts from Semantic Scholar (free, no key needed)."""
    dois = [p["doi"] for p in papers if p["doi"] and (not p["abstract"] or len(p["abstract"]) < 50)]
    if not dois:
        return

    # Batch query: POST /paper/search/batch with list of DOIs
    batch_size = 20
    for i in range(0, len(dois), batch_size):
        batch = dois[i : i + batch_size]
        params = [("fields", "abstract")]
        body = {"ids": [f"DOI:{d}" for d in batch]}
        try:
            r = requests.post(
                f"{SEMANTIC_URL}/batch",
                params=params,
                json=body,
                timeout=30,
            )
            if not r.ok:
                continue
            results = r.json()
            lookup = {}
            for entry in results:
                if entry and entry.get("abstract"):
                    lookup[entry.get("paperId") or ""] = entry["abstract"]

            # Map DOI back to our papers
            for j, doi in enumerate(batch):
                if j < len(results) and results[j] and results[j].get("abstract"):
                    abstract = results[j]["abstract"]
                    for p in papers:
                        if p["doi"] == doi:
                            p["abstract"] = abstract
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


def build_html(all_papers: Dict[str, List[Dict]]) -> str:
    total = sum(len(v) for v in all_papers.values())
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    lines = [f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:720px;margin:0 auto;padding:24px;">
<h1 style="color:#1a1a2e;font-size:22px;">&#127757; Earthquake Paper Digest</h1>
<p style="color:#666;font-size:13px;">{now} &middot; {total} article{'s' if total != 1 else ''} found</p>
<hr style="border:none;border-top:1px solid #eee;">
"""]

    for jname, papers in all_papers.items():
        if not papers:
            continue
        lines.append(f'<h2 style="color:#16213e;margin-top:28px;font-size:17px;">{jname}</h2><table style="width:100%;border-collapse:collapse;">')
        for i, p in enumerate(papers):
            summary = p["abstract"] or "No abstract available."
            abs_text = f'<div style="font-size:12px;color:#444;margin-top:6px;line-height:1.5;">{summary}</div>'
            lines.append(f"""<tr><td style="padding:14px 0;border-bottom:1px solid #f0f0f0;">
<div style="font-size:14px;font-weight:600;"><a href="{p['url']}" style="color:#0066cc;text-decoration:none;">{i+1}. {p['title']}</a></div>
<div style="font-size:12px;color:#888;margin-top:4px;">{p['authors']} &middot; {p['published']}</div>
{abs_text}
</td></tr>""")
        lines.append("</table>")

    lines.append(f"""<hr style="border:none;border-top:1px solid #eee;margin-top:24px;">
<p style="color:#999;font-size:11px;">Generated by Paper Tracker &middot; <a href="https://api.crossref.org" style="color:#999;">CrossRef API</a></p>
</body></html>""")
    return "\n".join(lines)


def send_email(html: str):
    ctx = ssl.create_default_context()
    with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"])) as s:
        s.starttls(context=ctx)
        s.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"&#127757; Earthquake Paper Digest — {datetime.now().strftime('%Y-%m-%d')}"
        msg["From"] = os.environ["EMAIL_FROM"]
        msg["To"] = os.environ["EMAIL_TO"]
        msg.attach(MIMEText(html, "html"))
        s.sendmail(os.environ["EMAIL_FROM"], os.environ["EMAIL_TO"], msg.as_string())


def main():
    print("=" * 60)
    print(f"Paper Tracker — {datetime.now().isoformat()}")
    print("=" * 60)

    all_papers = {}
    for j in JOURNALS:
        print(f"\nFetching {j['name']} …")
        all_papers[j["name"]] = fetch_papers(j["name"], j["issns"])
        print(f"  → {len(all_papers[j['name']])} papers")

    total = sum(len(v) for v in all_papers.values())
    print(f"\nTotal: {total} paper(s)")

    if not total:
        print("No papers found — skip email.")
        return

    html = build_html(all_papers)
    send_email(html)
    print("Email sent successfully!")


if __name__ == "__main__":
    main()
