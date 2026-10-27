"""Builds ATS 'footprint' queries and runs them through Serper.dev."""
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

SERPER_URL = "https://google.serper.dev/search"

ATS_FOOTPRINTS = {
    "Greenhouse": "(site:boards.greenhouse.io OR site:job-boards.greenhouse.io)",
    "Lever": "site:jobs.lever.co",
    "Ashby": "site:jobs.ashbyhq.com",
    "Workday": "site:myworkdayjobs.com",
    "Corporate Career Portals": "(inurl:careers OR inurl:jobs) -site:linkedin.com -site:indeed.com "
                                "-site:glassdoor.com -site:ziprecruiter.com -site:monster.com",
}

BOARD_DOMAINS = ("linkedin.com", "indeed.com", "glassdoor.com", "ziprecruiter.com", "monster.com",
                 "simplyhired.com", "talent.com", "jooble.org")


class SearchError(Exception):
    """User-friendly search failure."""


@dataclass
class SearchHit:
    url: str
    title: str
    snippet: str
    source: str


def classify_source(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "greenhouse.io" in host:
        return "Greenhouse"
    if "lever.co" in host:
        return "Lever"
    if "ashbyhq.com" in host:
        return "Ashby"
    if "myworkdayjobs.com" in host:
        return "Workday"
    return "Corporate Career Portals"


def build_queries(titles, skills, location="", remote="Any", ats=None, level="Any", max_queries=10):
    ats = ats or list(ATS_FOOTPRINTS)
    terms = list(titles[:4]) + list(skills[:4])
    suffix = ""
    if location:
        suffix += f' "{location}"'
    if remote == "Remote only":
        suffix += ' "Remote"'
    if level != "Any":
        suffix += f' "{level}"'
    queries = []
    for term in terms:
        for name in ats:
            queries.append(f'{ATS_FOOTPRINTS[name]} "{term}"{suffix}')
    return queries[:max_queries]


def run_searches(queries, api_key, progress=None, results_per_query=10):
    """Returns (hits, warnings). Stops gracefully on rate limits / bad keys."""
    if not api_key:
        raise SearchError("No Serper API key found. Add one in the sidebar or in Streamlit secrets.")
    hits, seen, warnings = [], set(), []
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    with httpx.Client(timeout=20) as client:
        for i, q in enumerate(queries):
            if progress:
                progress(i / max(len(queries), 1), f"Searching ({i + 1}/{len(queries)})")
            try:
                r = client.post(SERPER_URL, headers=headers, json={"q": q, "num": results_per_query})
            except httpx.HTTPError:
                warnings.append(f"Network problem on query {i + 1}; skipped.")
                continue
            if r.status_code in (401, 403):
                raise SearchError("Serper rejected the API key. Please double-check it.")
            if r.status_code == 429 or r.status_code == 402:
                warnings.append("Search API limit reached; showing results gathered so far.")
                break
            if r.status_code != 200:
                warnings.append(f"Search error {r.status_code} on query {i + 1}; skipped.")
                continue
            for item in r.json().get("organic", []):
                url = (item.get("link") or "").split("#")[0]
                host = urlparse(url).netloc.lower()
                if not url or url in seen or any(b in host for b in BOARD_DOMAINS):
                    continue
                seen.add(url)
                hits.append(SearchHit(url, item.get("title", ""), item.get("snippet", ""), classify_source(url)))
            time.sleep(0.2)
    return hits, warnings
