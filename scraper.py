"""Fetch job pages, verify they're live, dedupe, and compute math-based match scores."""
import asyncio
import json
import re
from dataclasses import dataclass, field
from datetime import date
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from resume_parser import extract_skills
from search_engine import SearchHit

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")

CLOSED_PHRASES = [
    "no longer accepting applications", "no longer available", "position has been filled",
    "job is no longer", "this job has expired", "posting has closed", "job not found",
    "page not found", "this position is closed", "no longer open", "we couldn't find that job",
]
GENERIC_TITLE_WORDS = {"senior", "junior", "lead", "manager", "engineer", "analyst", "developer",
                       "specialist", "principal", "staff", "associate", "director"}


@dataclass
class Job:
    url: str
    title: str
    company: str = ""
    location: str = ""
    source: str = ""
    description: str = ""
    verified: bool = False
    score: float = 0.0
    matched: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    category: str = "direct"  # direct | direct_translatable | subtle_translatable
    rationale: str = ""


def infer_company(url: str) -> str:
    parsed = urlparse(url)
    host, parts = parsed.netloc.lower(), [p for p in parsed.path.split("/") if p]
    if any(d in host for d in ("greenhouse.io", "lever.co", "ashbyhq.com")):
        return parts[0].replace("-", " ").title() if parts else host
    if "myworkdayjobs.com" in host:
        return host.split(".")[0].replace("-", " ").title()
    host = host.removeprefix("www.").removeprefix("careers.").removeprefix("jobs.")
    return host.split(".")[0].title()


def _find_jobposting(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except Exception:  # noqa: BLE001
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                t = item.get("@type")
                if t == "JobPosting" or (isinstance(t, list) and "JobPosting" in t):
                    return item
                stack.extend(v for v in item.values() if isinstance(v, (dict, list)))
            elif isinstance(item, list):
                stack.extend(item)
    return None


def _jp_location(jp) -> str:
    if str(jp.get("jobLocationType", "")).upper() == "TELECOMMUTE":
        return "Remote"
    loc = jp.get("jobLocation")
    loc = loc[0] if isinstance(loc, list) and loc else loc
    if isinstance(loc, dict):
        addr = loc.get("address", {})
        if isinstance(addr, dict):
            country = addr.get("addressCountry")
            country = country.get("name") if isinstance(country, dict) else country
            return ", ".join(str(x) for x in (addr.get("addressLocality"), addr.get("addressRegion"), country) if x)
    return ""


def _expired(jp) -> bool:
    vt = jp.get("validThrough")
    if not vt:
        return False
    try:
        return date.fromisoformat(str(vt)[:10]) < date.today()
    except ValueError:
        return False


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_job_page(html: str, hit: SearchHit):
    """Returns a verified Job, or None if the page is closed / not a job posting."""
    soup = BeautifulSoup(html, "html.parser")
    jp = _find_jobposting(soup)
    if jp and _expired(jp):
        return None
    for s in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        s.decompose()
    page_text = _clean(soup.get_text(" "))
    low = page_text[:3000].lower()
    if any(p in low for p in CLOSED_PHRASES) and len(page_text) < 1500:
        return None

    if jp:
        title = _clean(str(jp.get("title", ""))) or hit.title
        org = jp.get("hiringOrganization")
        company = (org.get("name") if isinstance(org, dict) else org) or infer_company(hit.url)
        desc_html = jp.get("description", "")
        desc = _clean(BeautifulSoup(desc_html, "html.parser").get_text(" ")) if desc_html else page_text
        location = _jp_location(jp)
        verified = True
    else:
        h1 = soup.find("h1")
        og = soup.find("meta", property="og:title")
        title = _clean(h1.get_text(" ")) if h1 else (og.get("content") if og else hit.title)
        company, desc, location = infer_company(hit.url), page_text, ""
        looks_like_job = bool(re.search(r"\bapply\b", low + page_text[-3000:].lower())) and bool(
            re.search(r"responsibilit|qualification|requirement|what you", page_text, re.I))
        if hit.source == "Corporate Career Portals" and not looks_like_job:
            return None  # likely a listing/landing page, not a posting
        verified = looks_like_job
    return Job(url=hit.url, title=title[:140], company=str(company)[:80], location=location,
               source=hit.source, description=desc[:6000], verified=verified)


def _fallback_job(hit: SearchHit) -> Job:
    """JS-rendered pages (e.g., Workday) often can't be scraped; use the search snippet."""
    return Job(url=hit.url, title=hit.title.split(" - ")[0][:140], company=infer_company(hit.url),
               source=hit.source, description=f"{hit.title}. {hit.snippet}", verified=False)


async def _fetch(client, sem, hit):
    async with sem:
        try:
            r = await client.get(hit.url)
        except httpx.HTTPError:
            return _fallback_job(hit)
    if r.status_code in (404, 410) or "error=true" in str(r.url):
        return None
    if r.status_code >= 400:
        return _fallback_job(hit)
    try:
        return parse_job_page(r.text, hit)
    except Exception:  # noqa: BLE001
        return _fallback_job(hit)


async def collect_jobs(hits, progress=None, concurrency=8):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(headers={"User-Agent": UA}, timeout=15, follow_redirects=True) as client:
        tasks = [asyncio.create_task(_fetch(client, sem, h)) for h in hits]
        jobs = []
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            job = await fut
            if job:
                jobs.append(job)
            if progress:
                progress(i / max(len(tasks), 1), f"Verifying postings ({i}/{len(tasks)})")
    return dedupe(jobs)


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


def dedupe(jobs):
    seen_url, seen_key, out = set(), set(), []
    for j in jobs:
        u = j.url.split("?")[0].rstrip("/")
        k = (_norm(j.title), _norm(j.company), _norm(j.location))
        if u in seen_url or k in seen_key:
            continue
        seen_url.add(u)
        seen_key.add(k)
        out.append(j)
    return out


def score_jobs(jobs, resume_text: str, resume_skills: list):
    """Math Mode: blend TF-IDF cosine, skill coverage and Jaccard. Zero API cost."""
    if not jobs:
        return jobs
    docs = [resume_text] + [f"{j.title}. {j.description}" for j in jobs]
    try:
        tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=6000, sublinear_tf=True)
        mat = tfidf.fit_transform(docs)
        sims = cosine_similarity(mat[0], mat[1:]).ravel()
    except ValueError:
        sims = [0.0] * len(jobs)
    rskills = set(resume_skills)
    for j, sim in zip(jobs, sims):
        jskills = set(extract_skills(f"{j.title}. {j.description}"))
        inter = rskills & jskills
        coverage = len(inter) / len(jskills) if jskills else 0.0
        jaccard = len(inter) / len(rskills | jskills) if (rskills | jskills) else 0.0
        tfidf_norm = min(1.0, float(sim) / 0.35)  # cosine on long docs rarely exceeds ~0.35
        j.score = round(100 * (0.4 * tfidf_norm + 0.4 * coverage + 0.2 * min(1.0, jaccard * 3)), 1)
        j.matched = sorted(inter)
        j.missing = sorted(jskills - rskills)
        j.rationale = (f"Your resume covers {len(inter)} of {len(jskills)} skills detected in this posting."
                       if jskills else "Few recognizable skill keywords in the posting; score is text-similarity based.")
    return jobs


def heuristic_category(job: Job, target_titles) -> str:
    """Math Mode approximation of the three tiers (AI Mode does this properly)."""
    tw = {w for t in target_titles for w in re.findall(r"[a-z]+", t.lower())
          if len(w) > 3 and w not in GENERIC_TITLE_WORDS}
    title_hit = bool(tw & set(re.findall(r"[a-z]+", job.title.lower())))
    if (title_hit and job.score >= 40) or job.score >= 65:
        return "direct"
    if job.score >= 40 or title_hit:
        return "direct_translatable"
    return "subtle_translatable"
