"""Hidden Jobs Finder - Streamlit app."""
import asyncio
import hashlib

import pandas as pd
import streamlit as st

from ai_analyzer import DEFAULT_MODELS, AIError, analyze_jobs
from resume_parser import ResumeError, parse_resume
from scraper import collect_jobs, heuristic_category, score_jobs
from search_engine import ATS_FOOTPRINTS, SearchError, build_queries, run_searches

st.set_page_config(page_title="Hidden Jobs Finder", page_icon="🕵️", layout="wide")

MAX_AI_JOBS = 30
TABS = [("direct", "🎯 Direct Matches"), ("direct_translatable", "🔀 Direct Translatable"),
        ("subtle_translatable", "🌉 Subtle Translatable")]


def secret(name: str) -> str:
    try:
        return st.secrets.get(name, "") or ""
    except Exception:  # noqa: BLE001 - no secrets file locally
        return ""


def add_terms(kind: str):
    raw = st.session_state.get(f"add_{kind}", "")
    new = [t.strip() for t in raw.split(",") if t.strip()]
    st.session_state[f"{kind}_options"] = sorted(set(st.session_state[f"{kind}_options"]) | set(new))
    st.session_state[f"{kind}_sel"] = sorted(set(st.session_state[f"{kind}_sel"]) | set(new))
    st.session_state[f"add_{kind}"] = ""


def render_card(j):
    with st.container(border=True):
        left, right = st.columns([5, 1])
        left.markdown(f"**{j.title}** · {j.company}")
        left.caption(f"📍 {j.location or 'Location not listed'} · {j.source}"
                     + ("" if j.verified else " · ⚠️ unverified (page couldn't be fully read)"))
        right.metric("Match", f"{j.score:.0f}%")
        if j.rationale:
            st.write(j.rationale)
        if j.matched:
            st.markdown("✅ " + " ".join(f":green-background[{s}]" for s in j.matched[:12]))
        if j.missing:
            st.markdown("🔴 Gaps: " + " ".join(f":red-background[{s}]" for s in j.missing[:12]))
        st.link_button("Apply ↗", j.url)  # opens in a new tab


# ---------------------------------------------------------------- SIDEBAR
st.session_state.setdefault("skills_options", [])
st.session_state.setdefault("skills_sel", [])
st.session_state.setdefault("titles_options", [])
st.session_state.setdefault("titles_sel", [])

with st.sidebar:
    st.title("🕵️ Hidden Jobs Finder")
    upload = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if upload is not None:
        data = upload.getvalue()
        digest = hashlib.md5(data).hexdigest()
        if st.session_state.get("resume_hash") != digest:
            try:
                profile = parse_resume(data)
                st.session_state.update(
                    resume_hash=digest, profile=profile,
                    skills_options=profile.skills, skills_sel=profile.skills,
                    titles_options=profile.titles, titles_sel=profile.titles)
                if not profile.skills:
                    st.warning("No known skills detected - add some manually below.")
            except ResumeError as exc:
                st.session_state.pop("profile", None)
                st.error(str(exc))

    if "profile" in st.session_state:
        st.subheader("Resume profile")
        st.multiselect("Job titles (remove with ✕)", st.session_state.titles_options, key="titles_sel")
        st.text_input("Add titles (comma-separated, Enter)", key="add_titles",
                      on_change=add_terms, args=("titles",))
        st.multiselect("Skills (remove with ✕)", st.session_state.skills_options, key="skills_sel")
        st.text_input("Add skills (comma-separated, Enter)", key="add_skills",
                      on_change=add_terms, args=("skills",))

    st.subheader("Analysis mode")
    mode = st.radio("Mode", ["Math Mode (100% Free)", "AI Strategic Mode"], label_visibility="collapsed")
    ai_mode = mode.startswith("AI")

    st.subheader("API keys")
    serper_secret = secret("SERPER_API_KEY")
    serper_key = serper_secret or st.text_input("Serper.dev API key", type="password")
    if serper_secret:
        st.caption("✅ Using Serper key from app secrets")
    provider, ai_key = "Anthropic Claude", ""
    if ai_mode:
        provider = st.selectbox("AI provider", list(DEFAULT_MODELS))
        s_name = "ANTHROPIC_API_KEY" if provider == "Anthropic Claude" else "OPENAI_API_KEY"
        ai_key = secret(s_name) or st.text_input(f"{provider} API key (BYOK)", type="password")
        if secret(s_name):
            st.caption("✅ Using AI key from app secrets")

    st.subheader("Search filters")
    location = st.text_input("Location (city, state, or country)")
    remote = st.selectbox("Remote status", ["Any", "Remote only", "On-site / Hybrid"])
    level = st.selectbox("Experience level", ["Any", "Entry", "Mid", "Senior", "Lead", "Principal", "Director"])
    st.caption("ATS networks")
    ats = [name for name in ATS_FOOTPRINTS if st.checkbox(name, value=True, key=f"ats_{name}")]
    max_queries = st.slider("Search depth (queries)", 3, 20, 8,
                            help="Each query uses 1 Serper credit. More queries = more results.")
    go = st.button("🔎 Scan the Web for Unlisted Jobs", type="primary", use_container_width=True)

# ---------------------------------------------------------------- RUN SCAN
if go:
    titles, skills = st.session_state.titles_sel, st.session_state.skills_sel
    if "profile" not in st.session_state:
        st.sidebar.error("Please upload a resume first.")
    elif not (titles or skills):
        st.sidebar.error("Add at least one job title or skill.")
    elif not ats:
        st.sidebar.error("Select at least one ATS network.")
    else:
        bar = st.progress(0.0, text="Starting...")
        warnings, jobs = [], []
        try:
            queries = build_queries(titles, skills, location, remote, ats, level, max_queries)
            hits, w = run_searches(queries, serper_key, lambda p, t: bar.progress(p * 0.35, text=t))
            warnings += w
            if not hits:
                st.session_state.results = []
                st.session_state.warnings = warnings
            else:
                jobs = asyncio.run(collect_jobs(hits, lambda p, t: bar.progress(0.35 + p * 0.4, text=t)))
                bar.progress(0.78, text="Scoring matches...")
                profile = st.session_state.profile
                jobs = score_jobs(jobs, profile.text, skills)
                if remote == "On-site / Hybrid":
                    jobs = [j for j in jobs if "remote" not in j.location.lower()]
                jobs.sort(key=lambda j: j.score, reverse=True)
                for j in jobs:
                    j.category = heuristic_category(j, titles)
                if ai_mode and jobs:
                    jobs, w = analyze_jobs(provider, ai_key, profile.text, skills, titles,
                                           jobs[:MAX_AI_JOBS],
                                           progress=lambda p, t: bar.progress(0.8 + p * 0.19, text=t))
                    warnings += w
                    if len(jobs) >= MAX_AI_JOBS:
                        warnings.append(f"AI analyzed the top {MAX_AI_JOBS} jobs by math score to control cost.")
                st.session_state.results = jobs
                st.session_state.warnings = warnings
                st.session_state.ran_ai = ai_mode
        except (SearchError, AIError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something unexpected happened ({type(exc).__name__}). Please try again.")
        bar.empty()

# ---------------------------------------------------------------- DASHBOARD
st.header("Discover hidden job postings")
results = st.session_state.get("results")
if results is None:
    st.info("👈 Upload your resume, pick your filters, and press **Scan the Web for Unlisted Jobs**.")
    st.stop()

for w in st.session_state.get("warnings", []):
    st.warning(w)
if not results:
    st.warning("No active postings found. Try more skills/titles, a broader location, more ATS networks, "
               "or a higher search depth.")
    st.stop()
if not st.session_state.get("ran_ai"):
    st.caption("Math Mode: tiers are estimated from title overlap and score. "
               "Switch to AI Strategic Mode for true transferable-skill analysis.")

by_cat = {k: [j for j in results if j.category == k] for k, _ in TABS}
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Jobs Found", len(results))
m2.metric("Direct Matches", len(by_cat["direct"]))
m3.metric("Strategic Pivots", len(by_cat["direct_translatable"]) + len(by_cat["subtle_translatable"]))
m4.metric("Average Match Score", f"{sum(j.score for j in results) / len(results):.0f}%")

tabs = st.tabs([f"{label} ({len(by_cat[k])})" for k, label in TABS])
for tab, (k, _) in zip(tabs, TABS):
    with tab:
        if not by_cat[k]:
            st.caption("Nothing in this category for this scan.")
        for j in sorted(by_cat[k], key=lambda x: x.score, reverse=True):
            render_card(j)

df = pd.DataFrame([{
    "Category": j.category, "Title": j.title, "Company": j.company, "Location": j.location,
    "Match %": j.score, "Rationale": j.rationale, "Matched Skills": ", ".join(j.matched),
    "Gap Keywords": ", ".join(j.missing), "Source": j.source, "Verified": j.verified, "Apply Link": j.url,
} for j in results])
st.download_button("⬇️ Download Results as CSV", df.to_csv(index=False).encode("utf-8"),
                   "hidden_jobs.csv", "text/csv")
