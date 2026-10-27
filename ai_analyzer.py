"""AI Strategic Mode: Claude/OpenAI categorize jobs into 3 tiers with rationale and gaps."""
import json
import re
import time

from scraper import Job

CATEGORIES = {"direct", "direct_translatable", "subtle_translatable", "exclude"}
DEFAULT_MODELS = {"Anthropic Claude": "claude-sonnet-5-5", "OpenAI": "gpt-4o-mini"}

SYSTEM_PROMPT = """You are a senior career strategist. Given a candidate resume summary and job postings,
classify each job into exactly one category:
- "direct": in-field role matching the candidate's current domain.
- "direct_translatable": high-overlap career pivot (e.g. Metadata Architecture -> Information Architecture).
- "subtle_translatable": adjacent/bridge role where non-traditional skills give the candidate an edge.
- "exclude": poor fit, or not an actual job posting.
Respond with ONLY valid JSON (no markdown fences):
{"results":[{"id":<int>,"category":"...","score":<0-100>,"rationale":"<=2 sentences, strategic and specific",
"matched_skills":["..."],"gap_keywords":["..."]}]}
Include every id given. Keep lists to at most 8 items each."""


class AIError(Exception):
    """User-friendly AI failure."""


def _call_llm(provider, api_key, model, user_prompt):
    if provider == "Anthropic Claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        try:
            msg = client.messages.create(model=model, max_tokens=3500, system=SYSTEM_PROMPT,
                                         messages=[{"role": "user", "content": user_prompt}])
        except anthropic.AuthenticationError:
            raise AIError("The Anthropic API key was rejected. Please check it.")
        except anthropic.RateLimitError:
            raise AIError("RATE_LIMIT")
        except anthropic.APIError as exc:
            raise AIError(f"Anthropic API error: {exc}")
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")

    import openai
    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}])
    except openai.AuthenticationError:
        raise AIError("The OpenAI API key was rejected. Please check it.")
    except openai.RateLimitError:
        raise AIError("RATE_LIMIT")
    except openai.APIError as exc:
        raise AIError(f"OpenAI API error: {exc}")
    return resp.choices[0].message.content or ""


def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def analyze_jobs(provider, api_key, resume_text, skills, titles, jobs, model=None,
                 batch_size=4, progress=None):
    """Mutates and returns jobs (excluded ones removed). Returns (jobs, warnings)."""
    if not api_key:
        raise AIError("AI Strategic Mode needs an API key (sidebar or Streamlit secrets).")
    model = model or DEFAULT_MODELS[provider]
    warnings, kept = [], []
    header = (f"RESUME SUMMARY:\n{resume_text[:3500]}\n\nTARGET TITLES: {', '.join(titles)}\n"
              f"SKILLS: {', '.join(skills)}\n\nJOBS:\n")
    batches = [jobs[i:i + batch_size] for i in range(0, len(jobs), batch_size)]
    for bi, batch in enumerate(batches):
        if progress:
            progress(bi / max(len(batches), 1), f"AI analysis ({bi + 1}/{len(batches)})")
        listing = "\n\n".join(
            f"[id={i}] {j.title} @ {j.company} ({j.location or 'location n/a'})\n{j.description[:1500]}"
            for i, j in enumerate(batch))
        raw = None
        for attempt in range(2):
            try:
                raw = _call_llm(provider, api_key, model, header + listing)
                break
            except AIError as exc:
                if str(exc) == "RATE_LIMIT" and attempt == 0:
                    time.sleep(8)
                    continue
                if str(exc) == "RATE_LIMIT":
                    warnings.append("AI rate limit hit; remaining jobs kept with Math Mode scores.")
                    kept.extend(batch[0:0])
                    for rest in batches[bi:]:
                        kept.extend(rest)
                    return kept, warnings
                raise
        try:
            results = _parse_json(raw or "").get("results", [])
        except Exception:  # noqa: BLE001
            warnings.append(f"Batch {bi + 1}: AI reply unreadable; kept Math Mode scores.")
            kept.extend(batch)
            continue
        by_id = {r.get("id"): r for r in results if isinstance(r, dict)}
        for i, j in enumerate(batch):
            r = by_id.get(i)
            if not r:
                kept.append(j)
                continue
            cat = r.get("category", "subtle_translatable")
            if cat == "exclude" or cat not in CATEGORIES:
                continue
            j.category = cat
            try:
                j.score = float(r.get("score", j.score))
            except (TypeError, ValueError):
                pass
            j.rationale = r.get("rationale") or j.rationale
            j.matched = [str(x) for x in r.get("matched_skills", [])][:8] or j.matched
            j.missing = [str(x) for x in r.get("gap_keywords", [])][:8]
            kept.append(j)
    return kept, warnings
