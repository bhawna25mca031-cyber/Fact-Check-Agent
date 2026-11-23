"""
Fact-Check Agent

Upload a PDF (or paste text), pull out the claims that can be checked, look
each one up on the live web and say whether it holds up.

AI model: any OpenAI-compatible API (Groq by default, free tier).
Web search: Tavily. Both keys live in the app's secrets, so visitors never
have to enter anything.
"""

import concurrent.futures as cf
import datetime as dt
import html
import json
import math
import os
import re
import time

import pandas as pd
import pdfplumber
import requests
import streamlit as st

st.set_page_config(page_title="Fact-Check Agent", page_icon="🔎", layout="wide")

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
TAVILY_URL = "https://api.tavily.com/search"
MAX_CHARS = 24_000  # keeps one request inside free-tier token limits
MAX_UPLOAD_MB = 10
WORKERS = 3  # free tiers have low rate limits, keep this small

STATUS = {
    "Verified": {"icon": "✓", "cls": "ok", "color": "#22c55e"},
    "Inaccurate": {"icon": "!", "cls": "warn", "color": "#f59e0b"},
    "False": {"icon": "✕", "cls": "bad", "color": "#ef4444"},
    "Unchecked": {"icon": "?", "cls": "na", "color": "#94a3b8"},
}

STRICTNESS = {
    "Lenient": "Small differences are fine (up to about 5%). Call it Verified if the substance is right.",
    "Standard": "Normal rounding is fine (1.99 billion vs 'about 2 billion'), but flag any real difference.",
    "Strict": "Numbers, dates and names must match exactly. Flag any difference, even a small one.",
}

SAMPLE_TEXT = """Quarterly Market Brief

[Page 1]
Google was founded in 1998 by Larry Page and Sergey Brin.
The first iPhone was released in 2005 and changed the smartphone market.
Bitcoin has a hard cap of 25 million coins.
Instagram has more than 500 million monthly active users.

[Page 2]
Mount Everest is 9,200 metres tall.
India overtook China as the world's most populous country in 2023.
Acme Quantum's QX-9000 chip delivers 4,000 error-corrected logical qubits.
The global population passed 6 billion people in 2024.
"""

# ------------------------------------------------------------------ styling

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(34,211,238,.16), transparent 55%),
    #070b17;
  background-attachment: fixed;
  color: #e8ecff;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2rem; max-width: 1150px; }
section[data-testid="stSidebar"] { background: rgba(14,20,42,.92); border-right: 1px solid rgba(255,255,255,.06); }

/* hero */
.hero { position: relative; overflow: hidden; padding: 38px 34px; border-radius: 22px;
  background: linear-gradient(135deg, rgba(124,92,255,.14), rgba(34,211,238,.07));
  border: 1px solid rgba(255,255,255,.08); margin-bottom: 22px; }
.orb { position: absolute; border-radius: 50%; filter: blur(46px); opacity: .55; animation: float 9s ease-in-out infinite; }
.o1 { width: 220px; height: 220px; background: #7c5cff; top: -70px; right: 8%; }
.o2 { width: 180px; height: 180px; background: #22d3ee; bottom: -80px; right: 30%; animation-delay: -4s; }
@keyframes float { 0%,100% { transform: translateY(0) scale(1); } 50% { transform: translateY(22px) scale(1.08); } }
.eyebrow { position: relative; font-size: .74rem; letter-spacing: .22em; color: #a5b4fc; font-weight: 600; }
.title { position: relative; font-family: 'Space Grotesk', sans-serif; font-size: 3rem; line-height: 1.05; margin: 8px 0 10px; font-weight: 700; color: #fff; }
.title span { background: linear-gradient(90deg, #8b7bff, #22d3ee, #8b7bff); background-size: 200% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: shine 5s linear infinite; }
@keyframes shine { to { background-position: -200% 0; } }
.sub { position: relative; max-width: 640px; color: #b6bfdc; font-size: 1.02rem; margin: 0 0 16px; }
.chips { position: relative; display: flex; flex-wrap: wrap; gap: 8px; }
.chips span { font-size: .78rem; padding: 5px 12px; border-radius: 999px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.1); color: #d7defa; }

/* how it works */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 6px 0 18px; }
.step { padding: 16px 18px; border-radius: 16px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08);
  transition: transform .2s, border-color .2s; animation: fadeUp .6s ease both; }
.step:hover { transform: translateY(-3px); border-color: rgba(124,92,255,.6); }
.step b { display: block; margin: 4px 0; color: #fff; }
.step p { margin: 0; color: #9aa6cc; font-size: .88rem; }
.step .n { display: inline-grid; place-items: center; width: 26px; height: 26px; border-radius: 8px; background: #7c5cff; color: #fff; font-weight: 700; font-size: .82rem; }

/* progress */
.pbar { height: 10px; border-radius: 99px; background: rgba(255,255,255,.08); overflow: hidden; }
.pbar > div { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #7c5cff, #22d3ee, #7c5cff);
  background-size: 200% 100%; animation: shimmer 1.3s linear infinite; transition: width .4s ease; }
@keyframes shimmer { to { background-position: -200% 0; } }
.plabel { display: flex; justify-content: space-between; font-size: .86rem; color: #b6bfdc; margin: 14px 0 8px; }
.tick { display: flex; align-items: center; gap: 10px; padding: 8px 12px; margin-top: 8px; border-radius: 10px;
  background: rgba(255,255,255,.04); font-size: .86rem; color: #cbd3f2; animation: fadeUp .4s ease both; }
.dot { width: 9px; height: 9px; border-radius: 50%; flex: none; }
.dot.ok { background: #22c55e; } .dot.warn { background: #f59e0b; } .dot.bad { background: #ef4444; } .dot.na { background: #94a3b8; }

/* dashboard */
.donut { max-width: 230px; margin: 0 auto; }
.donut svg { width: 100%; height: auto; }
.donut .seg { animation: grow 1.1s ease-out both; }
@keyframes grow { from { stroke-dasharray: 0 340; } }
.donut .score { font-family: 'Space Grotesk', sans-serif; font-size: 27px; font-weight: 700; fill: #fff; }
.donut .cap { font-size: 8.5px; fill: #9aa6cc; letter-spacing: .08em; }
.verdict { text-align: center; font-weight: 600; margin-top: 4px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.tile { padding: 18px 16px; border-radius: 16px; background: rgba(255,255,255,.045); border: 1px solid rgba(255,255,255,.08);
  animation: pop .55s cubic-bezier(.2,.9,.3,1.3) both; }
.tile .n { font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 700; line-height: 1; }
.tile .t { color: #9aa6cc; font-size: .82rem; margin-top: 6px; }
.tile.ok .n { color: #22c55e; } .tile.warn .n { color: #f59e0b; } .tile.bad .n { color: #ef4444; } .tile.all .n { color: #fff; }
@keyframes pop { from { opacity: 0; transform: scale(.85); } to { opacity: 1; transform: none; } }

/* claim cards */
.fc-card { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); border-left: 4px solid var(--c);
  border-radius: 14px; padding: 16px 18px; margin: 12px 0; animation: fadeUp .5s ease both; transition: transform .2s, box-shadow .2s; }
.fc-card:hover { transform: translateY(-2px); box-shadow: 0 12px 30px rgba(0,0,0,.35); }
.fc-card.ok { --c: #22c55e; } .fc-card.warn { --c: #f59e0b; } .fc-card.bad { --c: #ef4444; } .fc-card.na { --c: #94a3b8; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
.fc-top { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; }
.fc-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px; border-radius: 999px; font-size: .78rem; font-weight: 700; color: var(--c); background: rgba(255,255,255,.07); }
.fc-card.ok .fc-badge { background: rgba(34,197,94,.14); } .fc-card.warn .fc-badge { background: rgba(245,158,11,.14); }
.fc-card.bad .fc-badge { background: rgba(239,68,68,.14); } .fc-card.na .fc-badge { background: rgba(148,163,184,.14); }
.fc-meta { font-size: .76rem; color: #8b97bf; }
.fc-claim { font-size: 1.02rem; font-weight: 600; color: #fff; margin: 10px 0 8px; }
.fc-row { font-size: .92rem; color: #dbe1fb; margin-top: 6px; line-height: 1.5; }
.fc-row.dim { color: #9fabd1; }
.lbl { display: block; font-size: .68rem; text-transform: uppercase; letter-spacing: .12em; color: #7f8bb4; margin-bottom: 2px; }
.fc-fix { margin-top: 10px; padding: 10px 12px; border-radius: 10px; background: rgba(34,197,94,.08); border: 1px dashed rgba(34,197,94,.4); font-size: .92rem; color: #d7f5e1; }
.fc-src { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.fc-src a { font-size: .76rem; padding: 3px 10px; border-radius: 999px; background: rgba(124,92,255,.15); color: #c4b8ff !important; text-decoration: none; border: 1px solid rgba(124,92,255,.3); }
.fc-src a:hover { background: rgba(124,92,255,.3); }

/* streamlit widgets */
button[kind="primary"], [data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #7c5cff, #4f8cff) !important; border: 0 !important; border-radius: 12px !important;
  font-weight: 600 !important; padding: .55rem 1.4rem !important; transition: transform .15s, box-shadow .15s !important; }
button[kind="primary"]:hover, [data-testid="stBaseButton-primary"]:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(124,92,255,.45); }
[data-testid="stFileUploaderDropzone"] { border: 2px dashed rgba(124,92,255,.5); border-radius: 16px; background: rgba(124,92,255,.05); }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(124,92,255,.1); }
.foot { text-align: center; color: #6f7ba3; font-size: .8rem; margin: 30px 0 10px; }

@media (max-width: 760px) {
  .steps, .tiles { grid-template-columns: 1fr 1fr; }
  .title { font-size: 2.2rem; }
}
"""

# ------------------------------------------------------------------ helpers


def e(x):
    return html.escape("" if x is None else str(x))


def get_secret(name):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name, "")


def parse_json(text):
    """LLMs sometimes wrap JSON in code fences or add a sentence around it."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(1))
    raise ValueError("no JSON found in the model reply")


class ApiError(Exception):
    def __init__(self, source, status, detail=""):
        super().__init__(f"{source} {status}: {detail[:200]}")
        self.source, self.status, self.detail = source, status, detail


def explain_error(exc):
    if isinstance(exc, ApiError):
        who = "the AI service" if exc.source == "llm" else "the web search service"
        if exc.status in (401, 403):
            return f"The app's API key for {who} was rejected. The app owner needs to update it in the app secrets."
        if exc.status in (429, 432, 433):
            return f"Rate or quota limit reached on {who}. Wait a minute and try again, or lower 'Max claims'."
        if exc.status == 404:
            return "The AI service didn't recognise this model name. Check the Model field in the sidebar."
        return f"{who.capitalize()} returned an error ({exc.status}). {exc.detail[:120]}"
    if isinstance(exc, requests.exceptions.RequestException):
        return "Couldn't reach the service (network problem or timeout). Try again."
    return f"{type(exc).__name__}: {str(exc)[:200]}"


def call_llm(keys, model, prompt, system=None, json_mode=False, retries=5):
    """Plain HTTP call to any OpenAI-compatible chat API (Groq by default)."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    body = {"model": model, "messages": messages, "temperature": 0.1}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {keys['llm']}"}
    for attempt in range(retries):
        r = requests.post(f"{keys['base']}/chat/completions", headers=headers, json=body, timeout=60)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"] or ""
        if r.status_code == 400 and "response_format" in body and "response_format" in r.text:
            body.pop("response_format")  # this provider has no JSON mode, ask without it
            continue
        if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            try:
                wait = float(r.headers.get("retry-after", ""))
            except ValueError:
                wait = 4 * 2 ** attempt
            time.sleep(min(max(wait, 2), 30))
            continue
        raise ApiError("llm", r.status_code, r.text)
    raise ApiError("llm", 429, "retries used up")


def search_web(keys, query):
    headers = {"Authorization": f"Bearer {keys['search']}"}
    payload = {"query": query, "search_depth": "advanced", "max_results": 6}
    for attempt in range(3):
        r = requests.post(TAVILY_URL, headers=headers, json=payload, timeout=45)
        if r.status_code == 200:
            return r.json().get("results", [])
        if r.status_code == 429 and attempt < 2:
            time.sleep(3 * (attempt + 1))
            continue
        raise ApiError("search", r.status_code, r.text)


def sources_from(results):
    """Source links are the pages the search actually returned."""
    return [
        {"title": (r.get("title") or r["url"])[:60], "url": r["url"]}
        for r in results
        if r.get("url")
    ][:4]


def read_pdf(file):
    parts = []
    with pdfplumber.open(file) as pdf:
        for n, page in enumerate(pdf.pages, 1):
            t = page.extract_text() or ""
            if t.strip():
                parts.append(f"[Page {n}]\n{t}")
    return "\n\n".join(parts)


# ----------------------------------------------------------------- pipeline


def extract_claims(keys, model, text, max_claims):
    system = "You find checkable factual claims in documents. Reply with a JSON object only."
    prompt = f"""List up to {max_claims} specific claims from the document below that could be checked against
public sources: statistics, percentages, dates, financial figures, rankings, counts, technical specs.
Skip opinions and vague marketing language.

Return a JSON object with one key, "claims", whose value is an array. Each item has:
  "claim": the claim as written in the document, one sentence, numbers kept
  "category": one of statistic, date, financial, technical, other
  "page": the page number from the [Page N] markers, or null
  "search_query": a short neutral query that would find the true figure (name the entity, the metric and the year; do not repeat the claimed number)

DOCUMENT:
{text[:MAX_CHARS]}"""
    data = parse_json(call_llm(keys, model, prompt, system=system, json_mode=True))
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), [])
    clean, seen = [], set()
    for c in data:
        if not isinstance(c, dict):
            continue
        claim = str(c.get("claim", "")).strip()
        if not claim or claim.lower() in seen:
            continue
        seen.add(claim.lower())
        page = c.get("page")
        clean.append(
            {
                "claim": claim,
                "category": str(c.get("category") or "other").lower(),
                "page": page if isinstance(page, int) else None,
                "search_query": str(c.get("search_query") or claim),
            }
        )
    return clean[:max_claims]


def unchecked(claim, reason):
    return {
        **claim,
        "status": "Unchecked",
        "correct_fact": "",
        "explanation": reason,
        "suggested_fix": "",
        "confidence": "",
        "sources": [],
    }


def check_one(keys, model, claim, today, strictness):
    try:
        results = search_web(keys, claim["search_query"])
    except Exception as exc:
        return unchecked(claim, explain_error(exc))

    evidence = (
        "\n\n".join(
            f"[{i + 1}] {r.get('title', '')}\nURL: {r.get('url', '')}\n"
            f"Published: {r.get('published_date') or 'unknown'}\n{(r.get('content') or '')[:800]}"
            for i, r in enumerate(results[:5])
        )
        or "(no results found)"
    )
    system = "You are a careful fact-checker. Judge claims from the search results you are given. Reply with a JSON object only."
    prompt = f"""Today's date is {today}.

Claim from a document: "{claim['claim']}"

Web search results:
{evidence}

Decide whether the claim is right. {STRICTNESS[strictness]}
If the results are thin you may use well-established general knowledge, but then set confidence to "low" and say so.

Labels:
- "Verified": the sources confirm the claim.
- "Inaccurate": the claim is outdated, close but wrong, or partly wrong, and you can state the correct current figure.
- "False": no credible source supports it, or it looks made up.

Reply with ONLY a JSON object with these keys:
"status" (Verified, Inaccurate or False),
"correct_fact" (the real fact with its number and year; if nothing was found say so),
"explanation" (one or two sentences: what differs, and the year of the data),
"suggested_fix" (the original sentence rewritten so it is correct; empty if Verified),
"confidence" (high, medium or low)."""

    for _ in range(2):  # one retry if the reply isn't valid JSON
        try:
            data = parse_json(call_llm(keys, model, prompt, system=system, json_mode=True))
            if isinstance(data, list) and data:
                data = data[0]
            if not isinstance(data, dict):
                raise ValueError("reply was not an object")
            status = {"verified": "Verified", "inaccurate": "Inaccurate", "false": "False"}.get(
                str(data.get("status", "")).strip().lower(), "Unchecked"
            )
            return {
                **claim,
                "status": status,
                "correct_fact": str(data.get("correct_fact", "")),
                "explanation": str(data.get("explanation", "")),
                "suggested_fix": str(data.get("suggested_fix", "")),
                "confidence": str(data.get("confidence", "")).lower(),
                "sources": sources_from(results),
            }
        except ValueError:
            continue
        except Exception as exc:
            return unchecked(claim, explain_error(exc))
    return unchecked(claim, "The AI reply wasn't readable. Run the check again.")


# ------------------------------------------------------------------ html bits


def hero_html():
    return (
        '<div class="hero"><div class="orb o1"></div><div class="orb o2"></div>'
        '<div class="eyebrow">TRUTH LAYER FOR DOCUMENTS</div>'
        '<h1 class="title">Fact-Check <span>Agent</span></h1>'
        '<p class="sub">Drop in a PDF. Every statistic, date and figure is checked against the live web, '
        "and anything outdated or made up comes back with the correct fact.</p>"
        '<div class="chips"><span>Live web search</span><span>Claim-by-claim verdicts</span>'
        "<span>Suggested rewrites</span><span>Source links</span></div></div>"
    )


def steps_html():
    items = [
        ("1", "Extract", "Reads your document and lists every claim that can be checked."),
        ("2", "Verify", "Searches the web for each claim and compares the numbers."),
        ("3", "Report", "Marks it Verified, Inaccurate or False and shows the real fact."),
    ]
    out = "".join(
        f'<div class="step" style="animation-delay:{i * 0.1:.1f}s"><span class="n">{n}</span><b>{t}</b><p>{d}</p></div>'
        for i, (n, t, d) in enumerate(items)
    )
    return f'<div class="steps">{out}</div>'


def pbar_html(done, total, label):
    pct = int(100 * done / total) if total else 0
    return (
        f'<div class="plabel"><span>{e(label)}</span><span>{done}/{total}</span></div>'
        f'<div class="pbar"><div style="width:{max(pct, 4)}%"></div></div>'
    )


def ticker_html(rows):
    out = ""
    for r in rows:
        s = STATUS[r["status"]]
        claim = r["claim"] if len(r["claim"]) < 90 else r["claim"][:87] + "..."
        out += f'<div class="tick"><span class="dot {s["cls"]}"></span><b>{r["status"]}</b>&nbsp;{e(claim)}</div>'
    return out


def donut_html(counts):
    C = 2 * math.pi * 54
    kinds = ("Verified", "Inaccurate", "False")
    checked = sum(counts.get(k, 0) for k in kinds)
    segs, offset = "", 0.0
    for k in kinds:
        n = counts.get(k, 0)
        if not n:
            continue
        length = C * n / checked
        segs += (
            f'<circle class="seg" cx="70" cy="70" r="54" fill="none" stroke="{STATUS[k]["color"]}" '
            f'stroke-width="14" stroke-dasharray="{length:.2f} {C - length:.2f}" '
            f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 70 70)"/>'
        )
        offset += length
    score = round(100 * counts.get("Verified", 0) / checked) if checked else 0
    if not checked:
        verdict, color = "Nothing could be checked", "#94a3b8"
    elif score >= 80:
        verdict, color = "Mostly reliable", "#22c55e"
    elif score >= 50:
        verdict, color = "Needs a review", "#f59e0b"
    else:
        verdict, color = "Unreliable", "#ef4444"
    svg = (
        '<svg viewBox="0 0 140 140"><circle cx="70" cy="70" r="54" fill="none" '
        'stroke="rgba(255,255,255,.08)" stroke-width="14"/>'
        f'{segs}<text x="70" y="72" text-anchor="middle" class="score">{score}%</text>'
        '<text x="70" y="88" text-anchor="middle" class="cap">TRUST SCORE</text></svg>'
    )
    return f'<div class="donut">{svg}</div><div class="verdict" style="color:{color}">{verdict}</div>'


def tiles_html(counts, total):
    data = [
        ("all", total, "Claims found"),
        ("ok", counts.get("Verified", 0), "Verified"),
        ("warn", counts.get("Inaccurate", 0), "Inaccurate"),
        ("bad", counts.get("False", 0), "False"),
    ]
    out = "".join(
        f'<div class="tile {c}" style="animation-delay:{i * 0.08:.2f}s"><div class="n">{n}</div><div class="t">{t}</div></div>'
        for i, (c, n, t) in enumerate(data)
    )
    return f'<div class="tiles">{out}</div>'


def card_html(r, i):
    s = STATUS[r["status"]]
    meta = [r.get("category", "other").title()]
    if r.get("page"):
        meta.append(f"Page {r['page']}")
    if r.get("confidence"):
        meta.append(f"{r['confidence']} confidence")
    p = [f'<div class="fc-card {s["cls"]}" style="animation-delay:{min(i, 12) * 0.06:.2f}s">']
    p.append(
        f'<div class="fc-top"><span class="fc-badge">{s["icon"]} {r["status"]}</span>'
        f'<span class="fc-meta">{e(" · ".join(meta))}</span></div>'
    )
    p.append(f'<div class="fc-claim">“{e(r["claim"])}”</div>')
    if r.get("correct_fact"):
        p.append(f'<div class="fc-row"><span class="lbl">What the web says</span>{e(r["correct_fact"])}</div>')
    if r.get("explanation"):
        p.append(f'<div class="fc-row dim"><span class="lbl">Why</span>{e(r["explanation"])}</div>')
    if r.get("suggested_fix") and r["status"] in ("Inaccurate", "False"):
        p.append(f'<div class="fc-fix"><span class="lbl">Suggested rewrite</span>{e(r["suggested_fix"])}</div>')
    links = "".join(
        f'<a href="{e(x["url"])}" target="_blank" rel="noopener">{e(x["title"])}</a>'
        for x in r.get("sources", [])
        if str(x.get("url", "")).startswith("http")
    )
    if links:
        p.append(f'<div class="fc-src">{links}</div>')
    p.append("</div>")
    return "".join(p)


def apply_filters(rows, statuses, cats, query, sort):
    q = query.lower().strip()
    out = [
        r
        for r in rows
        if (not statuses or r["status"] in statuses)
        and (not cats or r["category"] in cats)
        and (not q or q in (r["claim"] + " " + r.get("correct_fact", "")).lower())
    ]
    if sort == "Most serious first":
        order = {"False": 0, "Inaccurate": 1, "Unchecked": 2, "Verified": 3}
        out = sorted(out, key=lambda r: order[r["status"]])
    return out


def to_dataframe(rows):
    return pd.DataFrame(
        [
            {
                "Status": r["status"],
                "Claim": r["claim"],
                "Category": r["category"],
                "Page": r.get("page"),
                "What the web says": r.get("correct_fact", ""),
                "Why": r.get("explanation", ""),
                "Suggested rewrite": r.get("suggested_fix", ""),
                "Sources": " | ".join(x["url"] for x in r.get("sources", [])),
            }
            for r in rows
        ]
    )


def to_markdown(rows, name, score):
    lines = [f"# Fact-check report: {name}", "", f"Trust score: {score}%", ""]
    for i, r in enumerate(rows, 1):
        lines.append(f"## {i}. {r['status']}: {r['claim']}")
        if r.get("correct_fact"):
            lines.append(f"- What the web says: {r['correct_fact']}")
        if r.get("explanation"):
            lines.append(f"- Why: {r['explanation']}")
        if r.get("suggested_fix") and r["status"] in ("Inaccurate", "False"):
            lines.append(f"- Suggested rewrite: {r['suggested_fix']}")
        for x in r.get("sources", []):
            lines.append(f"- Source: {x['url']}")
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------- UI

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(hero_html(), unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Settings")
    keys = {
        "llm": get_secret("LLM_API_KEY") or get_secret("GROQ_API_KEY"),
        "base": (get_secret("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        "search": get_secret("TAVILY_API_KEY"),
    }
    if keys["llm"] and keys["search"]:
        st.success("Search and AI engine ready")
    else:
        if not keys["llm"]:
            keys["llm"] = st.text_input("Groq API key", type="password", help="Free, no card needed")
            st.markdown("[Get a free Groq key](https://console.groq.com/keys)")
        if not keys["search"]:
            keys["search"] = st.text_input("Tavily API key", type="password", help="Free, no card needed")
            st.markdown("[Get a free Tavily key](https://app.tavily.com)")
    model = st.text_input("Model", get_secret("LLM_MODEL") or DEFAULT_MODEL)
    max_claims = st.slider("Max claims to check", 5, 30, 15)
    strictness = st.select_slider("Strictness", list(STRICTNESS), value="Standard")
    st.caption(STRICTNESS[strictness])

tab_pdf, tab_text, tab_demo = st.tabs(["Upload PDF", "Paste text", "Try the sample"])
with tab_pdf:
    up = st.file_uploader("Drop a PDF here", type=["pdf"], label_visibility="collapsed")
    go_pdf = st.button("Run fact-check", type="primary", key="go_pdf", disabled=up is None)
with tab_text:
    pasted = st.text_area("Paste any text", height=180, placeholder="Paste a report, press release, blog post...")
    go_text = st.button("Run fact-check", type="primary", key="go_text")
with tab_demo:
    st.write("A short made-up brief with true, outdated and invented claims mixed together. Handy for a quick demo.")
    with st.expander("See the sample text"):
        st.code(SAMPLE_TEXT, language=None)
    go_demo = st.button("Check the sample", type="primary", key="go_demo")

text = name = None
if go_pdf and up:
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"That file is over {MAX_UPLOAD_MB} MB. Please upload a smaller PDF.")
        st.stop()
    try:
        text, name = read_pdf(up), up.name
    except Exception as exc:
        st.error(f"Couldn't read that PDF: {exc}")
        st.stop()
elif go_text and pasted.strip():
    text, name = pasted, "Pasted text"
elif go_demo:
    text, name = SAMPLE_TEXT, "Sample trap document"

if text is not None:
    if not (keys["llm"] and keys["search"]):
        st.error("Add your Groq and Tavily API keys in the sidebar first.")
        st.stop()
    if len(text.strip()) < 40:
        st.error("Not enough text found. Scanned PDFs have no text layer, try a text-based PDF.")
        st.stop()

    today = dt.date.today().strftime("%B %d, %Y")

    with st.status("Reading the document and listing claims...", expanded=False) as status:
        try:
            claims = extract_claims(keys, model, text, max_claims)
        except Exception as exc:
            status.update(label="Couldn't extract claims", state="error")
            st.error(explain_error(exc))
            st.stop()
        status.update(label=f"Found {len(claims)} claims to check", state="complete")

    if not claims:
        st.warning("No checkable claims found in this document.")
        st.stop()

    bar, feed = st.empty(), st.empty()
    bar.markdown(pbar_html(0, len(claims), "Searching the web..."), unsafe_allow_html=True)
    rows, recent, done = [None] * len(claims), [], 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(check_one, keys, model, c, today, strictness): i for i, c in enumerate(claims)}
        for fut in cf.as_completed(futures):
            rows[futures[fut]] = fut.result()
            recent.append(rows[futures[fut]])
            done += 1
            bar.markdown(pbar_html(done, len(claims), "Searching the web..."), unsafe_allow_html=True)
            feed.markdown(ticker_html(recent[-3:]), unsafe_allow_html=True)
    bar.empty()
    feed.empty()
    st.session_state["result"] = {
        "rows": rows,
        "name": name,
        "model": model,
        "when": dt.datetime.now().strftime("%d %b %Y, %H:%M"),
    }

result = st.session_state.get("result")
if not result:
    st.markdown(steps_html(), unsafe_allow_html=True)
else:
    rows = result["rows"]
    counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()

    st.markdown(f"#### Results for {result['name']}")
    st.caption(f"Checked {result['when']} with {result['model']}")

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(donut_html(counts), unsafe_allow_html=True)
    with right:
        st.markdown(tiles_html(counts, len(rows)), unsafe_allow_html=True)
        if counts.get("Unchecked"):
            st.caption(f"{counts['Unchecked']} claim(s) couldn't be checked and are left out of the score.")

    f1, f2, f3, f4 = st.columns([2, 2, 3, 2])
    pick_status = f1.multiselect("Status (all if empty)", [k for k in STATUS if counts.get(k)])
    pick_cat = f2.multiselect("Category (all if empty)", sorted({r["category"] for r in rows}))
    query = f3.text_input("Search claims", placeholder="e.g. users, 2023, revenue")
    sort = f4.selectbox("Sort", ["Document order", "Most serious first"])

    shown = apply_filters(rows, pick_status, pick_cat, query, sort)
    if shown:
        st.markdown("".join(card_html(r, i) for i, r in enumerate(shown)), unsafe_allow_html=True)
    else:
        st.info("No claims match these filters.")

    checked = sum(counts.get(k, 0) for k in ("Verified", "Inaccurate", "False"))
    score = round(100 * counts.get("Verified", 0) / checked) if checked else 0
    d1, d2, d3, _ = st.columns([1, 1, 1, 2])
    d1.download_button("Download CSV", to_dataframe(rows).to_csv(index=False).encode(), "fact_check.csv", "text/csv")
    d2.download_button("Download JSON", json.dumps(rows, indent=2), "fact_check.json", "application/json")
    d3.download_button("Download report", to_markdown(rows, result["name"], score), "fact_check_report.md")

st.markdown(
    '<div class="foot">Verdicts come from web search results. Open the source links before relying on anything important.</div>',
    unsafe_allow_html=True,
)