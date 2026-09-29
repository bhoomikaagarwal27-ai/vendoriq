"""
ai_engine.py - Everything that talks to the Gemini API, plus the checks on what comes back.
Kept free of Streamlit code so it can be unit-tested.

LLM roles:  EXPLORER (propose plan)  ->  JUDGE (review + correct plan)  ->  ANALYST (explain) + CHAT
Failure handling:
  * no key / invalid key     -> rule-based mode (heuristic plan, rule-based insights), clearly labelled
  * model retired (404)      -> next model in the list
  * quota hit (429)          -> next model (separate quota), then rule-based fallback
  * server error / timeout   -> one retry, then next model
  * empty / blocked / invalid JSON -> treated as a failure, fallback used
  * plausible but wrong output -> caught by planner.validate_plan() and validate_insights()
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ValidationError

from prompts import (ANALYST_SYSTEM, ANALYST_USER, CANNED_INJECTION_REPLY, CHAT_SYSTEM, EXPLORER_SYSTEM,
                     EXPLORER_USER, JUDGE_SYSTEM, JUDGE_USER)
from safety import looks_like_injection
from scoring import ccol

DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-flash-lite-latest", "gemini-flash-latest"]
MAX_CHAT_CHARS = 600
MAX_HISTORY_TURNS = 8


# ---------------------------------------------------------------- schemas
class Criterion(BaseModel):
    column: str
    label: str
    direction: str
    weight: float
    role: str
    reason: str


class Ignored(BaseModel):
    column: str
    reason: str


class FilterSuggestion(BaseModel):
    column: str
    operator: str
    value: str
    reason: str


class Derived(BaseModel):
    name: str
    formula: str
    direction: str
    reason: str


class DataPlan(BaseModel):
    dataset_summary: str
    entity_column: str
    label_column: str
    group_column: str
    criteria: list[Criterion]
    ignored_columns: list[Ignored]
    filter_suggestions: list[FilterSuggestion]
    derived_metrics: list[Derived]
    data_quality_notes: list[str]
    analysis_questions: list[str]


class Issue(BaseModel):
    severity: str
    item: str
    problem: str
    fix: str


class Verdict(BaseModel):
    verdict: str
    issues: list[Issue]
    corrected_plan: DataPlan
    confidence: str


class OptionRisk(BaseModel):
    id: str
    risk: str


class Insights(BaseModel):
    recommended_id: str
    headline: str
    why: list[str]
    risks: list[OptionRisk]
    trade_offs: list[str]
    anomalies: list[str]
    negotiation_levers: list[str]
    further_analysis: list[str]
    confidence: str
    data_gaps: list[str]
    disagreement_note: str


@dataclass
class AIResult:
    ok: bool
    text: str = ""
    parsed: dict | None = None
    model: str = ""
    latency_ms: int = 0
    source: str = "gemini"          # gemini | cache | rule-based | blocked
    error: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    attempts: list = field(default_factory=list)


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def cache_key(*parts) -> str:
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- API call
def call_gemini(api_key: str, system: str, contents, models: list[str], json_schema=None,
                temperature: float = 0.2, max_tokens: int = 2048, timeout_ms: int = 60000) -> AIResult:
    """max_tokens includes the model's internal "thinking" tokens on Gemini 3.x, so keep it generous."""
    if not api_key:
        return AIResult(ok=False, source="rule-based", error="No Gemini API key configured.")
    try:
        from google import genai
        from google.genai import errors, types
    except ImportError:
        return AIResult(ok=False, source="rule-based", error="google-genai package not installed.")

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=timeout_ms))
    cfg = dict(system_instruction=system, temperature=temperature, max_output_tokens=max_tokens,
               automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    if json_schema is not None:
        cfg.update(response_mime_type="application/json", response_schema=json_schema)
    config = types.GenerateContentConfig(**cfg)

    attempts, last_error = [], "Unknown error"
    for model in models:
        for attempt in range(2):
            t0 = time.time()
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                ms = int((time.time() - t0) * 1000)
                text = (resp.text or "").strip()
                if not text:
                    reason = ""
                    try:
                        reason = str(resp.candidates[0].finish_reason)
                    except Exception:  # noqa: BLE001
                        pass
                    last_error = f"Empty or blocked response ({reason or 'no text'})."
                    attempts.append((model, "empty"))
                    break
                usage = getattr(resp, "usage_metadata", None)
                attempts.append((model, "ok"))
                return AIResult(ok=True, text=text, model=model, latency_ms=ms, attempts=attempts,
                                tokens_in=getattr(usage, "prompt_token_count", 0) or 0,
                                tokens_out=getattr(usage, "candidates_token_count", 0) or 0)
            except errors.ClientError as e:
                code = getattr(e, "code", 0)
                attempts.append((model, f"client {code}"))
                msg = str(getattr(e, "message", "") or e)
                if code == 400 and "api key" in msg.lower():
                    return AIResult(ok=False, source="rule-based", error="Gemini API key is invalid.", attempts=attempts)
                if code in (401, 403):
                    return AIResult(ok=False, source="rule-based", error="API key not allowed to use this model/project.", attempts=attempts)
                last_error = ("Free-tier quota reached (HTTP 429)." if code == 429 else
                              f"Model '{model}' not available (HTTP 404)." if code == 404 else
                              f"Request rejected (HTTP {code}): {msg[:120]}")
                break
            except errors.ServerError as e:
                attempts.append((model, f"server {getattr(e, 'code', '')}"))
                last_error = f"Gemini server error (HTTP {getattr(e, 'code', '5xx')})."
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except Exception as e:  # noqa: BLE001 - timeouts, network, DNS...
                attempts.append((model, type(e).__name__))
                last_error = f"Network/timeout problem: {type(e).__name__}."
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break
    return AIResult(ok=False, source="rule-based", error=last_error, attempts=attempts)


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?", "", t).rstrip("`").strip()
    return t


def _parse(res: AIResult, model_cls) -> AIResult:
    if not res.ok:
        return res
    try:
        res.parsed = model_cls.model_validate_json(_strip_fences(res.text)).model_dump()
    except (ValidationError, ValueError) as e:
        res.ok, res.source = False, "rule-based"
        res.error = f"AI returned output that does not match the expected format ({type(e).__name__})."
    return res


# ---------------------------------------------------------------- 1+2: explore, then judge
def explore(api_key: str, profile_llm: dict, models: list[str]) -> AIResult:
    user = EXPLORER_USER.format(profile_json=to_json(profile_llm))
    return _parse(call_gemini(api_key, EXPLORER_SYSTEM, user, models, json_schema=DataPlan,
                              temperature=0.2, max_tokens=8192), DataPlan)


def judge(api_key: str, profile_llm: dict, plan: dict, models: list[str]) -> AIResult:
    user = JUDGE_USER.format(profile_json=to_json(profile_llm), plan_json=to_json(plan))
    return _parse(call_gemini(api_key, JUDGE_SYSTEM, user, models, json_schema=Verdict,
                              temperature=0.1, max_tokens=8192), Verdict)


def explore_and_judge(api_key: str, profile_llm: dict, models: list[str]) -> dict:
    """Returns {'plan', 'explorer', 'judge', 'note'}; plan is None if the explorer failed."""
    ex = explore(api_key, profile_llm, models)
    if not ex.ok:
        return {"plan": None, "explorer": ex, "judge": None, "note": ex.error}
    plan = dict(ex.parsed, source="gemini explorer")
    jd = judge(api_key, profile_llm, ex.parsed, models)
    if not jd.ok:
        return {"plan": plan, "explorer": ex, "judge": jd,
                "note": f"Judge unavailable ({jd.error}) - explorer plan used without review."}
    v = jd.parsed
    corrected = v.get("corrected_plan") or {}
    if corrected.get("criteria"):
        plan = dict(corrected, source=f"gemini explorer + judge ({v.get('verdict')})")
    return {"plan": plan, "explorer": ex, "judge": jd, "note": ""}


# ---------------------------------------------------------------- IDs in model output
def id_regex(ids) -> re.Pattern | None:
    """Build a pattern that matches IDs of the same SHAPE as the real ones (to catch invented IDs)."""
    ids = [str(i) for i in ids if str(i)]
    if not ids:
        return None
    m = [re.fullmatch(r"([A-Za-z]{1,8})([-_ ]?)(\d{1,9})", i) for i in ids]
    if all(m):
        prefixes = sorted({x.group(1) for x in m}, key=len, reverse=True)
        return re.compile(r"\b(?:" + "|".join(map(re.escape, prefixes)) + r")[-_ ]?\d{1,9}\b")
    return None


def mentioned_ids(text: str, known: set, pattern: re.Pattern | None) -> tuple[set, set]:
    """Returns (known ids mentioned, unknown id-shaped tokens)."""
    if pattern is not None:
        found = set(pattern.findall(text))
        return found & known, found - known
    return {k for k in list(known)[:500] if k and k in text}, set()


def _numbers(text: str, pattern: re.Pattern | None) -> list[float]:
    if pattern is not None:
        text = pattern.sub(" ", text)
    text = re.sub(r"ISO\s?\d+", " ", text)
    vals = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            vals.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return vals


# ---------------------------------------------------------------- 3: analyst
def get_insights(api_key: str, context: dict, models: list[str]) -> AIResult:
    user = ANALYST_USER.format(data_json=to_json(context))
    return _parse(call_gemini(api_key, ANALYST_SYSTEM, user, models, json_schema=Insights,
                              temperature=0.2, max_tokens=6144), Insights)


def validate_insights(parsed: dict, context: dict, pool_ids: set, top_id: str) -> list[dict]:
    """Hallucination checks run on every AI answer before it is shown."""
    pattern = id_regex(pool_ids)
    checks = []
    rec = parsed.get("recommended_id", "")
    checks.append({"check": "Recommended option exists in the ranked pool", "passed": rec in pool_ids, "detail": rec or "(blank)"})
    checks.append({"check": "Agrees with transparent score model", "passed": rec == top_id,
                   "detail": "Same #1" if rec == top_id else f"AI says {rec}, model says {top_id} - model ranking is used"})
    parts = [parsed.get("headline", ""), parsed.get("disagreement_note", "")]
    for k in ("why", "trade_offs", "anomalies", "negotiation_levers", "further_analysis"):
        parts += parsed.get(k, [])
    parts += [r.get("risk", "") + " " + r.get("id", "") for r in parsed.get("risks", [])]
    text = " ".join(parts)
    _, unknown = mentioned_ids(text, set(pool_ids), pattern)
    bad_risk_ids = [r.get("id") for r in parsed.get("risks", []) if r.get("id") not in pool_ids]
    unknown |= set(x for x in bad_risk_ids if x)
    checks.append({"check": "No invented option IDs", "passed": not unknown,
                   "detail": "OK" if not unknown else "Unknown: " + ", ".join(sorted(unknown)[:5])})
    checks.append({"check": "Confidence label is valid", "passed": parsed.get("confidence") in {"High", "Medium", "Low"},
                   "detail": parsed.get("confidence", "")})
    echo = looks_like_injection(text)
    checks.append({"check": "No injected instructions echoed", "passed": not echo,
                   "detail": "OK" if not echo else "Output repeats instruction-like text"})
    source = _numbers(to_json(context), pattern)
    claimed = [n for n in _numbers(text, pattern) if not (float(n).is_integer() and 0 <= n <= 5)]
    grounded = sum(any(abs(n - s) <= max(0.05, 0.01 * abs(s)) or abs(abs(n) - abs(s)) <= 0.05 for s in source) for n in claimed)
    ratio = grounded / len(claimed) if claimed else None
    checks.append({"check": "Numbers traceable to data", "passed": ratio is None or ratio >= 0.8,
                   "detail": "No numbers quoted" if ratio is None else f"{grounded}/{len(claimed)} numbers found in the data"})
    return checks


# ---------------------------------------------------------------- chat
def screen_user_message(msg: str) -> tuple[bool, str]:
    if not msg or not msg.strip():
        return False, "Please type a question."
    if len(msg) > MAX_CHAT_CHARS:
        return False, f"Please keep questions under {MAX_CHAT_CHARS} characters."
    if looks_like_injection(msg):
        return False, CANNED_INJECTION_REPLY
    return True, ""


def chat_reply(api_key: str, context: dict, history: list[dict], user_msg: str, models: list[str]) -> AIResult:
    top = context["top_options"][0]["id"] if context.get("top_options") else "the top option"
    system = CHAT_SYSTEM.replace("{top_id}", str(top)).replace("{data_json}", to_json(context))
    turns = history[-MAX_HISTORY_TURNS:] + [{"role": "user", "content": user_msg}]
    contents = [{"role": "user" if t["role"] == "user" else "model", "parts": [{"text": t["content"]}]} for t in turns]
    return call_gemini(api_key, system, contents, models, temperature=0.2, max_tokens=2048)


_ROLE_KEYWORDS = [
    (("cheap", "price", "cost", "l1", "lowest rate", "expensive"), "cost"),
    (("fast", "quick", "lead", "urgent", "soon", "delivery", "on time", "on-time", "otif"), "delivery"),
    (("quality", "defect", "reject", "rating", "audit"), "quality"),
    (("risk", "safe", "complaint"), "risk"),
    (("credit", "payment", "working capital"), "financial"),
    (("esg", "sustain", "green", "certif", "iso"), "sustainability"),
    (("capacity", "volume", "moq", "scale"), "capacity"),
]


def offline_answer(question: str, scored: pd.DataFrame, criteria: list[dict]) -> str:
    """Keyword fallback when the AI is unavailable. Deterministic, grounded, works for any criteria."""
    if scored.empty:
        return "No option qualifies with the current filters, so there is nothing to compare."
    q = question.lower()
    target = next((c for c in criteria if c["label"].lower() in q or c["column"].replace("_", " ") in q), None)
    if target is None:
        for words, role in _ROLE_KEYWORDS:
            if any(w in q for w in words):
                target = next((c for c in sorted(criteria, key=lambda c: -c["weight"]) if c.get("role") == role), None)
                if target:
                    break
    if target is not None:
        col = ccol(target)
        r = scored.loc[scored[col].idxmin() if target["direction"] == "lower" else scored[col].idxmax()]
        name = f" ({r['_label']})" if str(r["_label"]) != str(r["_id"]) else ""
        return (f"{r['_id']}{name} is best on {target['label']}: {r[col]:g} "
                f"({'lower' if target['direction'] == 'lower' else 'higher'} is better).  _(Offline rule-based answer - AI unavailable.)_")
    if any(k in q for k in ("best", "top", "recommend", "first", "rank", "why")):
        r = scored.iloc[0]
        return f"{r['_id']} is ranked #1 with a score of {r['score']:.1f}/100.  _(Offline rule-based answer - AI unavailable.)_"
    labels = ", ".join(c["label"] for c in criteria[:5])
    return f"AI assistant is offline, so I can only answer simple questions such as 'who is best on {criteria[0]['label']}?' or 'who is ranked first?'. Criteria: {labels}."


def label_ids(text: str, id_to_name: dict, pattern: re.Pattern | None) -> str:
    """Show names next to IDs on screen (names are not sent to the AI when anonymising)."""
    if not id_to_name:
        return text
    seen = set()

    def repl(m):
        vid = m.group(0)
        name = id_to_name.get(vid)
        if name and name != vid and vid not in seen and name not in text:
            seen.add(vid)
            return f"{vid} ({name})"
        return vid
    if pattern is not None:
        return pattern.sub(repl, text)
    return text
