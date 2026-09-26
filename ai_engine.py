"""
ai_engine.py - Everything that talks to the Gemini API, plus the checks that run
on what comes back. Kept free of Streamlit code so it can be unit-tested.

Failure handling (what happens if the API is down or returns garbage):
  * no key / invalid key     -> app switches to rule-based mode and says so
  * model retired (404)      -> next model in the fallback list is tried
  * quota hit (429)          -> next model tried (each model has its own quota),
                                then rule-based fallback
  * server error / timeout   -> one retry after a short pause, then next model
  * empty or blocked reply   -> treated as a failure, fallback used
  * invalid JSON / bad IDs   -> output rejected by validate_recommendation(),
                                fallback shown next to a warning
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

import pandas as pd
from pydantic import BaseModel, ValidationError

from prompts import (CANNED_INJECTION_REPLY, CHAT_SYSTEM, RECOMMENDER_SYSTEM,
                     RECOMMENDER_USER_TEMPLATE)
from scoring import CRITERIA
from validation import looks_like_injection

# Free-tier friendly order: Flash-Lite models have a much larger free daily quota
# than Flash models. Aliases at the end survive model retirements.
DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
]
MAX_CHAT_CHARS = 600
MAX_HISTORY_TURNS = 8
ID_PATTERN = re.compile(r"\b[A-Z]\d{3}\b")


# ---------------------------------------------------------------- schema
class VendorRisk(BaseModel):
    vendor_id: str
    risk: str


class Recommendation(BaseModel):
    recommended_vendor_id: str
    headline: str
    why: list[str]
    risks: list[VendorRisk]
    negotiation_levers: list[str]
    l1_comparison: str
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


# ---------------------------------------------------------------- context
def build_context(scored: pd.DataFrame, disqualified: pd.DataFrame, req: dict, weights_pct: dict,
                  l1: dict, stability: pd.DataFrame, anonymise: bool = True) -> dict:
    """The ONLY data the model sees. All arithmetic is done here, not by the model."""
    cols = ["vendor_id", "rank", "score", "tier", "origin", "price_inr_per_kg", "freight_rs_per_kg",
            "landed_cost", "purity_pct", "qc_pass_rate_pct", "lead_time_days", "otif_pct", "credit_days",
            "moq_mt", "complaints_12m", "risk_points", "esg_score", "certifications", "notes"]
    if not anonymise:
        cols[1:1] = ["vendor_name", "location"]
    qty_kg = req["qty_mt"] * 1000
    vendors = []
    top = scored.iloc[0] if not scored.empty else None
    for _, r in scored.iterrows():
        v = {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in cols}
        v["order_value_rs"] = round(float(r["landed_cost"]) * qty_kg)
        v["points_by_criterion"] = {k: float(r[f"pts_{k}"]) for k in CRITERIA}
        if top is not None and r["vendor_id"] != top["vendor_id"]:
            v["gap_vs_top"] = {
                "score": round(float(r["score"] - top["score"]), 1),
                "landed_cost_rs_per_kg": round(float(r["landed_cost"] - top["landed_cost"]), 2),
                "lead_time_days": int(r["lead_time_days"] - top["lead_time_days"]),
                "otif_pct": round(float(r["otif_pct"] - top["otif_pct"]), 1),
                "qc_pass_rate_pct": round(float(r["qc_pass_rate_pct"] - top["qc_pass_rate_pct"]), 1),
                "credit_days": int(r["credit_days"] - top["credit_days"]),
            }
        vendors.append(v)
    dq = [{"vendor_id": r["vendor_id"], "reason": r["disqualified_because"]} for _, r in disqualified.iterrows()]
    return {
        "requirement": req,
        "weights_pct": {k: round(v * 100, 1) for k, v in weights_pct.items()},
        "scoring_method": "min-max normalised per criterion (1 = best in this list) x weight; total out of 100. "
                          "Scores are RELATIVE to the vendors in this list.",
        "qualified_vendors": vendors,
        "disqualified_vendors": dq,
        "l1_check": {k: v for k, v in l1.items()},
        "stability": stability.head(5).to_dict(orient="records"),
    }


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def cache_key(*parts) -> str:
    return hashlib.sha256("||".join(map(str, parts)).encode()).hexdigest()[:24]


# ---------------------------------------------------------------- API call
def call_gemini(api_key: str, system: str, contents, models: list[str], json_schema=None,
                temperature: float = 0.2, max_tokens: int = 2048, timeout_ms: int = 45000) -> AIResult:
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
        for attempt in range(2):                      # 1 retry on server/timeout errors
            t0 = time.time()
            try:
                resp = client.models.generate_content(model=model, contents=contents, config=config)
                ms = int((time.time() - t0) * 1000)
                text = (resp.text or "").strip()
                if not text:
                    reason = ""
                    try:
                        reason = str(resp.candidates[0].finish_reason)
                    except Exception:
                        pass
                    last_error = f"Empty or blocked response ({reason or 'no text'})."
                    attempts.append((model, "empty"))
                    break                              # try next model
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
                if code == 429:
                    last_error = "Free-tier quota reached (HTTP 429)."
                elif code == 404:
                    last_error = f"Model '{model}' not available (HTTP 404)."
                else:
                    last_error = f"Request rejected (HTTP {code}): {msg[:120]}"
                break                                  # next model
            except errors.ServerError as e:
                attempts.append((model, f"server {getattr(e, 'code', '')}"))
                last_error = f"Gemini server error (HTTP {getattr(e, 'code', '5xx')})."
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break
            except Exception as e:                     # timeouts, network, DNS...
                attempts.append((model, type(e).__name__))
                last_error = f"Network/timeout problem: {type(e).__name__}."
                if attempt == 0:
                    time.sleep(1.5)
                    continue
                break
    return AIResult(ok=False, source="rule-based", error=last_error, attempts=attempts)


def get_recommendation(api_key: str, context: dict, models: list[str]) -> AIResult:
    user = RECOMMENDER_USER_TEMPLATE.format(data_json=to_json(context))
    res = call_gemini(api_key, RECOMMENDER_SYSTEM, user, models, json_schema=Recommendation, temperature=0.2, max_tokens=4096)
    if not res.ok:
        return res
    try:
        parsed = Recommendation.model_validate_json(_strip_fences(res.text)).model_dump()
        res.parsed = parsed
    except (ValidationError, ValueError) as e:
        res.ok = False
        res.source = "rule-based"
        res.error = f"AI returned output that does not match the expected format ({type(e).__name__})."
    return res


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(json)?", "", t).rstrip("`").strip()
    return t


# ---------------------------------------------------------------- output checks
def _numbers(text: str) -> list[float]:
    text = ID_PATTERN.sub(" ", text)                 # vendor IDs are not numbers
    text = re.sub(r"ISO\s?\d+", " ", text)           # nor are ISO standard numbers
    vals = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            vals.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return vals


def validate_recommendation(parsed: dict, context: dict, top_id: str) -> tuple[list[dict], float | None]:
    """Hallucination checks run on every AI answer before it is shown."""
    checks = []
    qualified_ids = {v["vendor_id"] for v in context["qualified_vendors"]}
    known_ids = qualified_ids | {v["vendor_id"] for v in context["disqualified_vendors"]}

    rec = parsed.get("recommended_vendor_id", "")
    checks.append({"check": "Recommended vendor exists in qualified list",
                   "passed": rec in qualified_ids, "detail": rec or "(blank)"})
    checks.append({"check": "Agrees with transparent score model",
                   "passed": rec == top_id,
                   "detail": "Same #1" if rec == top_id else f"AI says {rec}, model says {top_id} - model ranking is used"})

    all_text = " ".join([parsed.get("headline", ""), parsed.get("l1_comparison", ""), parsed.get("disagreement_note", "")]
                        + parsed.get("why", []) + parsed.get("negotiation_levers", [])
                        + [r.get("risk", "") + " " + r.get("vendor_id", "") for r in parsed.get("risks", [])])
    mentioned = set(ID_PATTERN.findall(all_text))
    unknown = sorted(mentioned - known_ids)
    checks.append({"check": "No invented vendor IDs", "passed": not unknown,
                   "detail": "OK" if not unknown else "Unknown IDs: " + ", ".join(unknown)})

    checks.append({"check": "Confidence label is valid", "passed": parsed.get("confidence") in {"High", "Medium", "Low"},
                   "detail": parsed.get("confidence", "")})

    echo = looks_like_injection(all_text)
    checks.append({"check": "No injected instructions echoed", "passed": not echo,
                   "detail": "OK" if not echo else "Output repeats instruction-like text"})

    # Numeric grounding: every number the AI quotes should exist in the data we sent.
    source_nums = _numbers(to_json(context))
    claimed = [n for n in _numbers(all_text) if not (float(n).is_integer() and 0 <= n <= 3)]
    grounded = 0
    for n in claimed:
        if any(abs(n - s) <= max(0.05, 0.01 * abs(s)) or abs(abs(n) - abs(s)) <= 0.05 for s in source_nums):
            grounded += 1
    ratio = grounded / len(claimed) if claimed else None
    checks.append({"check": "Numbers traceable to data",
                   "passed": ratio is None or ratio >= 0.8,
                   "detail": "No numbers quoted" if ratio is None else f"{grounded}/{len(claimed)} numbers found in the data"})
    return checks, ratio


# ---------------------------------------------------------------- chat
def screen_user_message(msg: str) -> tuple[bool, str]:
    """Checks that run BEFORE any API call. Returns (allowed, reply_if_blocked)."""
    if not msg or not msg.strip():
        return False, "Please type a question."
    if len(msg) > MAX_CHAT_CHARS:
        return False, f"Please keep questions under {MAX_CHAT_CHARS} characters."
    if looks_like_injection(msg):
        return False, CANNED_INJECTION_REPLY
    return True, ""


def chat_reply(api_key: str, context: dict, history: list[dict], user_msg: str, models: list[str]) -> AIResult:
    top_id = context["qualified_vendors"][0]["vendor_id"] if context["qualified_vendors"] else "V001"
    system = CHAT_SYSTEM.replace("{top_id}", top_id).replace("{data_json}", to_json(context))
    turns = history[-MAX_HISTORY_TURNS:] + [{"role": "user", "content": user_msg}]
    contents = [{"role": "user" if t["role"] == "user" else "model", "parts": [{"text": t["content"]}]} for t in turns]
    return call_gemini(api_key, system, contents, models, temperature=0.2, max_tokens=2048)


def offline_answer(question: str, scored: pd.DataFrame) -> str:
    """Keyword fallback when the AI is unavailable. Deterministic and grounded."""
    if scored.empty:
        return "No vendor qualifies with the current requirements, so there is nothing to compare."
    q = question.lower()
    pick = None
    rules = [
        (("cheap", "lowest price", "l1", "cost", "price"), "landed_cost", True, "lowest landed cost", "Rs/kg"),
        (("fast", "quick", "lead time", "urgent", "soon"), "lead_time_days", True, "shortest lead time", "days"),
        (("reliab", "on time", "otif", "deliver"), "otif_pct", False, "best on-time delivery", "%"),
        (("quality", "qc", "reject"), "qc_pass_rate_pct", False, "best QC pass rate", "%"),
        (("credit", "payment", "working capital"), "credit_days", False, "longest credit period", "days"),
        (("esg", "sustain", "green"), "esg_score", False, "best ESG score", "/100"),
    ]
    for keys, col, asc, label, unit in rules:
        if any(k in q for k in keys):
            r = scored.sort_values(col, ascending=asc).iloc[0]
            val = f"{r[col]:.2f}" if col == "landed_cost" else f"{r[col]:g}"
            pick = f"{r['vendor_id']} ({r['vendor_name']}) has the {label}: {val}{'' if unit in ('%', '/100') else ' '}{unit}."
            break
    if pick is None and any(k in q for k in ("best", "top", "recommend", "first", "rank")):
        r = scored.iloc[0]
        pick = f"{r['vendor_id']} ({r['vendor_name']}) is ranked #1 with a score of {r['score']:.1f}/100."
    if pick is None:
        return ("AI assistant is offline, so I can only answer simple questions. Try: "
                "'cheapest', 'fastest', 'most reliable', 'best quality', 'longest credit' or 'who is ranked first'.")
    return pick + "  _(Offline rule-based answer - AI assistant unavailable.)_"


def label_ids(text: str, id_to_name: dict) -> str:
    """Show vendor names next to IDs on screen (names are never sent to the AI when anonymised)."""
    seen = set()

    def repl(m):
        vid = m.group(0)
        if vid in id_to_name and vid not in seen and id_to_name[vid] not in text:
            seen.add(vid)
            return f"{vid} ({id_to_name[vid]})"
        return vid
    return ID_PATTERN.sub(repl, text)
