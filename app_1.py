"""
VendorIQ - AI-assisted vendor analytics for ANY vendor dataset.
End-term project: AI Applications (Use case #6 - Vendor selection / procurement recommender).

Flow:  load 1..N files  ->  profile  ->  LLM EXPLORES the profile and proposes a plan
       ->  a second LLM call JUDGES the plan  ->  Python validates  ->  user edits
       ->  clean  ->  rank  ->  further analysis  ->  LLM explains  ->  chat
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_engine as ai
import analysis
import ingest
import planner
import profiler
import samples
import scoring
from prompts import PROMPT_VERSION

APP_VERSION = "2.0"
MAX_AI_CALLS_PER_SESSION = 30
MIN_SECONDS_BETWEEN_CALLS = 2
TABLE_ROWS_SHOWN = 500
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]   # validated order
OTHER_GREY = "#b8b6ae"
TOP3 = PALETTE[:3]

st.set_page_config(page_title="VendorIQ - AI Vendor Analytics", page_icon="🏭", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
  div[data-testid="stMetricValue"] {font-size: 1.4rem;}
  .small-note {color: #6b6b66; font-size: 0.85rem;}
  .ai-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             background:#eef4fc; color:#1c5cab; border:1px solid #cde2fb;}
  .fb-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             background:#fff5e6; color:#8a5a00; border:1px solid #f6d79a;}
  .ok-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             background:#eaf6ea; color:#1f6b1f; border:1px solid #9fd39f;}
</style>
""", unsafe_allow_html=True)


# ============================================================ helpers & state
def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default) or os.environ.get(name, default)
    except Exception:  # noqa: BLE001 - no secrets.toml present
        return os.environ.get(name, default)


@st.cache_resource
def shared_ai_cache() -> dict:
    """Shared by all users: identical requests reuse the earlier answer (saves free-tier quota)."""
    return {}


# cache_resource returns the same object (no copy) - important for big tables. Nothing below mutates them.
@st.cache_resource(show_spinner=False, max_entries=6)
def synthetic_files(n: int) -> tuple:
    return tuple(samples.large_synthetic(n))


@st.cache_resource(show_spinner=False, max_entries=6)
def read_files(files: tuple) -> tuple[list, list]:
    tables, errors = [], []
    for name, data in files:
        try:
            tables.extend(ingest.read_file(name, data))
        except ValueError as e:
            errors.append(str(e))
    return tables, errors


@st.cache_resource(show_spinner=False, max_entries=6)
def build_dataset(key: str, _tables: list, strategy: str, join_key: str | None):
    comb = ingest.combine(_tables, strategy, join_key)
    prof = profiler.profile_table(comb.df, comb.originals)
    return comb, prof


@st.cache_resource(show_spinner=False, max_entries=12)
def prepare_cached(key: str, plan_json: str, exclude_extremes: bool, _df: pd.DataFrame, _profile: list):
    return planner.prepare(_df, _profile, json.loads(plan_json), exclude_extremes)


def init_state():
    ss = st.session_state
    if "initialised" in ss:
        return
    ss.initialised = True
    ss.plans = {}          # dataset key -> plan state
    ss.insights = {}       # context hash -> result
    ss.chat = []
    ss.audit = []
    ss.ai_calls = 0
    ss.last_call = 0.0


def log_event(action: str, res: ai.AIResult | None, note: str = ""):
    st.session_state.audit.append({
        "time": datetime.now().strftime("%H:%M:%S"), "action": action,
        "source": res.source if res else "-", "model": res.model if res else "-",
        "latency_ms": res.latency_ms if res else 0,
        "tokens_in": res.tokens_in if res else 0, "tokens_out": res.tokens_out if res else 0,
        "status": ("ok" if res and res.ok else (res.error if res else "")) + (f" {note}" if note else ""),
    })


def can_call_ai() -> tuple[bool, str]:
    ss = st.session_state
    if ss.ai_calls >= MAX_AI_CALLS_PER_SESSION:
        return False, f"Session limit of {MAX_AI_CALLS_PER_SESSION} AI calls reached (protects the free quota)."
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - ss.last_call)
    if wait > 0:
        time.sleep(wait)
    return True, ""


def count_call():
    st.session_state.ai_calls += 1
    st.session_state.last_call = time.time()


def fmt(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:,.4g}" if abs(v) < 1e6 else f"{v:,.0f}"


init_state()
ss = st.session_state

# ============================================================ sidebar
with st.sidebar:
    st.markdown("### 🏭 VendorIQ")
    st.caption(f"v{APP_VERSION} · prompts {PROMPT_VERSION}")
    user_key = st.text_input("Gemini API key (optional)", type="password",
                             help="Leave blank to use the app owner's key. A key typed here stays in this browser session only.")
    api_key = user_key.strip() or secret("GEMINI_API_KEY")
    preferred = secret("GEMINI_MODEL")
    models = ([preferred] if preferred else []) + [m for m in ai.DEFAULT_MODELS if m != preferred]
    if api_key:
        st.success("AI: Gemini key configured", icon="🟢")
        st.caption("Model order: " + " → ".join(models[:3]) + " …")
    else:
        st.warning("AI: offline - keyword plan & rule-based text", icon="🟠")
    st.markdown("**Privacy**")
    anonymise = st.toggle("Anonymise names sent to AI", value=True,
                          help="Names, IDs and free-text examples are masked in the profile, and option names are not sent.")
    st.caption("The AI sees a statistical PROFILE of your table and the top-ranked rows - never the whole file. "
               "Free-tier API data may be used by Google to improve its products; do not upload confidential data.")
    st.markdown("**Usage this session**")
    st.progress(min(ss.ai_calls / MAX_AI_CALLS_PER_SESSION, 1.0), text=f"{ss.ai_calls} / {MAX_AI_CALLS_PER_SESSION} AI calls")
    if st.button("↺ Reset session", width="stretch"):
        for k in list(ss.keys()):
            del ss[k]
        st.rerun()

# ============================================================ header
st.title("VendorIQ · AI vendor analytics for any dataset")
st.markdown("Upload **one or many** vendor files in any format. **Gemini explores** the data and proposes how to analyse it, "
            "a **second Gemini call judges** that plan, Python **cleans, ranks and analyses**, and you stay in control of every weight.")
st.caption("⚠️ Sample datasets are fictional. Output is decision support - the purchase committee makes the final decision.")

tab_data, tab_plan, tab_rank, tab_ai, tab_chat, tab_about = st.tabs(
    ["① Data", "② AI data plan", "③ Ranking", "④ AI insights", "⑤ Ask VendorIQ", "⑥ How it works"])

# ============================================================ ① DATA
files: tuple = ()
header_of: dict = {}
with tab_data:
    options = list(samples.SAMPLES) + ["Upload your own files"]
    src = st.radio("Choose data", options, horizontal=True, key="source")
    if src == "Upload your own files":
        ups = st.file_uploader("Upload one or more files - CSV, TSV, TXT, Excel (all sheets), JSON or Parquet",
                               type=["csv", "tsv", "txt", "xlsx", "xls", "xlsm", "json", "parquet"],
                               accept_multiple_files=True)
        files = tuple((u.name, u.getvalue()) for u in ups) if ups else ()
        if not files:
            st.info("Drop your files above. Any column names, any number of columns, messy values (₹, %, 'days', lakh, Yes/No) are fine. "
                    "Several files are stacked or joined automatically.")
    elif src == "Large synthetic vendor base":
        n = st.select_slider("Rows to generate", [1000, 5000, 20000, 50000, 100000, 200000], value=20000)
        files = synthetic_files(n)
    else:
        files = tuple(samples.SAMPLES[src]())
    if files and src != "Upload your own files":
        with st.expander("Download these sample files (to try the upload yourself)"):
            for name, data in files:
                st.download_button(f"⬇ {name}", data, name, "text/csv", key=f"dl_{name}")

    tables, read_errors = read_files(files) if files else ([], [])
    for e in read_errors:
        st.error(e)

    comb, profile, dataset_key = None, [], ""
    if tables:
        st.markdown("**Files read**")
        st.dataframe(pd.DataFrame([{"table": t.name, "rows": len(t.df), "columns": t.df.shape[1]} for t in tables]),
                     hide_index=True, width="stretch")
        strategy, join_key = "auto", None
        if len(tables) > 1:
            c1, c2 = st.columns([2, 2])
            strategy = c1.radio("How should the files be combined?", ["auto", "stack", "join"], horizontal=True,
                                format_func={"auto": "Auto-detect", "stack": "Stack (append rows)", "join": "Join (match on a key)"}.get)
            if strategy == "join":
                normed = [profiler.normalise_columns(t.df)[0] for t in tables]
                keys = ingest.find_join_keys(normed)
                if keys:
                    join_key = c2.selectbox("Join key", keys)
                else:
                    c2.warning("No column is shared and unique in every file - the files will be stacked.")
        fp = hashlib.sha1(b"".join(hashlib.sha1(d).digest() + n.encode() for n, d in files)).hexdigest()[:16]
        dataset_key = f"{fp}|{strategy}|{join_key}"
        with st.spinner("Combining and profiling the data…"):
            comb, profile = build_dataset(dataset_key, tables, strategy, join_key)
        for line in comb.log:
            st.caption("• " + line)
        header_of = {p["column"]: p["original_header"] for p in profile}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(comb.df):,}")
        m2.metric("Columns", comb.df.shape[1])
        m3.metric("Usable numeric columns", len(profiler.candidate_criteria(profile)))
        m4.metric("Files / tables", len(tables))
        st.markdown("**Column profile** - what the app (and the AI) learned about each column")
        prof_df = pd.DataFrame([{
            "column": p["original_header"], "kind": p["kind"], "missing %": p["missing_pct"], "distinct": p["unique"],
            "min": fmt(p["min"]) if p.get("min") is not None else "", "median": fmt(p["median"]) if p.get("median") is not None else "",
            "max": fmt(p["max"]) if p.get("max") is not None else "", "unit": p.get("unit_hint", ""),
            "examples": ", ".join(p.get("examples", []))[:80]} for p in profile])
        st.dataframe(prof_df, hide_index=True, width="stretch", height=min(420, 38 + 35 * len(prof_df)))
        with st.expander(f"Preview the combined data (first 200 of {len(comb.df):,} rows)"):
            st.dataframe(comb.df.head(200), width="stretch")

# ============================================================ ② PLAN
plan, prepared, plan_state = None, None, None
num_filters, text_filters = [], []
with tab_plan:
    if comb is None:
        st.info("Load data in tab ① first.")
    else:
        if dataset_key not in ss.plans:
            hp, hlog = planner.validate_plan(planner.heuristic_plan(profile), profile)
            ss.plans[dataset_key] = {"plan": hp, "source": "heuristic", "version": 0, "log": hlog}
        plan_state = ss.plans[dataset_key]

        st.markdown("**How the plan is made:** ① Gemini **explores** the column profile (not your raw rows) → "
                    "② a second Gemini call **judges** the plan → ③ Python **validates** it against the real data → "
                    "④ **you** edit anything below.")
        c1, c2 = st.columns([1, 3])
        run = c1.button("🔍 Explore & judge with AI", type="primary", width="stretch")
        c2.caption("Uses 2 AI calls. Without AI, a keyword-and-statistics plan is used, so the app always works.")
        if run:
            prof_llm = profiler.profile_for_llm(profile, len(comb.df), anonymise)
            ck = ai.cache_key("explore", PROMPT_VERSION, models, ai.to_json(prof_llm))
            cached = shared_ai_cache().get(ck)
            if cached:
                result = cached
                log_event("explore+judge", ai.AIResult(ok=True, source="cache", model=cached["model"]))
            elif not api_key:
                result = {"plan": None, "note": "No API key configured."}
            else:
                ok, why = can_call_ai()
                if not ok:
                    result = {"plan": None, "note": why}
                else:
                    with st.spinner("Gemini is exploring the data profile, then a second call is judging the plan…"):
                        out = ai.explore_and_judge(api_key, prof_llm, models)
                    count_call()
                    log_event("explore", out["explorer"])
                    if out["judge"] is not None:
                        count_call()
                        log_event("judge", out["judge"])
                    result = {"plan": out["plan"], "note": out["note"],
                              "explorer": out["explorer"].parsed if out["explorer"] and out["explorer"].ok else None,
                              "judge": out["judge"].parsed if out["judge"] and out["judge"].ok else None,
                              "model": out["explorer"].model, "latency": out["explorer"].latency_ms + (out["judge"].latency_ms if out["judge"] else 0),
                              "tokens": sum((r.tokens_in + r.tokens_out) for r in (out["explorer"], out["judge"]) if r)}
                    if result["plan"] is not None:
                        shared_ai_cache()[ck] = result
            if result.get("plan") is not None:
                vp, vlog = planner.validate_plan(result["plan"], profile)
                ss.plans[dataset_key] = {"plan": vp, "source": result["plan"].get("source", "gemini"), "version": plan_state["version"] + 1,
                                         "log": vlog, "explorer": result.get("explorer"), "judge": result.get("judge"),
                                         "note": result.get("note", ""), "model": result.get("model", ""),
                                         "latency": result.get("latency", 0), "tokens": result.get("tokens", 0)}
                plan_state = ss.plans[dataset_key]
            else:
                st.warning(f"AI plan not available ({result.get('note')}). The keyword plan stays in use.", icon="🟠")

        plan = plan_state["plan"]
        if plan_state["source"] == "heuristic":
            st.markdown('<span class="fb-badge">Keyword & statistics plan (no AI)</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="ai-badge">🤖 {plan_state["source"]} · {plan_state.get("model", "")} · '
                        f'{plan_state.get("latency", 0) / 1000:.1f}s · {plan_state.get("tokens", 0)} tokens</span>', unsafe_allow_html=True)
        st.markdown(f"**What this dataset is:** {plan.get('dataset_summary', '')}")
        if plan_state.get("note"):
            st.caption(plan_state["note"])

        if plan_state.get("judge"):
            j = plan_state["judge"]
            verdict = j.get("verdict", "")
            icon = {"approve": "✅", "approve_with_changes": "🛠️", "reject": "⛔"}.get(verdict, "ℹ️")
            with st.expander(f"{icon} Judge verdict: {verdict.replace('_', ' ')} · confidence {j.get('confidence', '')} · "
                             f"{len(j.get('issues', []))} issue(s)", expanded=bool(j.get("issues"))):
                if j.get("issues"):
                    st.dataframe(pd.DataFrame(j["issues"]), hide_index=True, width="stretch")
                else:
                    st.caption("The judge found no problems with the explorer's plan.")
                if plan_state.get("explorer"):
                    st.markdown("**Explorer's original criteria** (before the judge)")
                    st.dataframe(pd.DataFrame(plan_state["explorer"].get("criteria", [])), hide_index=True, width="stretch")
        if plan_state.get("log"):
            with st.expander(f"🔧 Python validator made {len(plan_state['log'])} correction(s)"):
                for line in plan_state["log"]:
                    st.markdown(f"- {line}")

        cols_all = [p["column"] for p in profile]
        header = {p["column"]: p["original_header"] for p in profile}
        c1, c2, c3 = st.columns(3)
        ent = c1.selectbox("Each option is identified by", [""] + cols_all, index=([""] + cols_all).index(plan["entity_column"]) if plan["entity_column"] in cols_all else 0,
                           format_func=lambda c: header.get(c, "(row number)") if c else "(row number)", key=f"ent_{dataset_key}_{plan_state['version']}")
        lab = c2.selectbox("Display name", [""] + cols_all, index=([""] + cols_all).index(plan["label_column"]) if plan["label_column"] in cols_all else 0,
                           format_func=lambda c: header.get(c, "(same as ID)") if c else "(same as ID)", key=f"lab_{dataset_key}_{plan_state['version']}")
        grp = c3.selectbox("Compare separately within", [""] + cols_all, index=([""] + cols_all).index(plan["group_column"]) if plan["group_column"] in cols_all else 0,
                           format_func=lambda c: header.get(c, "(no grouping)") if c else "(no grouping)", key=f"grp_{dataset_key}_{plan_state['version']}")

        # criteria editor: every usable numeric column (+ derived metrics); plan rows first
        in_plan = {c["column"]: c for c in plan["criteria"]}
        fmin = {f["column"]: f["value"] for f in plan.get("filter_suggestions", []) if f["operator"] == ">="}
        fmax = {f["column"]: f["value"] for f in plan.get("filter_suggestions", []) if f["operator"] == "<="}
        ignored = {i.get("column"): i.get("reason", "") for i in plan.get("ignored_columns", []) if isinstance(i, dict)}
        cand = profiler.candidate_criteria(profile) + [d["name"] for d in plan.get("derived_metrics", [])]
        rows = []
        for col in cand:
            c = in_plan.get(col)
            d_dir, _ = planner.guess_direction(col)
            p = next((x for x in profile if x["column"] == col), {})
            rows.append({"use": c is not None, "column": col,
                         "label": c["label"] if c else header.get(col, col),
                         "better": c["direction"] if c else d_dir,
                         "weight": float(c["weight"]) if c else 0.0,
                         "min allowed": pd.to_numeric(fmin.get(col), errors="coerce"),
                         "max allowed": pd.to_numeric(fmax.get(col), errors="coerce"),
                         "role": c.get("role", "other") if c else planner.guess_role(col),
                         "why": (c.get("reason", "") if c else ignored.get(col, "not in plan"))[:120],
                         "range in data": f"{fmt(p.get('min'))} … {fmt(p.get('max'))}" if p else "derived"})
        if not rows:
            st.error("This table has no numeric columns to rank on (e.g. price, lead time, quality %). "
                     "VendorIQ will not invent a ranking - add numeric columns or upload a different file.")
            st.stop()
        crit_df = pd.DataFrame(rows).sort_values(["use", "weight"], ascending=[False, False]).reset_index(drop=True)
        st.markdown("**Decision criteria** - tick, re-weight, flip direction or set limits. Limits accept **any value**.")
        edited = st.data_editor(
            crit_df, hide_index=True, width="stretch", key=f"crit_{dataset_key}_{plan_state['version']}",
            disabled=["column", "why", "range in data"],
            column_config={
                "use": st.column_config.CheckboxColumn("Use", width="small"),
                "better": st.column_config.SelectboxColumn("Better", options=["lower", "higher"], required=True, width="small"),
                "weight": st.column_config.NumberColumn("Weight", min_value=0, max_value=100, step=1, width="small"),
                "min allowed": st.column_config.NumberColumn("Min allowed", help="Knock out options below this value (any number)."),
                "max allowed": st.column_config.NumberColumn("Max allowed", help="Knock out options above this value (any number)."),
                "role": st.column_config.SelectboxColumn("Role", options=["cost", "quality", "delivery", "risk", "financial",
                                                                          "sustainability", "capacity", "other"], width="small"),
            })
        used = edited[edited["use"] & (edited["weight"] > 0)]
        limit_only = edited[~(edited["use"] & (edited["weight"] > 0)) & (edited["min allowed"].notna() | edited["max allowed"].notna())]
        tf_rows = [{"use": True, "column": f["column"], "operator": f["operator"], "value": f["value"]}
                   for f in plan.get("filter_suggestions", []) if f["operator"] not in (">=", "<=") or f["column"] not in cand]
        with st.expander(f"Text / category filters ({len(tf_rows)} suggested)", expanded=bool(tf_rows)):
            tf = st.data_editor(pd.DataFrame(tf_rows or [{"use": False, "column": "", "operator": "==", "value": ""}]),
                                hide_index=True, num_rows="dynamic", width="stretch", key=f"tf_{dataset_key}_{plan_state['version']}",
                                column_config={"use": st.column_config.CheckboxColumn("Use"),
                                               "column": st.column_config.SelectboxColumn("Column", options=cols_all),
                                               "operator": st.column_config.SelectboxColumn("Operator", options=["==", "!=", "contains", "not contains", ">=", "<="]),
                                               "value": st.column_config.TextColumn("Value")})
            text_filters = [r for r in tf.to_dict("records") if r.get("use") and r.get("column")]
        exclude_extremes = st.toggle("Exclude implausible values from ranking (unit errors, % above 100, unexpected negatives)", value=True)

        if plan.get("derived_metrics"):
            st.caption("Derived metrics proposed by the AI (validated as safe arithmetic): "
                       + "; ".join(f"**{d['name']}** = `{d['formula']}`" for d in plan["derived_metrics"]))

        if used.empty:
            st.error("Tick at least one criterion with a weight above 0.")
        else:
            keep = pd.concat([used, limit_only.assign(weight=0.0)])          # limit-only rows: filter, no weight
            eff = dict(plan, entity_column=ent, label_column=lab or ent, group_column=grp,
                       criteria=[{"column": r["column"], "label": r["label"], "direction": r["better"], "weight": float(r["weight"]),
                                  "role": r["role"], "reason": r["why"]} for _, r in keep.iterrows()])
            eff, elog = planner.validate_plan(eff, profile)
            for line in elog:
                st.caption("🔧 " + line)
            num_filters = [{"column": r["column"], "label": r["label"], "min": r["min allowed"], "max": r["max allowed"]}
                           for _, r in edited.iterrows() if (pd.notna(r["min allowed"]) or pd.notna(r["max allowed"]))]
            plan = eff
            with st.spinner("Cleaning the data according to the plan…"):
                prepared = prepare_cached(dataset_key, json.dumps(eff, sort_keys=True, default=str), exclude_extremes, comb.df, profile)
            st.markdown("**Cleaning log** - every automatic decision")
            if prepared.report:
                st.dataframe(pd.DataFrame(prepared.report), hide_index=True, width="stretch")
            else:
                st.caption("No cleaning needed: no gaps, duplicates or implausible values in the chosen criteria.")
            if plan.get("data_quality_notes") or plan.get("analysis_questions"):
                a, b = st.columns(2)
                with a:
                    if plan.get("data_quality_notes"):
                        st.markdown("**Data-quality notes (AI)**")
                        for n_ in plan["data_quality_notes"][:6]:
                            st.markdown(f"- {n_}")
                with b:
                    if plan.get("analysis_questions"):
                        st.markdown("**Questions worth asking (AI)**")
                        for n_ in plan["analysis_questions"][:4]:
                            st.markdown(f"- {n_}")


# ============================================================ weight adjustment (tab ③)
PRESET_FACTORS = {
    "Plan weights (AI / edited)": {},
    "Balanced (equal weights)": "equal",
    "Cost first": {"cost": 3.0},
    "Quality first": {"quality": 3.0},
    "Delivery / urgent": {"delivery": 3.0},
    "Risk-averse": {"risk": 3.0, "quality": 1.5, "financial": 1.5},
    "Working-capital saver": {"financial": 3.0, "cost": 1.5},
}


def _preset_weights(plan_criteria: list[dict], preset: str) -> dict:
    f = PRESET_FACTORS[preset]
    if f == "equal":
        return {c["column"]: 10 for c in plan_criteria}
    return {c["column"]: int(min(100, round(c["weight"] * f.get(c.get("role", "other"), 1.0)))) for c in plan_criteria}


def weight_panel(plan_criteria: list[dict], dkey: str) -> list[dict]:
    """Sliders to change the weights AFTER the plan (AI or keyword) has identified the criteria.
    Changing weights re-ranks instantly and never uses AI quota."""
    sig = hashlib.sha1(json.dumps([(c["column"], c["weight"]) for c in plan_criteria]).encode()).hexdigest()[:10]
    kp = f"w_{dkey}_{sig}_"

    def apply_preset():
        for col, w in _preset_weights(plan_criteria, ss[kp + "preset"]).items():
            ss[kp + col] = w

    for c in plan_criteria:
        ss.setdefault(kp + c["column"], int(round(c["weight"])))
    with st.expander("⚖️ Adjust weights - what matters more to you? (re-ranks instantly, no AI call)", expanded=True):
        c1, c2 = st.columns([3, 1])
        c1.selectbox("Start from a preset", list(PRESET_FACTORS), key=kp + "preset", on_change=apply_preset,
                     help="Presets scale the plan's weights by role (cost, quality, delivery, risk, financial). Fine-tune with the sliders.")
        c2.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        def reset():
            ss[kp + "preset"] = "Plan weights (AI / edited)"
            apply_preset()
        c2.button("↺ Reset to plan", width="stretch", key=kp + "reset", on_click=reset)
        cols = st.columns(3)
        for i, c in enumerate(plan_criteria):
            arrow = "↓ lower is better" if c["direction"] == "lower" else "↑ higher is better"
            cols[i % 3].slider(f"{c['label'][:32]}  ({arrow})", 0, 100, step=1, key=kp + c["column"],
                               help=f"Plan weight: {c['weight']:g}. Role: {c.get('role', 'other')}.")
        new = [dict(c, weight=float(ss[kp + c["column"]])) for c in plan_criteria]
        total = sum(c["weight"] for c in new)
        if total <= 0:
            st.warning("All weights are 0 - the plan's weights are used instead.")
            return plan_criteria
        st.caption("Effective weights: " + " · ".join(f"{c['label'][:24]} {c['weight'] / total * 100:.0f}%" for c in
                                                        sorted(new, key=lambda c: -c["weight"]) if c["weight"] > 0))
        changed = [c["label"] for c, p in zip(new, plan_criteria) if c["weight"] != p["weight"]]
        if changed:
            st.caption(f"✏️ Changed from the plan: {', '.join(changed[:6])}{' …' if len(changed) > 6 else ''}. "
                       "To add or remove criteria, flip directions or set limits, use tab ②.")
    return [c for c in new if c["weight"] > 0]

# ============================================================ ③ RANKING
scored = excluded = stab = pd.DataFrame()
criteria, rot, decision, group, pareto_ids, context = [], {}, ("", ""), "All", [], {}
id_to_name, id_pat = {}, None
with tab_rank:
    if prepared is None or not any(c["weight"] > 0 for c in prepared.criteria):
        st.info("Complete tabs ① and ② first.")
    else:
        plan_criteria = [c for c in prepared.criteria if c["weight"] > 0]      # zero-weight rows are filter-only
        criteria = weight_panel(plan_criteria, dataset_key)
        data = prepared.data
        groups = data["_group"].value_counts()
        if len(groups) > 1:
            group = st.selectbox(f"Compare options within ({plan['group_column']})", list(groups.index),
                                 format_func=lambda g: f"{g}  ({groups[g]:,} options)")
        else:
            group = groups.index[0]
        pool = data[data["_group"] == group]
        qualified, excluded = planner.apply_filters(pool, comb.df, num_filters, text_filters)
        scored = scoring.score(qualified, criteria)
        stab = scoring.stability(scored, criteria)
        pareto = scoring.pareto_front(scored, criteria)
        scored["pareto"] = pareto.values
        pareto_ids = scored.loc[scored["pareto"], "_id"].tolist()
        rot = scoring.rule_of_thumb(scored, criteria)
        decision = scoring.decision_label(scored, stab, criteria)
        id_to_name = dict(zip(pool["_id"].astype(str).head(20000), pool["_label"].astype(str).head(20000)))
        id_pat = ai.id_regex(list(pool["_id"].astype(str).head(5000)))

        icon = {"Recommend": "✅", "Recommend with conditions": "⚠️", "Refer to committee": "🔎"}[decision[0]]
        {"Recommend": st.success, "Recommend with conditions": st.warning, "Refer to committee": st.error}[decision[0]](
            f"**{icon} {decision[0]}** — {decision[1]}")

        if scored.empty:
            st.markdown("**No option passes the filters.** Excluded options and reasons:")
            st.dataframe(excluded[["_id", "_label", "_excluded_reason"]].head(50), hide_index=True, width="stretch")
        else:
            top = scored.iloc[0]
            win = float(stab.loc[stab["_id"] == top["_id"], "win_share_pct"].iloc[0]) if not stab.empty else 0.0
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Options ranked", f"{len(scored):,} of {len(pool):,}")
            k2.metric("#1 option", str(top["_id"])[:18], str(top["_label"])[:28] if top["_label"] != top["_id"] else None,
                      delta_color="off", delta_arrow="off")
            k3.metric("Score", f"{top['score']:.1f} / 100", top["tier"], delta_color="off", delta_arrow="off")
            k4.metric("Rank stability", f"{win:.0f}%", help="Share of 500 random ±25% weight variations in which this option stays #1.")
            k5.metric("Pareto-efficient", f"{len(pareto_ids):,}", help="Options that no other option beats on every criterion at once.")

            st.markdown("#### Ranked options")
            show = pd.DataFrame({"#": scored["rank"], "ID": scored["_id"], "Name": scored["_label"], "Tier": scored["tier"],
                                 "Score": scored["score"], "Pareto": scored["pareto"].map({True: "✓", False: ""})})
            for c in criteria:
                show[f"{c['label']} ({'↓' if c['direction'] == 'lower' else '↑'})"] = scored[scoring.ccol(c)].map(lambda v: float(f"{v:.4g}"))
            show["Data flags"] = scored["_flags"]
            if len(show) > TABLE_ROWS_SHOWN:
                st.caption(f"Showing the top {TABLE_ROWS_SHOWN} of {len(show):,}. Download the CSV for all rows.")
            st.dataframe(show.head(TABLE_ROWS_SHOWN), hide_index=True, width="stretch",
                         column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f")})
            buf = io.StringIO()
            show.to_csv(buf, index=False)
            st.download_button("⬇ Download full ranking (CSV)", buf.getvalue(), f"vendoriq_ranking_{group}.csv", "text/csv")

            # colour slots: 7 heaviest criteria get fixed colours, the rest fold into "Other criteria"
            by_w = sorted(criteria, key=lambda c: -c["weight"])
            shown_c, other_c = by_w[:7], by_w[7:]
            g1, g2 = st.columns([3, 2], gap="large")
            with g1:
                st.markdown("#### Why this order? Points per criterion (top 15)")
                t15 = scored.head(15)
                ylab = [f"{i} · {str(n)[:22]}" if str(n) != str(i) else str(i) for i, n in zip(t15["_id"], t15["_label"])]
                fig = go.Figure()
                for i, c in enumerate(shown_c):
                    fig.add_bar(y=ylab, x=t15[f"pts__{c['column']}"], name=c["label"][:28], orientation="h",
                                marker=dict(color=PALETTE[i], line=dict(color="#ffffff", width=2)),
                                customdata=t15[scoring.ccol(c)],
                                hovertemplate=f"<b>%{{y}}</b><br>{c['label']}: %{{customdata:,.4g}}<br>Points: %{{x:.1f}}<extra></extra>")
                if other_c:
                    fig.add_bar(y=ylab, x=sum(t15[f"pts__{c['column']}"] for c in other_c), name="Other criteria", orientation="h",
                                marker=dict(color=OTHER_GREY, line=dict(color="#ffffff", width=2)),
                                hovertemplate="<b>%{y}</b><br>Other criteria: %{x:.1f} pts<extra></extra>")
                fig.update_layout(barmode="stack", height=170 + 34 * len(t15), margin=dict(l=190, r=20, t=80, b=40),
                                  yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#0b0b0b")),
                                  xaxis=dict(title=dict(text="Score points (out of 100)", font=dict(size=12, color="#52514e")),
                                             range=[0, 100], dtick=20, gridcolor="#e1e0d9", tickfont=dict(color="#52514e")),
                                  legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal", font=dict(size=11)),
                                  plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35,
                                  font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                st.plotly_chart(fig, width="stretch", theme=None)
            with g2:
                st.markdown("#### Rank stability (500 weight variations)")
                s2 = stab[stab.win_share_pct > 0].head(8)
                fig2 = go.Figure(go.Bar(x=s2["win_share_pct"], y=s2["_id"].astype(str), orientation="h", marker_color=PALETTE[0],
                                        text=[f"{v:.0f}%" for v in s2["win_share_pct"]], textposition="outside",
                                        hovertemplate="%{y}: #1 in %{x:.1f}% of runs<extra></extra>"))
                fig2.update_layout(height=110 + 42 * max(len(s2), 1), margin=dict(l=10, r=30, t=10, b=40),
                                   xaxis=dict(range=[0, 115], title=dict(text="% of runs ranked #1", font=dict(size=12, color="#52514e")),
                                              gridcolor="#e1e0d9", tickfont=dict(color="#52514e")),
                                   yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=12, color="#0b0b0b")),
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.4,
                                   font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                st.plotly_chart(fig2, width="stretch", theme=None)
                st.caption("Below 70% means the #1 spot depends on small changes in weights - treat it as a close call.")
                if len(scored) >= 2 and len(criteria) >= 3:
                    st.markdown("#### Top 3 profile")
                    cats = [c["label"][:18] for c in shown_c]
                    fig3 = go.Figure()
                    for i, (_, r) in enumerate(scored.head(3).iterrows()):
                        vals = [float(r[f"norm__{c['column']}"]) for c in shown_c]
                        fig3.add_scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=str(r["_id"]), line=dict(color=TOP3[i], width=2))
                    fig3.update_layout(height=360, margin=dict(l=90, r=90, t=30, b=30), paper_bgcolor="rgba(0,0,0,0)",
                                       polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#e1e0d9"),
                                                  angularaxis=dict(tickfont=dict(size=10, color="#52514e"), gridcolor="#e1e0d9"),
                                                  bgcolor="rgba(0,0,0,0)"),
                                       legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                                       font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                    st.plotly_chart(fig3, width="stretch", theme=None)
                    st.caption("1 = best in this pool on that criterion (relative, not absolute).")

            st.markdown("#### Expert cross-check: the single-criterion rule of thumb")
            if rot.get("same"):
                st.success(f"The #1 option **{rot['top_id']}** is also the **{rot['rule']}** option - model and rule of thumb agree.", icon="🤝")
            else:
                pct = f" ({rot['gap_pct']:.1f}%)" if rot.get("gap_pct") is not None else ""
                st.info(f"Rule of thumb ({'L1: ' if rot.get('rule_is_l1') else ''}{rot['rule']}) picks **{rot['rule_id']}**. "
                        f"The model prefers **{rot['top_id']}**, which is **{fmt(abs(rot['gap']))}{pct} "
                        f"{'worse' if rot['gap'] != 0 else 'different'}** on {rot['criterion']} ({fmt(rot['top_value'])} vs {fmt(rot['rule_value'])}). "
                        "In return you get: " + ("; ".join(rot["what_you_get"]) if rot["what_you_get"] else
                                                 "no measurable advantage - reconsider the weights") + ".", icon="⚖️")
        if not excluded.empty:
            with st.expander(f"Excluded from this pool ({len(excluded):,})", expanded=scored.empty):
                st.dataframe(excluded[["_id", "_label", "_excluded_reason"]].rename(
                    columns={"_id": "ID", "_label": "Name", "_excluded_reason": "Reason"}).head(500), hide_index=True, width="stretch")
        if not scored.empty:
            trade = analysis.trade_offs(scored, criteria)
            context = analysis.build_context(scored, excluded, criteria, plan, group, stab, rot, pareto_ids, trade,
                                             prepared.report, anonymise=anonymise)

# ============================================================ ④ AI INSIGHTS
with tab_ai:
    if not context:
        st.info("Complete tab ③ first - the analysis runs on the ranked pool shown there.")
    else:
        st.markdown(f"#### Further analysis for **{group}** (computed by Python)")
        a, b = st.columns([3, 2], gap="large")
        with a:
            trade = context["trade_offs"]
            if trade:
                st.markdown("**Trade-offs and synergies between criteria** (Spearman, oriented so + = good goes with good)")
                st.dataframe(pd.DataFrame(trade)[["a", "b", "rho", "type", "meaning"]].rename(
                    columns={"a": "Criterion A", "b": "Criterion B", "rho": "ρ", "type": "Type", "meaning": "Meaning"}),
                    hide_index=True, width="stretch")
            else:
                st.caption("No notable correlation between criteria (or too few options).")
            cm = analysis.correlation_matrix(scored, criteria) if len(scored) >= 5 else pd.DataFrame()
            if len(cm) >= 2:
                heat = go.Figure(go.Heatmap(z=cm.values, x=[c[:16] for c in cm.columns], y=[c[:16] for c in cm.index],
                                            zmin=-1, zmax=1, colorscale=[[0, "#d03b3b"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                                            text=cm.values, texttemplate="%{text:.2f}", hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"))
                heat.update_layout(height=160 + 34 * len(cm), margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                                   xaxis=dict(automargin=True, tickangle=-35, tickfont=dict(color="#52514e")),
                                   yaxis=dict(automargin=True, autorange="reversed", tickfont=dict(color="#0b0b0b")),
                                   font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=11))
                st.plotly_chart(heat, width="stretch", theme=None)
                st.caption("Blue = synergy (good on one, good on the other). Red = trade-off. Grey = no relation.")
        with b:
            st.markdown("**Criterion ranges in this pool**")
            st.dataframe(pd.DataFrame(context["criterion_stats"]).T.reset_index().rename(columns={"index": "criterion"}),
                         hide_index=True, width="stretch")
            st.markdown(f"**Pareto-efficient options:** {len(pareto_ids):,}")
            st.caption(", ".join(map(str, pareto_ids[:25])) + (" …" if len(pareto_ids) > 25 else ""))
            gs = analysis.group_summary(prepared.data, criteria)
            if not gs.empty:
                st.markdown("**Comparison across groups** (medians)")
                st.dataframe(gs.rename_axis(header_of.get(plan["group_column"], "group")), width="stretch")

        st.divider()
        top_id = str(scored.iloc[0]["_id"])
        ctx_hash = ai.cache_key("insights", PROMPT_VERSION, models, ai.to_json(context))
        c1, c2 = st.columns([1, 3])
        run = c1.button("✨ Generate AI insights", type="primary", width="stretch")
        c2.caption("Gemini explains the ranking and the analysis above. Same inputs → same answer (cached).")
        if run:
            prev = ss.insights.get(ctx_hash)
            if prev and prev["source"] != "rule-based":
                pass
            elif ctx_hash in shared_ai_cache():
                ss.insights[ctx_hash] = {**shared_ai_cache()[ctx_hash], "source": "cache"}
                log_event("insights", ai.AIResult(ok=True, source="cache", model=shared_ai_cache()[ctx_hash]["model"]))
            else:
                allowed, why = can_call_ai() if api_key else (False, "No API key configured.")
                res = ai.AIResult(ok=False, source="rule-based", error=why)
                if allowed:
                    with st.spinner("Gemini is reading the ranking and the analysis…"):
                        res = ai.get_insights(api_key, context, models)
                    count_call()
                log_event("insights", res)
                fallback = analysis.rule_based_insights(scored, criteria, rot, stab, decision, context["trade_offs"], pareto_ids)
                if res.ok:
                    checks = ai.validate_insights(res.parsed, context, set(scored["_id"].astype(str)), top_id)
                    hard = any(not c["passed"] for c in checks if c["check"] in ("Recommended option exists in the ranked pool", "No invented option IDs"))
                    entry = {"parsed": res.parsed, "checks": checks, "source": "gemini", "model": res.model,
                             "latency_ms": res.latency_ms, "tokens": res.tokens_in + res.tokens_out, "error": ""}
                    if hard:
                        entry.update(parsed=fallback, source="rule-based", rejected_ai=res.parsed,
                                     error="AI output failed a safety check - rule-based summary shown instead.")
                    else:
                        shared_ai_cache()[ctx_hash] = entry
                    ss.insights[ctx_hash] = entry
                else:
                    ss.insights[ctx_hash] = {"parsed": fallback, "checks": [], "source": "rule-based", "model": "-",
                                             "latency_ms": 0, "tokens": 0, "error": res.error}
        rec = ss.insights.get(ctx_hash)
        if rec is None:
            st.caption("Press the button to generate. If the AI is unavailable, a rule-based summary is shown instead.")
        else:
            p = rec["parsed"]
            L = lambda t: ai.label_ids(str(t), id_to_name, id_pat)  # noqa: E731
            if rec["source"] == "rule-based":
                st.markdown(f'<span class="fb-badge">Rule-based fallback</span> &nbsp; <span class="small-note">{rec["error"]}</span>', unsafe_allow_html=True)
            else:
                extra = "served from cache" if rec["source"] == "cache" else f"{rec['latency_ms'] / 1000:.1f}s · {rec['tokens']} tokens"
                st.markdown(f'<span class="ai-badge">🤖 AI-generated · {rec["model"]} · {extra}</span>', unsafe_allow_html=True)
            st.markdown(f"### {L(p.get('headline', ''))}")
            st.markdown(f"**Confidence:** {p.get('confidence', '')}  ·  **Score-model #1:** {top_id}  ·  **Decision rule:** {decision[0]}")
            if p.get("disagreement_note"):
                st.warning("AI concern about the #1 option: " + L(p["disagreement_note"]), icon="🧐")
            a, b = st.columns(2, gap="large")
            with a:
                for title, key_ in (("Why this option", "why"), ("Trade-offs", "trade_offs"), ("Negotiation levers", "negotiation_levers")):
                    if p.get(key_):
                        st.markdown(f"**{title}**")
                        for w in p[key_]:
                            st.markdown(f"- {L(w)}")
            with b:
                if p.get("risks"):
                    st.markdown("**Risks to watch**")
                    for r in p["risks"]:
                        st.markdown(f"- **{r.get('id', '')}** - {L(r.get('risk', ''))}")
                for title, key_ in (("Anomalies in the data", "anomalies"), ("Further analysis to run", "further_analysis"), ("Data gaps", "data_gaps")):
                    if p.get(key_):
                        st.markdown(f"**{title}**")
                        for w in p[key_]:
                            st.markdown(f"- {L(w)}")
            if rec["checks"]:
                n_ok = sum(c["passed"] for c in rec["checks"])
                with st.expander(f"🔍 AI output checks - {n_ok}/{len(rec['checks'])} passed", expanded=n_ok < len(rec["checks"])):
                    st.dataframe(pd.DataFrame([{**c, "passed": "✅" if c["passed"] else "⚠️"} for c in rec["checks"]]),
                                 hide_index=True, width="stretch")
                    if rec.get("rejected_ai"):
                        st.json(rec["rejected_ai"])
            with st.expander("Exactly what was sent to the AI (transparency)"):
                st.json(context, expanded=False)
            md = [f"# VendorIQ insights - {group}", f"_Generated {datetime.now():%d %b %Y %H:%M} · source: {rec['source']} · model: {rec['model']}_",
                  "", f"**{p.get('headline', '')}**", "", f"Decision rule: {decision[0]} - {decision[1]}"]
            for title, key_ in (("Why", "why"), ("Trade-offs", "trade_offs"), ("Negotiation levers", "negotiation_levers"),
                                ("Anomalies", "anomalies"), ("Further analysis", "further_analysis")):
                md += ["", f"## {title}"] + [f"- {x}" for x in p.get(key_, [])]
            md += ["", "## Risks"] + [f"- {r.get('id')}: {r.get('risk')}" for r in p.get("risks", [])]
            md += ["", "## Top 10", "| # | ID | Name | Score | Tier |", "|---|---|---|---|---|"]
            md += [f"| {r['rank']} | {r['_id']} | {r['_label']} | {r['score']:.1f} | {r['tier']} |" for _, r in scored.head(10).iterrows()]
            md += ["", "_AI-assisted decision support. Verify before issuing a purchase order._"]
            st.download_button("⬇ Download insights report (.md)", "\n".join(md), "vendoriq_insights.md")

# ============================================================ ⑤ CHAT
with tab_chat:
    if not context:
        st.info("Complete tab ③ first - the assistant answers questions about that ranking.")
    else:
        st.info(f"You are chatting with an **AI assistant (Google Gemini)**, not a person. It knows the top-ranked options and the analysis "
                f"for **{group}** and remembers the last {ai.MAX_HISTORY_TURNS} messages.", icon="🤖")
        top_id = str(scored.iloc[0]["_id"])
        c_a = criteria[0]["label"]
        c_b = criteria[1]["label"] if len(criteria) > 1 else c_a
        suggestions = [f"Why is {top_id} ranked first?", f"Compare the top 2 on {c_a} and {c_b}",
                       "What are the main trade-offs?", f"Draft a short RFQ email to {top_id}"]
        picked = st.pills("Try:", suggestions, key="chat_pill", label_visibility="collapsed")
        question = ss.get("chat_box") or (picked if picked and ss.get("last_pill") != picked else None)
        if picked:
            ss.last_pill = picked
        if question:
            allowed, canned = ai.screen_user_message(question)
            if not allowed:
                answer, meta = canned, "Blocked before reaching the AI (input screening)."
                log_event("chat", ai.AIResult(ok=False, source="blocked", error="screened"), question[:40])
            elif not api_key:
                answer, meta = ai.offline_answer(question, scored, criteria), "Rule-based (AI offline)."
                log_event("chat", ai.AIResult(ok=False, source="rule-based", error="no key"))
            else:
                ok, why = can_call_ai()
                if not ok:
                    answer, meta = ai.offline_answer(question, scored, criteria), why
                else:
                    with st.spinner("Thinking…"):
                        res = ai.chat_reply(api_key, dict(context, top_options=analysis.build_context(
                            scored, excluded, criteria, plan, group, stab, rot, pareto_ids, context["trade_offs"],
                            prepared.report, anonymise=anonymise, top_n=20)["top_options"]), ss.chat, question, models)
                    count_call()
                    log_event("chat", res)
                    if res.ok:
                        answer, meta = res.text, f"AI-generated · {res.model} · {res.latency_ms / 1000:.1f}s"
                    else:
                        answer, meta = ai.offline_answer(question, scored, criteria), f"AI unavailable ({res.error}) - rule-based answer shown."
            ss.chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "meta": meta}]
        for m in ss.chat:
            with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🤖"):
                st.markdown(ai.label_ids(m["content"], id_to_name, id_pat) if m["role"] != "user" else m["content"])
                if m.get("meta"):
                    st.caption(m["meta"])
        st.chat_input("Ask about the options, e.g. 'Why is the runner-up second?'", key="chat_box", max_chars=ai.MAX_CHAT_CHARS)
        if ss.chat and st.button("Clear chat"):
            ss.chat = []
            st.rerun()

# ============================================================ ⑥ ABOUT
with tab_about:
    st.markdown("### How VendorIQ works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; ranksep=0.25; nodesep=0.3; pad=0.1;
      node [shape=box, style="rounded,filled", fillcolor="#f4f7fb", color="#9ec5f4", fontname="Helvetica", fontsize=15, margin="0.15,0.08"];
      edge [color="#898781", arrowsize=0.7];
      A [label="1..N files\\nany format"]; B [label="Combine\\n& profile"];
      C [label="Gemini\\nEXPLORES", fillcolor="#fff5e6", color="#f6d79a"]; D [label="Gemini\\nJUDGES", fillcolor="#fff5e6", color="#f6d79a"];
      E [label="Python\\nvalidates"]; F [label="You edit\\nthe plan"]; G [label="Clean, rank\\n& analyse"];
      H [label="Gemini\\nexplains", fillcolor="#fff5e6", color="#f6d79a"]; I [label="Committee\\ndecides", fillcolor="#eaf6ea", color="#9fd39f"];
      A -> B -> C -> D -> E -> F -> G -> H -> I; B -> E [label="keyword plan\\n(no AI)", style=dashed, fontsize=11, fontcolor="#52514e"]; }
    """, width="stretch")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
**Division of labour**
- **Gemini** explores a *profile* of the data (column names, types, statistics) and proposes the plan;
  a **second Gemini call judges** that plan; later Gemini **explains** the results and answers questions.
- **Python** reads, combines and cleans the data, validates the AI's plan, computes every score,
  the stability test, the Pareto set, trade-offs and the rule-of-thumb check.
- **You** can change every criterion, weight, direction and limit - and you make the decision.

**Scale** - up to {ingest.MAX_FILES} files per upload, up to {ingest.MAX_TOTAL_ROWS:,} rows analysed
(larger data is sampled), any column names, any value range. The AI never receives the whole file,
so its cost does not grow with the data.
""")
    with c2:
        st.markdown("""
**Known limits - do not use unsupervised for**
- Final vendor award or PO release (needs committee approval).
- Anything that needs *current* market prices - the model does not know them.
- Columns whose meaning is ambiguous: check the plan's directions before trusting a ranking.

**Privacy** - with anonymisation on, names, IDs and free-text examples are masked before anything is
sent to Gemini. The API key is stored in Streamlit secrets, never in code.
""")
    st.markdown("### Audit log (this session)")
    if ss.audit:
        st.dataframe(pd.DataFrame(ss.audit), hide_index=True, width="stretch")
        st.download_button("⬇ Download audit log", pd.DataFrame(ss.audit).to_csv(index=False), "vendoriq_audit_log.csv")
    else:
        st.caption("No AI calls yet.")
