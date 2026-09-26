"""
VendorIQ - AI-assisted vendor selection for chemical raw materials.
End-term project: AI Applications (Use case #6 - Vendor selection / procurement recommender).

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)
"""
from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ai_engine as ai
from prompts import PROMPT_VERSION
from scoring import (CRITERIA, PRESETS, PURITY_SPEC, add_derived_metrics, apply_hard_filters,
                     decision_label, l1_cross_check, rule_based_summary, score_vendors,
                     stability_analysis, weights_to_pct)
from validation import validate_requirement, validate_vendor_data

APP_VERSION = "1.0"
MAX_AI_CALLS_PER_SESSION = 30
MIN_SECONDS_BETWEEN_CALLS = 2
HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "sample_data", "vendors_sample.csv")
EDGE = os.path.join(HERE, "sample_data", "vendors_edge_cases.csv")
TEMPLATE = os.path.join(HERE, "sample_data", "vendor_template.csv")

# Categorical palette (validated for colour-blind separation, fixed order).
COLORS = {"cost": "#2a78d6", "quality": "#eb6834", "lead_time": "#1baf7a", "reliability": "#eda100",
          "payment": "#e87ba4", "risk": "#4a3aa7", "sustainability": "#008300"}
TOP3 = ["#2a78d6", "#eb6834", "#1baf7a"]

st.set_page_config(page_title="VendorIQ - AI Vendor Selection", page_icon="🏭", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
  div[data-testid="stMetricValue"] {font-size: 1.45rem;}
  .small-note {color: #6b6b66; font-size: 0.85rem;}
  .ai-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             background:#eef4fc; color:#1c5cab; border:1px solid #cde2fb;}
  .fb-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.8rem;
             background:#fff5e6; color:#8a5a00; border:1px solid #f6d79a;}
</style>
""", unsafe_allow_html=True)


# ============================================================ helpers & state
def secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default) or os.environ.get(name, default)
    except Exception:                       # no secrets.toml present
        return os.environ.get(name, default)


@st.cache_resource
def shared_ai_cache() -> dict:
    """Shared across all users of the deployed app: identical requests reuse
    the earlier answer instead of spending free-tier quota again."""
    return {}


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])


def init_state():
    ss = st.session_state
    if "initialised" in ss:
        return
    qp = st.query_params                      # settings survive a page refresh via the URL
    ss.initialised = True
    ss.source = "Sample dataset"
    ss.preset = qp.get("preset", "Balanced") if qp.get("preset") in PRESETS else "Balanced"
    for k, v in PRESETS[ss.preset].items():
        ss[f"w_{k}"] = v
    ss.qty = float(qp.get("qty", 25))
    ss.days = int(qp.get("days", 15))
    ss.material_qp = qp.get("material", "Methanol")
    ss.chat = []
    ss.audit = []
    ss.ai_calls = 0
    ss.last_call = 0.0
    ss.recs = {}                               # context-hash -> result (per session)
    ss.upload_bytes = None


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
        return False, f"Session limit of {MAX_AI_CALLS_PER_SESSION} AI calls reached (protects the free quota). Rule-based mode is used."
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - ss.last_call)
    if wait > 0:
        time.sleep(wait)
    return True, ""


def on_preset_change():
    for k, v in PRESETS[st.session_state.preset].items():
        st.session_state[f"w_{k}"] = v


def on_material_change():
    st.session_state.min_purity = float(PURITY_SPEC.get(st.session_state.material, 0.0))


init_state()
ss = st.session_state

# ============================================================ sidebar
with st.sidebar:
    st.markdown("### 🏭 VendorIQ")
    st.caption(f"v{APP_VERSION} · prompt {PROMPT_VERSION}")

    user_key = st.text_input("Gemini API key (optional)", type="password",
                             help="Leave blank to use the app owner's key. A key typed here stays in this browser session only and is never stored.")
    api_key = user_key.strip() or secret("GEMINI_API_KEY")
    preferred = secret("GEMINI_MODEL")
    models = ([preferred] if preferred else []) + [m for m in ai.DEFAULT_MODELS if m != preferred]

    if api_key:
        st.success("AI: Gemini key configured", icon="🟢")
        st.caption("Model order: " + " → ".join(models[:3]) + " …")
    else:
        st.warning("AI: offline - rule-based explanations only", icon="🟠")

    st.markdown("**Privacy**")
    anonymise = st.toggle("Anonymise vendor names sent to AI", value=True,
                          help="Vendor names and cities are replaced by vendor IDs before anything is sent to Google's Gemini API.")
    st.caption("Only the ranked table (numbers, IDs, notes) is sent to Google Gemini. "
               "Free-tier API data may be used by Google to improve its products - do not upload confidential prices.")

    st.markdown("**Usage this session**")
    st.progress(min(ss.ai_calls / MAX_AI_CALLS_PER_SESSION, 1.0),
                text=f"{ss.ai_calls} / {MAX_AI_CALLS_PER_SESSION} AI calls")

    if st.button("↺ Reset session", width="stretch"):
        for k in list(ss.keys()):
            del ss[k]
        st.query_params.clear()
        st.rerun()

# ============================================================ header
st.title("VendorIQ · AI-assisted vendor selection")
st.markdown("Rank raw-material vendors on **cost, quality, delivery, credit, risk and ESG** with weights you control; "
            "then let **Gemini explain** the result, flag risks and suggest negotiation levers.")
st.caption("⚠️ Sample vendors and prices are fictional. Output is decision support - the purchase committee makes the final decision.")

tab_data, tab_rank, tab_ai, tab_chat, tab_about = st.tabs(
    ["① Data", "② Requirement & ranking", "③ AI recommendation", "④ Ask VendorIQ (chat)", "⑤ How it works & limits"])

# ============================================================ ① DATA
with tab_data:
    c1, c2 = st.columns([2, 1])
    with c1:
        src = st.radio("Choose data source", ["Sample dataset", "Upload your CSV", "Edge-case test file"],
                       key="source", horizontal=True)
    with c2:
        with open(TEMPLATE, "rb") as f:
            st.download_button("⬇ Download CSV template", f, "vendor_template.csv", "text/csv", width="stretch")

    raw, size = None, None
    if src == "Sample dataset":
        raw = load_csv(SAMPLE)
        st.caption("26 fictional vendors across 5 raw materials used in resin, formaldehyde and rubber-chemical plants.")
    elif src == "Edge-case test file":
        raw = load_csv(EDGE)
        st.caption("Deliberately broken rows (blank price, negative lead time, OTIF 150%, duplicate ID, text in number, "
                   "unit error, prompt injection) to show how validation protects the model.")
    else:
        up = st.file_uploader("Upload vendor CSV (max 2 MB, 2,000 rows)", type=["csv"])
        if up is not None:
            size = up.size
            try:
                raw = pd.read_csv(up, dtype=str, keep_default_na=False, na_values=[""])
            except Exception as e:
                st.error(f"Could not read this file as CSV ({type(e).__name__}). Save it as CSV (UTF-8) and try again.")
        else:
            st.info("Upload a CSV in the template format, or switch back to the sample dataset.")

    report = validate_vendor_data(raw, size) if raw is not None else None

    if report is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows in file", report.rows_in)
        m2.metric("Accepted", len(report.clean))
        m3.metric("Repaired & flagged", report.rows_repaired)
        m4.metric("Rejected", report.rows_rejected)
        if report.fatal:
            st.error(report.fatal)
        for w in report.warnings:
            st.warning(w)
        if not report.issues.empty:
            st.markdown("**Validation log** - every automatic decision is listed here")
            sev_icon = {"Error": "⛔ Error", "Warning": "⚠️ Warning", "Security": "🛡️ Security", "Info": "ℹ️ Info"}
            show = report.issues.copy()
            show["severity"] = show["severity"].map(sev_icon)
            st.dataframe(show, hide_index=True, width="stretch")
        elif report.ok:
            st.success("All rows passed validation.", icon="✅")
        if report.ok:
            with st.expander("Preview clean data", expanded=False):
                st.dataframe(report.clean, hide_index=True, width="stretch")

data_ok = report is not None and report.ok
vendors = add_derived_metrics(report.clean) if data_ok else pd.DataFrame()

# ============================================================ ② RANKING
scored = disq = stab = pd.DataFrame()
l1, decision, req, w_pct, context = {}, ("", ""), {}, {}, {}

with tab_rank:
    if not data_ok:
        st.info("Load valid data in tab ① first.")
    else:
        materials = sorted(vendors["material"].unique())
        if "material" not in ss or ss.material not in materials:
            ss.material = ss.material_qp if ss.material_qp in materials else materials[0]
            ss.min_purity = float(PURITY_SPEC.get(ss.material, 0.0))

        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("1 · Your requirement")
            st.selectbox("Raw material", materials, key="material", on_change=on_material_change)
            a, b = st.columns(2)
            a.number_input("Quantity needed (MT)", min_value=0.0, max_value=20000.0, step=5.0, key="qty")
            b.slider("Needed within (days)", 1, 180, key="days")
            a.number_input("Minimum purity spec (%)", 0.0, 100.0, step=0.01, key="min_purity", format="%.2f",
                           help="Pre-filled from plant QC spec; vendors below this are knocked out.")
            b.number_input("Max landed cost (Rs/kg, 0 = no limit)", 0.0, 100000.0, step=0.5, key="budget")
            st.toggle("Vendor must have ISO 9001", value=True, key="iso")
            if ss.material not in PURITY_SPEC:
                st.caption("No stored purity spec for this material - enter one if it applies.")

        with right:
            st.subheader("2 · What matters more?")
            st.selectbox("Start from a preset", list(PRESETS), key="preset", on_change=on_preset_change)
            cols = st.columns(2)
            for i, (k, meta) in enumerate(CRITERIA.items()):
                cols[i % 2].slider(f"{meta['label']} ({'lower' if meta['better']=='lower' else 'higher'} is better)",
                                   0, 50, step=5, key=f"w_{k}")
            weights = {k: ss[f"w_{k}"] for k in CRITERIA}

        errs = validate_requirement(ss.qty, ss.days, weights)
        if errs:
            for e in errs:
                st.error(e)
        else:
            w_pct = weights_to_pct(weights)
            st.caption("Effective weights: " + " · ".join(f"{CRITERIA[k]['label']} {v*100:.0f}%" for k, v in w_pct.items() if v > 0))
            st.query_params.update(material=ss.material, preset=ss.preset, qty=str(ss.qty), days=str(ss.days))

            pool = vendors[vendors["material"] == ss.material]
            qual, disq = apply_hard_filters(pool, ss.qty, ss.days, ss.min_purity, ss.iso,
                                            ss.budget if ss.budget > 0 else None)
            scored = score_vendors(qual, weights)
            stab = stability_analysis(qual, weights)
            l1 = l1_cross_check(scored, ss.qty)
            decision = decision_label(scored, stab)
            req = {"material": ss.material, "qty_mt": ss.qty, "needed_within_days": ss.days,
                   "min_purity_pct": ss.min_purity, "iso_9001_required": ss.iso,
                   "max_landed_cost_rs_per_kg": ss.budget or None}

            st.divider()
            icon = {"Recommend": "✅", "Recommend with conditions": "⚠️", "Refer to committee": "🔎"}[decision[0]]
            box = {"Recommend": st.success, "Recommend with conditions": st.warning, "Refer to committee": st.error}[decision[0]]
            box(f"**{icon} {decision[0]}** — {decision[1]}")

            if scored.empty:
                st.markdown("**No vendor meets all hard requirements.** Nearest options (fewest knock-outs):")
                near = disq.assign(n=disq["disqualified_because"].str.count(";") + 1).sort_values(["n", "landed_cost"])
                st.dataframe(near[["vendor_id", "vendor_name", "disqualified_because"]].head(3), hide_index=True, width="stretch")
            else:
                top = scored.iloc[0]
                win = float(stab.loc[stab.vendor_id == top.vendor_id, "win_share_pct"].iloc[0])
                k1, k2, k3, k4, k5 = st.columns(5)
                k1.metric("Qualified vendors", f"{len(scored)} of {len(pool)}")
                k2.metric("#1 vendor", top["vendor_id"], top["vendor_name"], delta_color="off", delta_arrow="off")
                k3.metric("Score", f"{top['score']:.1f} / 100", top["tier"], delta_color="off", delta_arrow="off")
                k4.metric("Rank stability", f"{win:.0f}%", help="Share of 500 random ±25% weight variations in which this vendor stays #1.")
                k5.metric("Order value", f"₹{top['landed_cost']*ss.qty*1000/1e5:,.2f} L", f"{ss.qty:g} MT × ₹{top['landed_cost']:.2f}/kg", delta_color="off", delta_arrow="off")

                st.markdown("#### Ranked shortlist")
                table = scored[["rank", "vendor_id", "vendor_name", "tier", "score", "landed_cost", "lead_time_days",
                                "otif_pct", "qc_pass_rate_pct", "credit_days", "risk_points", "esg_score", "origin"]]
                st.dataframe(table, hide_index=True, width="stretch", column_config={
                    "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                    "landed_cost": st.column_config.NumberColumn("Landed ₹/kg", format="%.2f"),
                    "lead_time_days": "Lead (d)", "otif_pct": "OTIF %", "qc_pass_rate_pct": "QC pass %",
                    "credit_days": "Credit (d)", "risk_points": "Risk pts", "esg_score": "ESG",
                    "vendor_name": "Vendor", "vendor_id": "ID", "rank": "#", "tier": "Tier", "origin": "Origin"})

                g1, g2 = st.columns([3, 2], gap="large")
                with g1:
                    st.markdown("#### Why this order? Points earned per criterion")
                    fig = go.Figure()
                    ylab = [f"{r.vendor_id} · {r.vendor_name[:22]}" for r in scored.itertuples()]
                    for k, meta in CRITERIA.items():
                        if w_pct.get(k, 0) == 0:
                            continue
                        fig.add_bar(y=ylab, x=scored[f"pts_{k}"], name=meta["label"], orientation="h",
                                    marker=dict(color=COLORS[k], line=dict(color="#ffffff", width=2)),
                                    customdata=scored[meta["column"]],
                                    hovertemplate=f"<b>%{{y}}</b><br>{meta['label']}: %{{customdata}} {meta['unit']}<br>Points: %{{x:.1f}}<extra></extra>")
                    fig.update_layout(barmode="stack", height=150 + 48 * len(scored), margin=dict(l=190, r=20, t=70, b=40),
                                      yaxis=dict(autorange="reversed", automargin=False, tickfont=dict(size=12, color="#0b0b0b")),
                                      xaxis=dict(title=dict(text="Score points (out of 100)", font=dict(size=12, color="#52514e")),
                                                 range=[0, 100], dtick=20, gridcolor="#e1e0d9", tickfont=dict(color="#52514e")),
                                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, traceorder="normal", font=dict(size=11)),
                                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.35,
                                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                    st.plotly_chart(fig, width="stretch", theme=None)
                with g2:
                    st.markdown("#### Rank stability (500 weight variations)")
                    s2 = stab[stab.win_share_pct > 0]
                    fig2 = go.Figure(go.Bar(x=s2["win_share_pct"], y=s2["vendor_id"], orientation="h",
                                            marker_color="#2a78d6", text=[f"{v:.0f}%" for v in s2["win_share_pct"]],
                                            textposition="outside", hovertemplate="%{y}: #1 in %{x:.1f}% of runs<extra></extra>"))
                    fig2.update_layout(height=110 + 42 * max(len(s2), 1), margin=dict(l=10, r=30, t=10, b=40),
                                       xaxis=dict(range=[0, 115], title=dict(text="% of runs ranked #1", font=dict(size=12, color="#52514e")),
                                                  gridcolor="#e1e0d9", tickfont=dict(color="#52514e")),
                                       yaxis=dict(autorange="reversed", automargin=True, tickfont=dict(size=12, color="#0b0b0b")),
                                       plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", bargap=0.4,
                                       font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                    st.plotly_chart(fig2, width="stretch", theme=None)
                    st.caption("Below 70% means the #1 spot depends on small changes in weights - treat it as a close call.")

                    if len(scored) >= 2:
                        st.markdown("#### Top 3 profile")
                        cats = [CRITERIA[k]["label"] for k in CRITERIA]
                        fig3 = go.Figure()
                        for i, r in enumerate(scored.head(3).itertuples()):
                            vals = [getattr(r, f"norm_{k}") for k in CRITERIA]
                            fig3.add_scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], name=r.vendor_id,
                                                  line=dict(color=TOP3[i], width=2), fill="none")
                        fig3.update_layout(height=360, margin=dict(l=90, r=90, t=30, b=30),
                                           polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor="#e1e0d9"),
                                                      angularaxis=dict(tickfont=dict(size=10, color="#52514e"), gridcolor="#e1e0d9"),
                                                      bgcolor="rgba(0,0,0,0)"),
                                           legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                                           paper_bgcolor="rgba(0,0,0,0)",
                                           font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"))
                        st.plotly_chart(fig3, width="stretch", theme=None)
                        st.caption("1 = best in this list on that criterion (relative, not absolute).")

                st.markdown("#### Expert cross-check: the L1 rule")
                if l1["same"]:
                    st.success(f"The #1 vendor **{l1['top_id']}** is also the **L1** (lowest landed cost) vendor - model and rule of thumb agree.", icon="🤝")
                else:
                    st.info(f"Rule of thumb says buy from L1 vendor **{l1['l1_id']}**. The model prefers **{l1['top_id']}**, "
                            f"which costs **₹{l1['premium_per_kg']:.2f}/kg more ({l1['premium_pct']}%)** = "
                            f"**₹{l1['premium_total_rs']:,.0f}** extra on {ss.qty:g} MT. In return you get: "
                            + ("; ".join(l1["what_you_get"]) if l1["what_you_get"] else "no measurable advantage - reconsider the weights")
                            + ".", icon="⚖️")

                buf = io.StringIO()
                scored.drop(columns=[c for c in scored.columns if c.startswith("norm_")]).to_csv(buf, index=False)
                st.download_button("⬇ Download ranking (CSV)", buf.getvalue(), f"vendoriq_{ss.material}_ranking.csv", "text/csv")

            if not disq.empty:
                with st.expander(f"Knocked out by hard requirements ({len(disq)})", expanded=scored.empty):
                    st.dataframe(disq[["vendor_id", "vendor_name", "disqualified_because"]], hide_index=True, width="stretch")

            if not scored.empty:
                context = ai.build_context(scored, disq, req, w_pct, l1, stab, anonymise=anonymise)

id_to_name = dict(zip(vendors["vendor_id"], vendors["vendor_name"])) if data_ok else {}

# ============================================================ ③ AI RECOMMENDATION
with tab_ai:
    if not context:
        st.info("Complete tab ② first - the AI explains the ranking computed there.")
    else:
        top_id = scored.iloc[0]["vendor_id"]
        ctx_hash = ai.cache_key(PROMPT_VERSION, models, ai.to_json(context))
        st.markdown(f"Gemini will explain the ranking for **{req['material']}**, flag risks and suggest negotiation levers. "
                    "It receives only the table from tab ② " + ("(vendor names hidden)." if anonymise else "(vendor names included)."))

        c1, c2 = st.columns([1, 3])
        run = c1.button("✨ Generate AI recommendation", type="primary", width="stretch")
        c2.caption("Same inputs → same answer: results are cached, so a double-click or a refresh does not spend another API call.")

        if run:
            if ctx_hash in ss.recs:
                pass                                              # idempotent: already have it
            elif ctx_hash in shared_ai_cache():
                res = shared_ai_cache()[ctx_hash]
                ss.recs[ctx_hash] = {**res, "source": "cache"}
                log_event("recommendation", ai.AIResult(ok=True, source="cache", model=res["model"]))
            else:
                allowed, why = can_call_ai() if api_key else (False, "No API key configured.")
                res = ai.AIResult(ok=False, source="rule-based", error=why)
                if allowed:
                    with st.spinner("Gemini is reading the ranking…"):
                        res = ai.get_recommendation(api_key, context, models)
                    ss.ai_calls += 1
                    ss.last_call = time.time()
                log_event("recommendation", res)
                if res.ok:
                    checks, ratio = ai.validate_recommendation(res.parsed, context, top_id)
                    entry = {"parsed": res.parsed, "checks": checks, "source": "gemini", "model": res.model,
                             "latency_ms": res.latency_ms, "tokens": res.tokens_in + res.tokens_out, "error": ""}
                    hard_fail = any(not c["passed"] for c in checks
                                    if c["check"] in ("Recommended vendor exists in qualified list", "No invented vendor IDs"))
                    if hard_fail:
                        entry.update(parsed=rule_based_summary(scored, l1, stab, decision), source="rule-based",
                                     error="AI output failed a safety check (see checks) - rule-based summary shown instead.",
                                     rejected_ai=res.parsed)
                    else:
                        shared_ai_cache()[ctx_hash] = entry
                    ss.recs[ctx_hash] = entry
                else:
                    ss.recs[ctx_hash] = {"parsed": rule_based_summary(scored, l1, stab, decision), "checks": [],
                                         "source": "rule-based", "model": "-", "latency_ms": 0, "tokens": 0,
                                         "error": res.error}

        rec = ss.recs.get(ctx_hash)
        if rec is None:
            st.caption("Press the button to generate. If the AI is unavailable, a rule-based summary is shown instead - the app never breaks.")
        else:
            p = rec["parsed"]
            if rec["source"] == "rule-based":
                st.markdown(f'<span class="fb-badge">Rule-based fallback</span> &nbsp; <span class="small-note">{rec["error"]}</span>', unsafe_allow_html=True)
            else:
                extra = "served from cache" if rec["source"] == "cache" else f"{rec['latency_ms']/1000:.1f}s · {rec['tokens']} tokens"
                st.markdown(f'<span class="ai-badge">🤖 AI-generated · {rec["model"]} · {extra}</span>', unsafe_allow_html=True)

            st.markdown(f"### {ai.label_ids(p['headline'], id_to_name)}")
            conf = p.get("confidence", "")
            st.markdown(f"**Confidence:** {conf}  ·  **Score-model #1:** {top_id}  ·  **Decision rule:** {decision[0]}")
            if p.get("disagreement_note"):
                st.warning("AI concern about the #1 vendor: " + ai.label_ids(p["disagreement_note"], id_to_name), icon="🧐")

            a, b = st.columns(2, gap="large")
            with a:
                st.markdown("**Why this vendor**")
                for w in p.get("why", []):
                    st.markdown(f"- {ai.label_ids(w, id_to_name)}")
                st.markdown("**Negotiation levers**")
                for w in p.get("negotiation_levers", []):
                    st.markdown(f"- {ai.label_ids(w, id_to_name)}")
            with b:
                st.markdown("**Risks to watch**")
                for r in p.get("risks", []):
                    st.markdown(f"- **{r.get('vendor_id', '')}** - {ai.label_ids(r.get('risk', ''), id_to_name)}")
                st.markdown("**Versus the L1 rule**")
                st.markdown(ai.label_ids(p.get("l1_comparison", ""), id_to_name))
                if p.get("data_gaps"):
                    st.markdown("**Data gaps the AI noticed**")
                    for g in p["data_gaps"]:
                        st.markdown(f"- {g}")

            if rec["checks"]:
                n_ok = sum(c["passed"] for c in rec["checks"])
                with st.expander(f"🔍 AI output checks - {n_ok}/{len(rec['checks'])} passed", expanded=n_ok < len(rec["checks"])):
                    st.dataframe(pd.DataFrame([{**c, "passed": "✅" if c["passed"] else "⚠️"} for c in rec["checks"]]),
                                 hide_index=True, width="stretch")
                    if rec.get("rejected_ai"):
                        st.json(rec["rejected_ai"])
            with st.expander("Exactly what was sent to the AI (transparency)"):
                st.json(context, expanded=False)

            md = [f"# VendorIQ recommendation - {req['material']}", f"_Generated {datetime.now():%d %b %Y %H:%M} · source: {rec['source']} · model: {rec['model']}_", "",
                  f"**{p['headline']}**", "", f"Decision rule: {decision[0]} - {decision[1]}", "", "## Why"]
            md += [f"- {w}" for w in p.get("why", [])] + ["", "## Risks"] + [f"- {r['vendor_id']}: {r['risk']}" for r in p.get("risks", [])]
            md += ["", "## Negotiation levers"] + [f"- {w}" for w in p.get("negotiation_levers", [])]
            md += ["", "## L1 comparison", p.get("l1_comparison", ""), "", "## Ranking",
                   "| # | ID | Vendor | Score | Tier | Landed Rs/kg |", "|---|---|---|---|---|---|"]
            md += [f"| {r.rank} | {r.vendor_id} | {r.vendor_name} | {r.score:.1f} | {r.tier} | {r.landed_cost:.2f} |" for r in scored.itertuples()]
            md += ["", "_AI-assisted decision support. Verify before issuing a purchase order._"]
            st.download_button("⬇ Download recommendation report (.md)", "\n".join(md), "vendoriq_recommendation.md")

# ============================================================ ④ CHAT
with tab_chat:
    if not context:
        st.info("Complete tab ② first - the assistant answers questions about that ranking.")
    else:
        st.info("You are chatting with an **AI assistant (Google Gemini)**, not a person. It only knows the ranked table "
                f"for **{req['material']}** and remembers the last {ai.MAX_HISTORY_TURNS} messages. "
                "For decisions it can't support, it will refer you to the purchase committee.", icon="🤖")
        top_id = scored.iloc[0]["vendor_id"]
        suggestions = [f"Why is {top_id} ranked first?", "Compare the top 2 on cost and delivery",
                       "Which vendor if I need it in 4 days?", f"Draft a short RFQ email to {top_id}"]
        picked = st.pills("Try:", suggestions, key="chat_pill", label_visibility="collapsed")

        typed = ss.get("chat_box")
        question = typed or (picked if picked and ss.get("last_pill") != picked else None)
        if picked:
            ss.last_pill = picked
        if question:
            allowed, canned = ai.screen_user_message(question)
            if not allowed:
                answer, meta = canned, "Blocked before reaching the AI (input screening)."
                log_event("chat", ai.AIResult(ok=False, source="blocked", error="screened"), question[:40])
            elif not api_key:
                answer, meta = ai.offline_answer(question, scored), "Rule-based (AI offline)."
                log_event("chat", ai.AIResult(ok=False, source="rule-based", error="no key"))
            else:
                ok, why = can_call_ai()
                if not ok:
                    answer, meta = ai.offline_answer(question, scored), why
                else:
                    with st.spinner("Thinking…"):
                        res = ai.chat_reply(api_key, context, ss.chat, question, models)
                    ss.ai_calls += 1
                    ss.last_call = time.time()
                    log_event("chat", res)
                    if res.ok:
                        answer, meta = res.text, f"AI-generated · {res.model} · {res.latency_ms/1000:.1f}s"
                    else:
                        answer = ai.offline_answer(question, scored)
                        meta = f"AI unavailable ({res.error}) - rule-based answer shown."
            ss.chat += [{"role": "user", "content": question}, {"role": "assistant", "content": answer, "meta": meta}]

        for m in ss.chat:
            with st.chat_message(m["role"], avatar="🧑‍💼" if m["role"] == "user" else "🤖"):
                st.markdown(ai.label_ids(m["content"], id_to_name) if m["role"] != "user" else m["content"])
                if m.get("meta"):
                    st.caption(m["meta"])

        st.chat_input("Ask about the vendors, e.g. 'Why is V004 below V001?'", key="chat_box", max_chars=ai.MAX_CHAT_CHARS)
        if ss.chat and st.button("Clear chat"):
            ss.chat = []
            st.rerun()

# ============================================================ ⑤ ABOUT
with tab_about:
    st.markdown("### How VendorIQ works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; ranksep=0.25; nodesep=0.3; pad=0.1;
      node [shape=box, style="rounded,filled", fillcolor="#f4f7fb", color="#9ec5f4", fontname="Helvetica", fontsize=16, margin="0.15,0.08"];
      edge [color="#898781", arrowsize=0.7];
      A [label="Vendor CSV"]; B [label="Validate\\n& clean"]; C [label="Hard\\nfilters"]; D [label="Score, stability\\n& L1 check"];
      E [label="Gemini\\nexplains", fillcolor="#fff5e6", color="#f6d79a"]; F [label="Output\\nchecks"];
      G [label="Committee\\ndecides", fillcolor="#eaf6ea", color="#9fd39f"];
      A -> B -> C -> D -> E -> F -> G; D -> G [label="fallback", style=dashed, fontsize=12, fontcolor="#52514e"]; }
    """, width="stretch")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
**Division of labour**
- **Python** does every calculation: landed cost, filters, scores, ranks, stability, L1 premium.
- **Gemini** only *explains* those numbers in plain language and answers questions about them.
- **You** set the weights and make the decision.

**Scoring method** - each criterion is min-max scaled within the shortlist (1 = best), multiplied by
its weight, summed to 100. Landed cost = price + freight (₹4 per tonne-km). Risk points =
5 per complaint + 15 if imported + 10 if no ISO 9001.
""")
    with c2:
        st.markdown("""
**Known limits - do not use unsupervised for**
- Final vendor award or PO release (needs committee approval).
- Anything that needs *current* market prices - the model does not know them and is told not to guess.
- New vendors with no history - scores rely on past OTIF, QC and complaints.

**Privacy** - with anonymisation on, vendor names and cities never leave this app. The API key is
stored in Streamlit secrets, never in code.
""")
    st.markdown("### Audit log (this session)")
    if ss.audit:
        st.dataframe(pd.DataFrame(ss.audit), hide_index=True, width="stretch")
        st.download_button("⬇ Download audit log", pd.DataFrame(ss.audit).to_csv(index=False), "vendoriq_audit_log.csv")
    else:
        st.caption("No AI calls yet.")
