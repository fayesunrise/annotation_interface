"""
TK Human Annotation Interface
Human-LLM collaborative re-annotation for flagged cases (MEGAnno+ style)
2 coders | 101 flagged cases | 20 shared IRR cases | manual re-query
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import requests
from datetime import datetime
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TK Annotation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths — adjust to your machine ───────────────────────────────────────────
DATA_DIR        = Path(r"/Users/sunfeiyue/Documents/2026_WINTER/annotation_interface/annotation_results")
ANNOTATIONS_CSV = DATA_DIR / "annotation_annotated.csv"   # your Round 1 file
DECISIONS_CSV   = DATA_DIR / "human_decisions.csv"
STATE_JSON      = DATA_DIR / "coder_state.json"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL           = "gpt-4o-mini"

# ── Run columns ───────────────────────────────────────────────────────────────
LABEL_COLS = [f"tk_label_run{i}" for i in range(1, 5)]
CONF_COLS  = [f"tk_conf_run{i}"  for i in range(1, 5)]
RUN_NAMES  = {1: "OB researcher", 2: "Economist", 3: "Manager",
              4: "Paraphrased OB"}

OVERRIDE_REASONS = [
    "Select reason…",
    "Brand / company language — not role-specific",
    "Seniority trap — title inflates perceived TK",
    "Missing domain context — LLM lacks specialist knowledge",
    "Posting is ambiguous / incomplete",
    "LLM picked wrong evidence quote",
    "Other (explain in note)",
]

TK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
TK_LABELS = {0: "TK-0  Procedural", 1: "TK-1  Mixed", 2: "TK-2  High tacit"}


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & SETUP
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_annotations():
    df = pd.read_csv(ANNOTATIONS_CSV)
    for c in LABEL_COLS + CONF_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tk_modal"]      = df[LABEL_COLS].mode(axis=1)[0]
    df["label_spread"]  = df[LABEL_COLS].std(axis=1)
    df["conf_mean"]     = df[CONF_COLS].mean(axis=1)
    df["job_id"]        = df["job_id"].astype(str)
    return df


def build_queue(df):
    """Return flagged cases sorted by verifier_score descending."""
    flagged = df[df["human_review_flag"] == True].copy()
    return flagged.sort_values("verifier_score", ascending=False).reset_index(drop=True)


def get_irr_pool(queue_df, n=20):
    """Top-n cases by verifier_score — seen by both coders for IRR."""
    return set(queue_df.head(n)["job_id"].tolist())


def assign_cases(queue_df, irr_pool):
    """Split non-IRR cases between Coder A and Coder B."""
    exclusive = queue_df[~queue_df["job_id"].isin(irr_pool)].copy()
    half = len(exclusive) // 2
    return {
        "Coder A": irr_pool | set(exclusive.iloc[:half]["job_id"].tolist()),
        "Coder B": irr_pool | set(exclusive.iloc[half:]["job_id"].tolist()),
    }


def load_decisions():
    if DECISIONS_CSV.exists():
        d = pd.read_csv(DECISIONS_CSV, dtype={"job_id": str})
        return d
    cols = ["job_id", "coder_id", "human_label", "decision_type",
            "override_reason", "note", "requery_label", "requery_note",
            "timestamp"]
    return pd.DataFrame(columns=cols)


def save_decision(record: dict):
    decisions = load_decisions()
    # Remove any prior decision for this job_id + coder_id combo
    decisions = decisions[
        ~((decisions["job_id"] == record["job_id"]) &
          (decisions["coder_id"] == record["coder_id"]))
    ]
    new_row = pd.DataFrame([record])
    decisions = pd.concat([decisions, new_row], ignore_index=True)
    decisions.to_csv(DECISIONS_CSV, index=False)


def load_state():
    if STATE_JSON.exists():
        with open(STATE_JSON) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# RE-QUERY  (MEGAnno+ option B)
# ══════════════════════════════════════════════════════════════════════════════

def build_requery_prompt(description: str, human_note: str) -> str:
    """Inject human expert note into the original codebook prompt."""
    desc = description[:3200]
    return f"""You are an Organizational Behavior researcher specialising in tacit knowledge.

Tacit Knowledge (TK) is multidimensional (Leonard & Insch, 2005):
  1. COGNITIVE  — Institutional judgment and self-organisation beyond formal procedures.
  2. TECHNICAL  — Local tool mastery vs. big-picture institutional tasks.
  3. SOCIAL     — Relationship navigation and stakeholder interaction.

SCORING RULES:
  TK-0 (~15%): Procedural. A novice with a manual can do this.
  TK-1 (~35%): Mixed. Some judgment alongside structured workflow.
  TK-2 (~50%): Deep TK. A master is required. Cannot be codified.

IMPORTANT — a human expert has reviewed the previous LLM annotation for this posting
and provided the following correction note:

  "{human_note}"

Please reconsider your score in light of this expert feedback.

Respond ONLY as JSON:
{{"score": <0|1|2>, "confidence": <0-100>,
  "evidence": "<direct quote max 20 words>",
  "rationale": "<2-3 sentences ending with Dominant: X. Also present: Y. Absent: Z.>"}}

Job posting:
{desc}"""


def call_requery(description: str, human_note: str):
    prompt = build_requery_prompt(description, human_note)
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}",
               "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0,
               "max_tokens": 400,
               "response_format": {"type": "json_object"}}
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions",
                             headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        score = int(result.get("score", -1))
        if score not in [0, 1, 2]:
            return None, "Invalid score returned"
        return result, None
    except Exception as e:
        return None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════

def inject_css():
    st.markdown("""
    <style>
    /* Run card */
    .run-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        background: #fafafa;
    }
    .run-card-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 6px;
    }
    .score-badge {
        font-weight: 700;
        font-size: 15px;
        padding: 2px 10px;
        border-radius: 12px;
        color: white;
    }
    .score-0 { background: #4CAF50; }
    .score-1 { background: #FF9800; }
    .score-2 { background: #F44336; }
    .conf-bar-bg {
        background: #e8e8e8;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin-top: 4px;
    }
    .conf-bar-fill {
        height: 6px;
        border-radius: 4px;
        background: #5c85d6;
    }
    /* Job desc */
    .desc-box {
        background: #f5f7fa;
        border-left: 3px solid #5c85d6;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 13.5px;
        line-height: 1.65;
        max-height: 340px;
        overflow-y: auto;
    }
    /* Verifier bar */
    .verifier-bar-bg {
        background: #eee;
        border-radius: 4px;
        height: 10px;
        width: 100%;
    }
    .verifier-bar-fill {
        height: 10px;
        border-radius: 4px;
    }
    /* Status pills */
    .pill-done    { background:#d4edda; color:#155724; padding:2px 9px;
                    border-radius:10px; font-size:12px; font-weight:600; }
    .pill-pending { background:#fff3cd; color:#856404; padding:2px 9px;
                    border-radius:10px; font-size:12px; font-weight:600; }
    .pill-conflict{ background:#f8d7da; color:#721c24; padding:2px 9px;
                    border-radius:10px; font-size:12px; font-weight:600; }
    /* Requery result */
    .requery-box {
        border: 2px dashed #9c6ef8;
        border-radius: 8px;
        padding: 12px 14px;
        background: #f8f5ff;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT: Run cards (View B core)
# ══════════════════════════════════════════════════════════════════════════════

def render_run_card(run_num: int, row: pd.Series):
    label   = int(row[f"tk_label_run{run_num}"])   if not pd.isna(row[f"tk_label_run{run_num}"]) else None
    conf    = int(row[f"tk_conf_run{run_num}"])    if not pd.isna(row[f"tk_conf_run{run_num}"]) else None
    evid    = str(row[f"tk_evidence_run{run_num}"]) if not pd.isna(row[f"tk_evidence_run{run_num}"]) else "—"
    rat     = str(row[f"tk_rationale_run{run_num}"]) if not pd.isna(row[f"tk_rationale_run{run_num}"]) else "—"
    persona = RUN_NAMES.get(run_num, f"Run {run_num}")

    if label is None:
        return

    tk_label_str = TK_LABELS.get(label, str(label))
    st.markdown(f"""
    <div class="run-card">
      <div class="run-card-header">
        <span class="score-badge score-{label}">TK-{label}</span>
        <span style="font-weight:600;font-size:13px;">R{run_num} — {persona}</span>
        <span style="margin-left:auto;font-size:12px;color:#666;">conf {conf}</span>
      </div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf}%;"></div></div>
      <div style="margin-top:8px;font-size:12px;">
        <span style="color:#888;">Evidence: </span>
        <em>"{evid}"</em>
      </div>
      <div style="margin-top:4px;font-size:12px;color:#444;">{rat}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPONENT: Verifier score breakdown
# ══════════════════════════════════════════════════════════════════════════════

def render_verifier_breakdown(row: pd.Series):
    score = row["verifier_score"]
    pct   = min(int(score * 100 / 0.85 * 100), 100)   # normalise to max observed

    if score >= 0.7:
        color, label = "#F44336", "Very high instability"
    elif score >= 0.5:
        color, label = "#FF9800", "High instability"
    else:
        color, label = "#FF9800", "Moderate instability"

    spread = row["label_spread"]
    cf_mean = row["conf_mean"]

    st.markdown(f"""
    <div style="margin-bottom:6px;">
      <span style="font-size:12px;color:#666;">Verifier score: </span>
      <strong>{score:.3f}</strong>
      <span style="margin-left:8px;font-size:12px;color:{color};">{label}</span>
    </div>
    <div class="verifier-bar-bg">
      <div class="verifier-bar-fill" style="width:{pct}%;background:{color};"></div>
    </div>
    <div style="margin-top:6px;font-size:12px;color:#666;">
      Label spread: <strong>{spread:.2f}</strong> &nbsp;|&nbsp;
      Mean confidence: <strong>{cf_mean:.0f}</strong> &nbsp;|&nbsp;
      schaal_tk (benchmark): <strong>{row['schaal_tk']:.3f}</strong>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — login + queue overview
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(queue_df, decisions_df, assignments, coder_id):
    st.sidebar.title("TK Annotation")
    st.sidebar.markdown("---")

    # Progress
    my_cases   = assignments.get(coder_id, set())
    done_ids   = set(decisions_df[decisions_df["coder_id"] == coder_id]["job_id"].tolist())
    n_done     = len(done_ids & my_cases)
    n_total    = len(my_cases)
    pct        = int(n_done / n_total * 100) if n_total else 0

    st.sidebar.markdown(f"**Coder:** {coder_id}")
    st.sidebar.markdown(f"**Progress:** {n_done} / {n_total} ({pct}%)")
    st.sidebar.progress(pct / 100)

    st.sidebar.markdown("---")

    # Filters
    st.sidebar.markdown("**Filter queue**")
    occ_options = ["All occupations"] + sorted(queue_df["schaal_title"].unique().tolist())
    sel_occ = st.sidebar.selectbox("Occupation", occ_options, key="filter_occ")

    sen_options = ["All seniority", "senior", "mid", "junior"]
    sel_sen = st.sidebar.selectbox("Seniority", sen_options, key="filter_sen")

    status_options = ["All", "Pending", "Done", "IRR overlap"]
    sel_status = st.sidebar.selectbox("Status", status_options, key="filter_status")

    st.sidebar.markdown("---")

    # IRR conflict summary
    irr_ids = get_irr_pool(queue_df)
    irr_decisions = decisions_df[decisions_df["job_id"].isin(irr_ids)]
    if len(irr_decisions) > 0:
        irr_pivot = irr_decisions.pivot_table(
            index="job_id", columns="coder_id", values="human_label", aggfunc="first"
        )
        if "Coder A" in irr_pivot and "Coder B" in irr_pivot:
            conflicts = (irr_pivot["Coder A"] != irr_pivot["Coder B"]).sum()
            st.sidebar.markdown(f"**IRR conflicts:** {conflicts} / {len(irr_pivot)} shared cases")

    return sel_occ, sel_sen, sel_status, done_ids, my_cases


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — queue view (View A)
# ══════════════════════════════════════════════════════════════════════════════

def render_queue(queue_df, decisions_df, assignments, coder_id,
                 sel_occ, sel_sen, sel_status, done_ids, my_cases):

    irr_ids = get_irr_pool(queue_df)
    my_queue = queue_df[queue_df["job_id"].isin(my_cases)].copy()

    # Apply filters
    if sel_occ != "All occupations":
        my_queue = my_queue[my_queue["schaal_title"] == sel_occ]
    if sel_sen != "All seniority":
        my_queue = my_queue[my_queue["seniority"] == sel_sen]
    if sel_status == "Pending":
        my_queue = my_queue[~my_queue["job_id"].isin(done_ids)]
    elif sel_status == "Done":
        my_queue = my_queue[my_queue["job_id"].isin(done_ids)]
    elif sel_status == "IRR overlap":
        my_queue = my_queue[my_queue["job_id"].isin(irr_ids)]

    st.subheader(f"Queue — {len(my_queue)} cases shown")

    if len(my_queue) == 0:
        st.info("No cases match the current filter.")
        return

    for _, row in my_queue.iterrows():
        job_id  = row["job_id"]
        is_done = job_id in done_ids
        is_irr  = job_id in irr_ids

        # Status pill
        if is_done:
            pill = '<span class="pill-done">Done</span>'
        else:
            pill = '<span class="pill-pending">Pending</span>'
        if is_irr:
            pill += ' <span style="background:#cce5ff;color:#004085;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;">IRR</span>'

        modal_label = int(row["tk_modal"]) if not pd.isna(row["tk_modal"]) else "?"
        modal_color = TK_COLORS.get(modal_label, "#888")

        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""
            <div style="padding:8px 4px;border-bottom:1px solid #eee;">
              {pill}
              <span style="margin-left:8px;font-weight:600;">{row['job_title'][:60]}</span><br>
              <span style="font-size:12px;color:#666;">
                {row['schaal_title']} · {row['seniority']} ·
                modal <span style="color:{modal_color};font-weight:700;">TK-{modal_label}</span> ·
                verifier {row['verifier_score']:.3f}
              </span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Annotate →", key=f"open_{job_id}"):
                st.session_state["active_job_id"] = job_id
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — annotation card (View B + C)
# ══════════════════════════════════════════════════════════════════════════════

def render_annotation_card(row: pd.Series, decisions_df: pd.DataFrame, coder_id: str):
    job_id = row["job_id"]

    # Load prior decision if exists
    prior = decisions_df[
        (decisions_df["job_id"] == job_id) &
        (decisions_df["coder_id"] == coder_id)
    ]
    has_prior = len(prior) > 0
    prior_rec = prior.iloc[0].to_dict() if has_prior else {}

    # ── Header ────────────────────────────────────────────────────────────────
    col_back, col_title = st.columns([1, 7])
    with col_back:
        if st.button("← Back"):
            st.session_state["active_job_id"] = None
            st.session_state.pop("requery_result", None)
            st.rerun()
    with col_title:
        modal = int(row["tk_modal"]) if not pd.isna(row["tk_modal"]) else "?"
        irr_ids = get_irr_pool(
            pd.DataFrame([row])   # minimal — just used for badge
        )
        st.markdown(
            f"### {row['job_title'][:80]}"
            f"<br><span style='font-size:13px;color:#666;'>"
            f"{row['schaal_title']} &nbsp;·&nbsp; {row['seniority'].capitalize()} "
            f"&nbsp;·&nbsp; Modal: <strong style='color:{TK_COLORS.get(modal,'#888')};'>TK-{modal}</strong> "
            f"&nbsp;·&nbsp; Verifier: <strong>{row['verifier_score']:.3f}</strong>"
            f"</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Two-column layout: left = job + runs, right = decision ────────────────
    left, right = st.columns([3, 2], gap="large")

    # ── LEFT: Job description ─────────────────────────────────────────────────
    with left:
        with st.expander("Job description", expanded=True):
            st.markdown(
                f'<div class="desc-box">{row["description"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### LLM annotation runs")
        render_verifier_breakdown(row)
        st.markdown("")

        # Toggle show/hide LLM runs
        show_key = f"show_runs_{job_id}"
        if show_key not in st.session_state:
            st.session_state[show_key] = False

        btn_label = "▼ Hide LLM annotations" if st.session_state[show_key] else "▶ Show LLM annotations"
        if st.button(btn_label, key=f"toggle_runs_{job_id}"):
            st.session_state[show_key] = not st.session_state[show_key]
            st.rerun()

        if st.session_state[show_key]:
            n_runs = sum(1 for i in range(1, 5) if not pd.isna(row[f"tk_label_run{i}"]))
            for i in range(1, n_runs + 1):
                render_run_card(i, row)

        # Requery result panel (if triggered)
        if "requery_result" in st.session_state and st.session_state["requery_result"]:
            res = st.session_state["requery_result"]
            r_score = res.get("score", "?")
            r_conf  = res.get("confidence", "?")
            r_evid  = res.get("evidence", "")
            r_rat   = res.get("rationale", "")
            st.markdown(f"""
            <div class="requery-box">
              <div style="font-weight:700;margin-bottom:6px;">
                Re-query result &nbsp;
                <span class="score-badge score-{r_score}">TK-{r_score}</span>
                &nbsp; conf {r_conf}
              </div>
              <div style="font-size:12px;"><em>"{r_evid}"</em></div>
              <div style="font-size:12px;color:#444;margin-top:4px;">{r_rat}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── RIGHT: Decision panel (View C) ────────────────────────────────────────
    with right:
        st.markdown("#### Your decision")

        if has_prior:
            st.success(f"Previously saved: TK-{int(prior_rec['human_label'])} "
                       f"({prior_rec['decision_type']})")

        # Score selection
        default_score = int(prior_rec.get("human_label", modal)) if has_prior else modal
        score_choice = st.radio(
            "Assign TK label",
            options=[0, 1, 2],
            format_func=lambda x: TK_LABELS[x],
            index=default_score,
            horizontal=True,
            key=f"score_{job_id}",
        )

        # Decision type helper
        if score_choice == modal:
            decision_type = "accept_modal"
            st.markdown(
                "<span style='color:#4CAF50;font-size:13px;'>✓ Agrees with modal label</span>",
                unsafe_allow_html=True,
            )
        else:
            decision_type = "override"
            st.markdown(
                f"<span style='color:#F44336;font-size:13px;'>↗ Override: modal was TK-{modal}</span>",
                unsafe_allow_html=True,
            )

        # Override reason (shown for overrides)
        override_reason = ""
        if decision_type == "override":
            override_reason = st.selectbox(
                "Override reason",
                OVERRIDE_REASONS,
                key=f"reason_{job_id}",
                index=OVERRIDE_REASONS.index(prior_rec.get("override_reason", OVERRIDE_REASONS[0]))
                if has_prior and prior_rec.get("override_reason") in OVERRIDE_REASONS else 0,
            )

        # Note field
        note = st.text_area(
            "Note (required for override, optional otherwise)",
            value=prior_rec.get("note", "") if has_prior else "",
            height=100,
            placeholder="e.g. 'The evidence quote is from a company boilerplate section, "
                        "not describing what the role actually does.'",
            key=f"note_{job_id}",
        )

        st.markdown("---")

        # Re-query button
        st.markdown("**Re-query LLM with your note**")
        st.caption("Injects your note into the prompt and gets a revised LLM score. "
                   "You can then accept it or ignore it.")

        requery_note = st.text_input(
            "Note to inject (leave blank to use note above)",
            value="",
            placeholder="Or type a specific correction here…",
            key=f"rq_note_{job_id}",
        )

        col_rq, col_rq_status = st.columns([2, 3])
        with col_rq:
            if st.button("Re-query LLM →", key=f"requery_{job_id}"):
                inject = requery_note.strip() or note.strip()
                if not inject:
                    st.warning("Write a note first before re-querying.")
                else:
                    with st.spinner("Querying LLM…"):
                        result, err = call_requery(row["description"], inject)
                    if err:
                        st.error(f"API error: {err}")
                    else:
                        st.session_state["requery_result"] = result
                        st.rerun()

        # If requery done, offer to accept it
        requery_label = None
        requery_note_saved = ""
        if "requery_result" in st.session_state and st.session_state["requery_result"]:
            res = st.session_state["requery_result"]
            with col_rq_status:
                st.markdown(
                    f"<span style='color:#9c6ef8;font-size:13px;'>"
                    f"Re-query: TK-{res['score']} (conf {res['confidence']})</span>",
                    unsafe_allow_html=True,
                )
            if st.checkbox("Accept re-query score as my label", key=f"accept_rq_{job_id}"):
                score_choice  = res["score"]
                decision_type = ("requery_accept" if res["score"] == modal
                                 else "requery_override")
                requery_label = res["score"]
                requery_note_saved = requery_note.strip() or note.strip()

        st.markdown("---")

        # Submit
        can_submit = True
        if decision_type == "override" and override_reason == OVERRIDE_REASONS[0]:
            st.warning("Please select an override reason.")
            can_submit = False
        if decision_type == "override" and not note.strip():
            st.warning("Please add a note explaining the override.")
            can_submit = False

        if st.button("✓ Save & next case", type="primary",
                     disabled=not can_submit, key=f"submit_{job_id}"):
            record = {
                "job_id":           job_id,
                "coder_id":         coder_id,
                "human_label":      score_choice,
                "decision_type":    decision_type,
                "override_reason":  override_reason,
                "note":             note.strip(),
                "requery_label":    requery_label,
                "requery_note":     requery_note_saved,
                "timestamp":        datetime.now().isoformat(),
            }
            save_decision(record)
            st.session_state.pop("requery_result", None)

            # Advance to next pending case
            df_all    = load_annotations()
            queue_df  = build_queue(df_all)
            decisions = load_decisions()
            assignments = assign_cases(queue_df, get_irr_pool(queue_df))
            my_cases  = assignments.get(coder_id, set())
            done_ids  = set(decisions[decisions["coder_id"] == coder_id]["job_id"].tolist())
            my_queue  = queue_df[queue_df["job_id"].isin(my_cases)]
            pending   = my_queue[~my_queue["job_id"].isin(done_ids)]

            if len(pending) > 0:
                st.session_state["active_job_id"] = pending.iloc[0]["job_id"]
            else:
                st.session_state["active_job_id"] = None
                st.success("All your cases are done!")
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# IRR DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def render_irr_dashboard(queue_df, decisions_df):
    st.subheader("IRR Dashboard — shared cases (top 20 by verifier score)")

    irr_ids = get_irr_pool(queue_df)
    irr_dec = decisions_df[decisions_df["job_id"].isin(irr_ids)]

    if len(irr_dec) == 0:
        st.info("No IRR decisions saved yet.")
        return

    pivot = irr_dec.pivot_table(
        index="job_id", columns="coder_id", values="human_label", aggfunc="first"
    ).reset_index()

    if "Coder A" in pivot.columns and "Coder B" in pivot.columns:
        pivot["agreement"] = pivot["Coder A"] == pivot["Coder B"]
        pivot = pivot.merge(
            queue_df[["job_id", "job_title", "schaal_title",
                      "seniority", "tk_modal", "verifier_score"]],
            on="job_id", how="left",
        )

        n_agree   = pivot["agreement"].sum()
        n_total   = pivot["agreement"].notna().sum()

        # Cohen's kappa
        if n_total >= 2:
            from sklearn.metrics import cohen_kappa_score
            try:
                kappa = cohen_kappa_score(
                    pivot["Coder A"].dropna(),
                    pivot["Coder B"].dropna(),
                )
                st.metric("Cohen's κ", f"{kappa:.3f}",
                          help="0.6–0.8 = substantial agreement; >0.8 = near perfect")
            except Exception:
                pass

        col1, col2, col3 = st.columns(3)
        col1.metric("Cases annotated by both", n_total)
        col2.metric("Agreement", f"{n_agree}/{n_total}")
        col3.metric("Conflict", f"{n_total - n_agree}/{n_total}")

        st.markdown("#### Case-by-case breakdown")
        display = pivot[["job_title", "schaal_title", "seniority",
                         "tk_modal", "Coder A", "Coder B",
                         "agreement", "verifier_score"]].copy()
        display.columns = ["Job title", "Occupation", "Seniority",
                           "Modal", "Coder A", "Coder B",
                           "Agree", "Verifier"]
        display["Agree"] = display["Agree"].map({True: "✓", False: "✗"})
        st.dataframe(display, use_container_width=True, height=420)
    else:
        st.info("Waiting for both coders to annotate shared cases.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    inject_css()

    # ── Session defaults ──────────────────────────────────────────────────────
    if "coder_id" not in st.session_state:
        st.session_state["coder_id"] = None
    if "active_job_id" not in st.session_state:
        st.session_state["active_job_id"] = None

    # ── Login screen ──────────────────────────────────────────────────────────
    if st.session_state["coder_id"] is None:
        st.title("TK Annotation Interface")
        st.markdown("#### Select your coder ID to begin")
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Coder A", use_container_width=True, type="primary"):
                st.session_state["coder_id"] = "Coder A"
                st.rerun()
        with col2:
            if st.button("Coder B", use_container_width=True):
                st.session_state["coder_id"] = "Coder B"
                st.rerun()
        st.markdown("---")
        st.caption("20 shared IRR cases · Coder A: 61 cases · Coder B: 60 cases")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    df          = load_annotations()
    queue_df    = build_queue(df)
    irr_pool    = get_irr_pool(queue_df)
    assignments = assign_cases(queue_df, irr_pool)
    decisions   = load_decisions()
    coder_id    = st.session_state["coder_id"]

    # ── Sidebar ───────────────────────────────────────────────────────────────
    sel_occ, sel_sen, sel_status, done_ids, my_cases = render_sidebar(
        queue_df, decisions, assignments, coder_id
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_queue, tab_irr, tab_export = st.tabs(["Queue", "IRR Dashboard", "Export"])

    with tab_queue:
        if st.session_state["active_job_id"] is not None:
            job_id = st.session_state["active_job_id"]
            row    = df[df["job_id"] == job_id].iloc[0]
            render_annotation_card(row, decisions, coder_id)
        else:
            render_queue(queue_df, decisions, assignments, coder_id,
                         sel_occ, sel_sen, sel_status, done_ids, my_cases)

    with tab_irr:
        render_irr_dashboard(queue_df, decisions)

    with tab_export:
        st.subheader("Export final labels")
        st.markdown("""
        Merges human decisions back into the full annotation file.
        - Cases with a `human_label` → use human label as final
        - Remaining cases → use `tk_modal` as final
        """)

        if st.button("Generate final dataset"):
            final = df.copy()
            dec_merged = decisions.groupby("job_id").apply(
                lambda g: g.sort_values("timestamp").iloc[-1]
            ).reset_index(drop=True)

            dec_map = dec_merged.set_index("job_id")["human_label"].to_dict()
            type_map = dec_merged.set_index("job_id")["decision_type"].to_dict()

            final["human_label"]    = final["job_id"].map(dec_map)
            final["decision_type"]  = final["job_id"].map(type_map)
            final["tk_final_label"] = final.apply(
                lambda r: r["human_label"] if not pd.isna(r.get("human_label")) else r["tk_modal"],
                axis=1,
            )
            final["label_source"] = final["human_label"].apply(
                lambda x: "human" if not pd.isna(x) else "llm_modal"
            )

            out_path = DATA_DIR / "tk_final_labels.csv"
            final.to_csv(out_path, index=False, encoding="utf-8-sig")
            st.success(f"Saved: {out_path}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Human-labelled", final["label_source"].eq("human").sum())
            col2.metric("LLM modal", final["label_source"].eq("llm_modal").sum())
            col3.metric("Total", len(final))

            st.markdown("**Final label distribution:**")
            st.bar_chart(final["tk_final_label"].value_counts().sort_index())

            csv_bytes = final.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download tk_final_labels.csv",
                data=csv_bytes,
                file_name="tk_final_labels.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()