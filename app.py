

import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime
from pathlib import Path
import os

# dotenv: only used locally, silently skipped on Cloud
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(
    page_title="TK Annotation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths — always relative to this file, works everywhere
DATA_DIR        = Path(__file__).parent
ANNOTATIONS_CSV = DATA_DIR / "annotation_annotated.csv"

def get_openai_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.getenv("OPENAI_API_KEY", "")

MODEL = "gpt-4o-mini"

LABEL_COLS = [f"tk_label_run{i}" for i in range(1, 5)]
CONF_COLS  = [f"tk_conf_run{i}"  for i in range(1, 5)]
RUN_NAMES  = {1: "OB researcher", 2: "Economist", 3: "Manager", 4: "Paraphrased OB"}

OVERRIDE_REASONS = [
    "Select reason\u2026",
    "Brand / company language \u2014 not role-specific",
    "Seniority trap \u2014 title inflates perceived TK",
    "Missing domain context \u2014 LLM lacks specialist knowledge",
    "Posting is ambiguous / incomplete",
    "LLM picked wrong evidence quote",
    "Other (explain in note)",
]

TK_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
TK_LABELS = {0: "TK-0  Procedural", 1: "TK-1  Mixed", 2: "TK-2  High tacit"}


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_annotations():
    if not ANNOTATIONS_CSV.exists():
        st.error(f"annotation_annotated.csv not found in {DATA_DIR}. "
                 "Make sure it is committed to the same folder as app.py.")
        st.stop()
    df = pd.read_csv(ANNOTATIONS_CSV)
    for c in LABEL_COLS + CONF_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tk_modal"]     = df[LABEL_COLS].mode(axis=1)[0]
    df["label_spread"] = df[LABEL_COLS].std(axis=1)
    df["conf_mean"]    = df[CONF_COLS].mean(axis=1)
    df["job_id"]       = df["job_id"].astype(str)
    return df


def build_queue(df):
    return df[df["human_review_flag"] == True].copy().sort_values(
        "verifier_score", ascending=False).reset_index(drop=True)


def get_irr_pool(queue_df, n=20):
    return set(queue_df.head(n)["job_id"].tolist())


def assign_cases(queue_df, irr_pool):
    exclusive = queue_df[~queue_df["job_id"].isin(irr_pool)].copy()
    half = len(exclusive) // 2
    return {
        "Coder A": irr_pool | set(exclusive.iloc[:half]["job_id"].tolist()),
        "Coder B": irr_pool | set(exclusive.iloc[half:]["job_id"].tolist()),
    }


# ── Decisions: session_state only (Cloud filesystem is read-only) ─────────────
DECISION_COLS = ["job_id", "coder_id", "human_label", "decision_type",
                 "override_reason", "note", "requery_label", "requery_note", "timestamp"]

def get_decisions() -> pd.DataFrame:
    if "decisions_store" not in st.session_state:
        st.session_state["decisions_store"] = pd.DataFrame(columns=DECISION_COLS)
    return st.session_state["decisions_store"]


def save_decision(record: dict):
    decisions = get_decisions()
    decisions = decisions[
        ~((decisions["job_id"] == record["job_id"]) &
          (decisions["coder_id"] == record["coder_id"]))
    ]
    st.session_state["decisions_store"] = pd.concat(
        [decisions, pd.DataFrame([record])], ignore_index=True)


# ── Re-query ──────────────────────────────────────────────────────────────────
def build_requery_prompt(description: str, human_note: str) -> str:
    return f"""You are an Organizational Behavior researcher specialising in tacit knowledge.

Tacit Knowledge (TK) is multidimensional (Leonard & Insch, 2005):
  1. COGNITIVE  - Institutional judgment and self-organisation beyond formal procedures.
  2. TECHNICAL  - Local tool mastery vs. big-picture institutional tasks.
  3. SOCIAL     - Relationship navigation and stakeholder interaction.

SCORING RULES:
  TK-0 (~15%): Procedural. A novice with a manual can do this.
  TK-1 (~35%): Mixed. Some judgment alongside structured workflow.
  TK-2 (~50%): Deep TK. A master is required. Cannot be codified.

IMPORTANT - a human expert has reviewed the previous LLM annotation and provided:
  "{human_note}"

Please reconsider your score in light of this expert feedback.

Respond ONLY as JSON:
{{"score": <0|1|2>, "confidence": <0-100>,
  "evidence": "<direct quote max 20 words>",
  "rationale": "<2-3 sentences ending with Dominant: X. Also present: Y. Absent: Z.>"}}

Job posting:
{description[:3200]}"""


def call_requery(description: str, human_note: str):
    api_key = get_openai_key()
    if not api_key:
        return None, "No OpenAI API key. Add OPENAI_API_KEY to Streamlit secrets."
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": build_requery_prompt(description, human_note)}],
               "temperature": 0.0, "max_tokens": 400,
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


# ── CSS ───────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown("""<style>
    .run-card { border:1px solid #e0e0e0; border-radius:8px; padding:12px 14px; margin-bottom:8px; background:#fafafa; }
    .run-card-header { display:flex; align-items:center; gap:10px; margin-bottom:6px; }
    .score-badge { font-weight:700; font-size:15px; padding:2px 10px; border-radius:12px; color:white; }
    .score-0 { background:#4CAF50; } .score-1 { background:#FF9800; } .score-2 { background:#F44336; }
    .conf-bar-bg  { background:#e8e8e8; border-radius:4px; height:6px; width:100%; margin-top:4px; }
    .conf-bar-fill { height:6px; border-radius:4px; background:#5c85d6; }
    .desc-box { background:#f5f7fa; border-left:3px solid #5c85d6; padding:12px 16px;
                border-radius:0 6px 6px 0; font-size:13.5px; line-height:1.65;
                max-height:340px; overflow-y:auto; }
    .verifier-bar-bg  { background:#eee; border-radius:4px; height:10px; width:100%; }
    .verifier-bar-fill { height:10px; border-radius:4px; }
    .pill-done    { background:#d4edda; color:#155724; padding:2px 9px; border-radius:10px; font-size:12px; font-weight:600; }
    .pill-pending { background:#fff3cd; color:#856404; padding:2px 9px; border-radius:10px; font-size:12px; font-weight:600; }
    .requery-box  { border:2px dashed #9c6ef8; border-radius:8px; padding:12px 14px; background:#f8f5ff; margin-top:8px; }
    </style>""", unsafe_allow_html=True)


# ── Components ────────────────────────────────────────────────────────────────
def render_run_card(run_num: int, row: pd.Series):
    label = int(row[f"tk_label_run{run_num}"]) if not pd.isna(row[f"tk_label_run{run_num}"]) else None
    conf  = int(row[f"tk_conf_run{run_num}"])  if not pd.isna(row[f"tk_conf_run{run_num}"]) else None
    evid  = str(row.get(f"tk_evidence_run{run_num}", "")) or "—"
    rat   = str(row.get(f"tk_rationale_run{run_num}", "")) or "—"
    if label is None:
        return
    st.markdown(f"""<div class="run-card">
      <div class="run-card-header">
        <span class="score-badge score-{label}">TK-{label}</span>
        <span style="font-weight:600;font-size:13px;">R{run_num} — {RUN_NAMES.get(run_num,f'Run {run_num}')}</span>
        <span style="margin-left:auto;font-size:12px;color:#666;">conf {conf}</span>
      </div>
      <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{conf}%;"></div></div>
      <div style="margin-top:8px;font-size:12px;"><span style="color:#888;">Evidence: </span><em>"{evid}"</em></div>
      <div style="margin-top:4px;font-size:12px;color:#444;">{rat}</div>
    </div>""", unsafe_allow_html=True)


def render_verifier_breakdown(row: pd.Series):
    score = row["verifier_score"]
    pct   = min(int(score * 100 / 0.85 * 100), 100)
    color = "#F44336" if score >= 0.7 else "#FF9800"
    lbl   = "Very high instability" if score >= 0.7 else "High instability" if score >= 0.5 else "Moderate instability"
    st.markdown(f"""<div style="margin-bottom:6px;">
      <span style="font-size:12px;color:#666;">Verifier score: </span><strong>{score:.3f}</strong>
      <span style="margin-left:8px;font-size:12px;color:{color};">{lbl}</span>
    </div>
    <div class="verifier-bar-bg"><div class="verifier-bar-fill" style="width:{pct}%;background:{color};"></div></div>
    <div style="margin-top:6px;font-size:12px;color:#666;">
      Label spread: <strong>{row['label_spread']:.2f}</strong> &nbsp;|&nbsp;
      Mean confidence: <strong>{row['conf_mean']:.0f}</strong> &nbsp;|&nbsp;
      schaal_tk: <strong>{row['schaal_tk']:.3f}</strong>
    </div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar(queue_df, decisions_df, assignments, coder_id):
    st.sidebar.title("TK Annotation")
    st.sidebar.markdown("---")
    my_cases = assignments.get(coder_id, set())
    done_ids = set(decisions_df[decisions_df["coder_id"] == coder_id]["job_id"].tolist())
    n_done   = len(done_ids & my_cases)
    n_total  = len(my_cases)
    pct      = int(n_done / n_total * 100) if n_total else 0
    st.sidebar.markdown(f"**Coder:** {coder_id}")
    st.sidebar.markdown(f"**Progress:** {n_done} / {n_total} ({pct}%)")
    st.sidebar.progress(pct / 100)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filter queue**")
    sel_occ    = st.sidebar.selectbox("Occupation", ["All occupations"] + sorted(queue_df["schaal_title"].unique().tolist()), key="filter_occ")
    sel_sen    = st.sidebar.selectbox("Seniority",  ["All seniority","senior","mid","junior"], key="filter_sen")
    sel_status = st.sidebar.selectbox("Status",     ["All","Pending","Done","IRR overlap"], key="filter_status")
    st.sidebar.markdown("---")
    irr_ids = get_irr_pool(queue_df)
    irr_dec = decisions_df[decisions_df["job_id"].isin(irr_ids)]
    if len(irr_dec) > 0:
        irr_pivot = irr_dec.pivot_table(index="job_id", columns="coder_id", values="human_label", aggfunc="first")
        if "Coder A" in irr_pivot and "Coder B" in irr_pivot:
            conflicts = (irr_pivot["Coder A"] != irr_pivot["Coder B"]).sum()
            st.sidebar.markdown(f"**IRR conflicts:** {conflicts} / {len(irr_pivot)} shared cases")
    if len(decisions_df) > 0:
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Save your work**")
        st.sidebar.caption("Download decisions to preserve annotations across sessions.")
        st.sidebar.download_button(
            "⬇ Download decisions CSV",
            data=decisions_df.to_csv(index=False).encode("utf-8"),
            file_name=f"decisions_{coder_id.replace(' ','_')}.csv",
            mime="text/csv",
        )
    return sel_occ, sel_sen, sel_status, done_ids, my_cases


# ── Queue view ────────────────────────────────────────────────────────────────
def render_queue(queue_df, decisions_df, assignments, coder_id,
                 sel_occ, sel_sen, sel_status, done_ids, my_cases):
    irr_ids  = get_irr_pool(queue_df)
    my_queue = queue_df[queue_df["job_id"].isin(my_cases)].copy()
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
        pill    = '<span class="pill-done">Done</span>' if job_id in done_ids else '<span class="pill-pending">Pending</span>'
        if job_id in irr_ids:
            pill += ' <span style="background:#cce5ff;color:#004085;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:600;">IRR</span>'
        modal_label = int(row["tk_modal"]) if not pd.isna(row["tk_modal"]) else "?"
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"""<div style="padding:8px 4px;border-bottom:1px solid #eee;">
              {pill} <span style="margin-left:8px;font-weight:600;">{row['job_title'][:60]}</span><br>
              <span style="font-size:12px;color:#666;">
                {row['schaal_title']} · {row['seniority']} ·
                modal <span style="color:{TK_COLORS.get(modal_label,'#888')};font-weight:700;">TK-{modal_label}</span> ·
                verifier {row['verifier_score']:.3f}
              </span></div>""", unsafe_allow_html=True)
        with col2:
            if st.button("Annotate →", key=f"open_{job_id}"):
                st.session_state["active_job_id"] = job_id
                st.rerun()


# ── Annotation card ───────────────────────────────────────────────────────────
def render_annotation_card(row: pd.Series, decisions_df: pd.DataFrame, coder_id: str):
    job_id    = row["job_id"]
    prior     = decisions_df[(decisions_df["job_id"] == job_id) & (decisions_df["coder_id"] == coder_id)]
    has_prior = len(prior) > 0
    prior_rec = prior.iloc[0].to_dict() if has_prior else {}

    col_back, col_title = st.columns([1, 7])
    with col_back:
        if st.button("← Back"):
            st.session_state["active_job_id"] = None
            st.session_state.pop("requery_result", None)
            st.rerun()
    with col_title:
        modal = int(row["tk_modal"]) if not pd.isna(row["tk_modal"]) else "?"
        st.markdown(
            f"### {row['job_title'][:80]}"
            f"<br><span style='font-size:13px;color:#666;'>"
            f"{row['schaal_title']} &nbsp;·&nbsp; {row['seniority'].capitalize()} "
            f"&nbsp;·&nbsp; Modal: <strong style='color:{TK_COLORS.get(modal,'#888')};'>TK-{modal}</strong> "
            f"&nbsp;·&nbsp; Verifier: <strong>{row['verifier_score']:.3f}</strong></span>",
            unsafe_allow_html=True)
    st.markdown("---")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.expander("Job description", expanded=True):
            st.markdown(f'<div class="desc-box">{row["description"]}</div>', unsafe_allow_html=True)
        st.markdown("#### LLM annotation runs")
        render_verifier_breakdown(row)
        st.markdown("")
        show_key = f"show_runs_{job_id}"
        if show_key not in st.session_state:
            st.session_state[show_key] = False
        if st.button("▼ Hide LLM annotations" if st.session_state[show_key] else "▶ Show LLM annotations",
                     key=f"toggle_runs_{job_id}"):
            st.session_state[show_key] = not st.session_state[show_key]
            st.rerun()
        if st.session_state[show_key]:
            for i in range(1, 5):
                if not pd.isna(row[f"tk_label_run{i}"]):
                    render_run_card(i, row)
        if st.session_state.get("requery_result"):
            res = st.session_state["requery_result"]
            st.markdown(f"""<div class="requery-box">
              <div style="font-weight:700;margin-bottom:6px;">Re-query result &nbsp;
                <span class="score-badge score-{res.get('score','?')}">TK-{res.get('score','?')}</span>
                &nbsp; conf {res.get('confidence','?')}</div>
              <div style="font-size:12px;"><em>"{res.get('evidence','')}"</em></div>
              <div style="font-size:12px;color:#444;margin-top:4px;">{res.get('rationale','')}</div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown("#### Your decision")
        if has_prior:
            st.success(f"Previously saved: TK-{int(prior_rec['human_label'])} ({prior_rec['decision_type']})")
        default_score = int(prior_rec.get("human_label", modal)) if has_prior else modal
        score_choice  = st.radio("Assign TK label", options=[0,1,2],
                                  format_func=lambda x: TK_LABELS[x],
                                  index=default_score, horizontal=True, key=f"score_{job_id}")
        if score_choice == modal:
            decision_type = "accept_modal"
            st.markdown("<span style='color:#4CAF50;font-size:13px;'>✓ Agrees with modal label</span>", unsafe_allow_html=True)
        else:
            decision_type = "override"
            st.markdown(f"<span style='color:#F44336;font-size:13px;'>↗ Override: modal was TK-{modal}</span>", unsafe_allow_html=True)
        override_reason = ""
        if decision_type == "override":
            default_r = (OVERRIDE_REASONS.index(prior_rec.get("override_reason", OVERRIDE_REASONS[0]))
                         if has_prior and prior_rec.get("override_reason") in OVERRIDE_REASONS else 0)
            override_reason = st.selectbox("Override reason", OVERRIDE_REASONS, index=default_r, key=f"reason_{job_id}")
        note = st.text_area("Note (required for override, optional otherwise)",
                            value=prior_rec.get("note","") if has_prior else "",
                            height=100, key=f"note_{job_id}",
                            placeholder="e.g. 'Evidence quote is from boilerplate, not the actual role.'")
        st.markdown("---")
        st.markdown("**Re-query LLM with your note**")
        st.caption("Injects your note into the prompt and gets a revised LLM score.")
        requery_note = st.text_input("Note to inject (leave blank to use note above)",
                                     value="", key=f"rq_note_{job_id}")
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
        requery_label = None
        requery_note_saved = ""
        if st.session_state.get("requery_result"):
            res = st.session_state["requery_result"]
            with col_rq_status:
                st.markdown(f"<span style='color:#9c6ef8;font-size:13px;'>Re-query: TK-{res['score']} (conf {res['confidence']})</span>", unsafe_allow_html=True)
            if st.checkbox("Accept re-query score as my label", key=f"accept_rq_{job_id}"):
                score_choice       = res["score"]
                decision_type      = "requery_accept" if res["score"] == modal else "requery_override"
                requery_label      = res["score"]
                requery_note_saved = requery_note.strip() or note.strip()
        st.markdown("---")
        can_submit = True
        if decision_type == "override" and override_reason == OVERRIDE_REASONS[0]:
            st.warning("Please select an override reason.")
            can_submit = False
        if decision_type == "override" and not note.strip():
            st.warning("Please add a note explaining the override.")
            can_submit = False
        if st.button("✓ Save & next case", type="primary", disabled=not can_submit, key=f"submit_{job_id}"):
            save_decision({"job_id": job_id, "coder_id": coder_id,
                           "human_label": score_choice, "decision_type": decision_type,
                           "override_reason": override_reason, "note": note.strip(),
                           "requery_label": requery_label, "requery_note": requery_note_saved,
                           "timestamp": datetime.now().isoformat()})
            st.session_state.pop("requery_result", None)
            df_all  = load_annotations()
            q       = build_queue(df_all)
            dec     = get_decisions()
            asgn    = assign_cases(q, get_irr_pool(q))
            done_c  = set(dec[dec["coder_id"] == coder_id]["job_id"].tolist())
            pending = q[q["job_id"].isin(asgn.get(coder_id, set())) & ~q["job_id"].isin(done_c)]
            st.session_state["active_job_id"] = pending.iloc[0]["job_id"] if len(pending) > 0 else None
            if len(pending) == 0:
                st.success("All your cases are done!")
            st.rerun()


# ── IRR Dashboard ─────────────────────────────────────────────────────────────
def render_irr_dashboard(queue_df, decisions_df):
    st.subheader("IRR Dashboard — shared cases (top 20 by verifier score)")
    irr_ids = get_irr_pool(queue_df)
    irr_dec = decisions_df[decisions_df["job_id"].isin(irr_ids)]
    if len(irr_dec) == 0:
        st.info("No IRR decisions saved yet.")
        return
    pivot = irr_dec.pivot_table(index="job_id", columns="coder_id", values="human_label", aggfunc="first").reset_index()
    if "Coder A" in pivot.columns and "Coder B" in pivot.columns:
        pivot["agreement"] = pivot["Coder A"] == pivot["Coder B"]
        pivot = pivot.merge(queue_df[["job_id","job_title","schaal_title","seniority","tk_modal","verifier_score"]], on="job_id", how="left")
        n_agree = pivot["agreement"].sum()
        n_total = pivot["agreement"].notna().sum()
        if n_total >= 2:
            try:
                from sklearn.metrics import cohen_kappa_score
                kappa = cohen_kappa_score(pivot["Coder A"].dropna(), pivot["Coder B"].dropna())
                st.metric("Cohen's κ", f"{kappa:.3f}", help="0.6–0.8 = substantial; >0.8 = near perfect")
            except Exception:
                pass
        c1, c2, c3 = st.columns(3)
        c1.metric("Both annotated", n_total)
        c2.metric("Agreement", f"{n_agree}/{n_total}")
        c3.metric("Conflict",  f"{n_total-n_agree}/{n_total}")
        display = pivot[["job_title","schaal_title","seniority","tk_modal","Coder A","Coder B","agreement","verifier_score"]].copy()
        display.columns = ["Job title","Occupation","Seniority","Modal","Coder A","Coder B","Agree","Verifier"]
        display["Agree"] = display["Agree"].map({True: "✓", False: "✗"})
        st.dataframe(display, use_container_width=True, height=420)
    else:
        st.info("Waiting for both coders to annotate shared cases.")


# ── Export ────────────────────────────────────────────────────────────────────
def render_export(df, decisions_df):
    st.subheader("Export final labels")
    st.markdown("""Merges human decisions into the full annotation file.
- Cases with `human_label` → use human label as final
- Remaining → use `tk_modal` as final""")
    st.info("⚠️ Decisions are stored in your browser session. Download the CSV before closing the tab.", icon="ℹ️")
    if len(decisions_df) == 0:
        st.warning("No decisions saved yet.")
        return
    if st.button("Generate final dataset"):
        final = df.copy()
        dec_merged = decisions_df.sort_values("timestamp").groupby("job_id").last().reset_index()
        final["human_label"]    = final["job_id"].map(dec_merged.set_index("job_id")["human_label"])
        final["decision_type"]  = final["job_id"].map(dec_merged.set_index("job_id")["decision_type"])
        final["tk_final_label"] = final.apply(
            lambda r: r["human_label"] if not pd.isna(r.get("human_label")) else r["tk_modal"], axis=1)
        final["label_source"] = final["human_label"].apply(
            lambda x: "human" if not pd.isna(x) else "llm_modal")
        c1, c2, c3 = st.columns(3)
        c1.metric("Human-labelled", final["label_source"].eq("human").sum())
        c2.metric("LLM modal",      final["label_source"].eq("llm_modal").sum())
        c3.metric("Total",          len(final))
        st.markdown("**Final label distribution:**")
        st.bar_chart(final["tk_final_label"].value_counts().sort_index())
        st.download_button("⬇ Download tk_final_labels.csv",
                           data=final.to_csv(index=False).encode("utf-8"),
                           file_name="tk_final_labels.csv", mime="text/csv")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    inject_css()
    if "coder_id"     not in st.session_state: st.session_state["coder_id"]     = None
    if "active_job_id" not in st.session_state: st.session_state["active_job_id"] = None

    if st.session_state["coder_id"] is None:
        st.title("TK Annotation Interface")
        st.markdown("#### Select your coder ID to begin")
        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Coder A", use_container_width=True, type="primary"):
                st.session_state["coder_id"] = "Coder A"; st.rerun()
        with col2:
            if st.button("Coder B", use_container_width=True):
                st.session_state["coder_id"] = "Coder B"; st.rerun()
        st.markdown("---")
        st.caption("20 shared IRR cases · Coder A: 61 cases · Coder B: 60 cases")
        return

    df          = load_annotations()
    queue_df    = build_queue(df)
    assignments = assign_cases(queue_df, get_irr_pool(queue_df))
    decisions   = get_decisions()
    coder_id    = st.session_state["coder_id"]

    sel_occ, sel_sen, sel_status, done_ids, my_cases = render_sidebar(
        queue_df, decisions, assignments, coder_id)

    tab_queue, tab_irr, tab_export = st.tabs(["Queue", "IRR Dashboard", "Export"])
    with tab_queue:
        if st.session_state["active_job_id"] is not None:
            render_annotation_card(df[df["job_id"] == st.session_state["active_job_id"]].iloc[0], decisions, coder_id)
        else:
            render_queue(queue_df, decisions, assignments, coder_id, sel_occ, sel_sen, sel_status, done_ids, my_cases)
    with tab_irr:
        render_irr_dashboard(queue_df, decisions)
    with tab_export:
        render_export(df, decisions)


if __name__ == "__main__":
    main()