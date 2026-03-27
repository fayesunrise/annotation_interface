# TK Annotation Interface

Human-LLM collaborative re-annotation for flagged cases.
Adapted from MEGAnno+ (Kim et al., 2024) for tacit knowledge annotation.

## Setup

```bash
pip install -r requirements.txt
```

## Configure

Open `app.py` and update the top section:

```python
DATA_DIR        = Path(r"C:\Research-2026\TK-annotation\annotation_results")
ANNOTATIONS_CSV = DATA_DIR / "annotation_annotated.csv"   # your Round 1 file
DECISIONS_CSV   = DATA_DIR / "human_decisions.csv"        # auto-created
STATE_JSON      = DATA_DIR / "coder_state.json"           # auto-created
OPENAI_API_KEY  = "sk-..."                                # your key
MODEL           = "gpt-4o-mini"
```

## Run

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## Workflow

1. Coder selects their ID (Coder A or Coder B) on the login screen
2. Queue tab shows their assigned cases sorted by verifier_score
3. Click "Annotate →" to open the annotation card
4. Left panel: job description + 5 LLM run cards with evidence and rationale
5. Right panel: assign TK-0/1/2, select override reason if disagreeing, write note
6. Optional: click "Re-query LLM →" to inject note and get revised LLM score
7. Click "Save & next case" — auto-advances to next pending case
8. IRR Dashboard tab shows Cohen's κ and conflict table for the 20 shared cases
9. Export tab generates tk_final_labels.csv merging human + LLM labels

## Case assignment

| Pool | Cases | Both coders |
|------|-------|-------------|
| IRR overlap | 20 (top verifier_score) | Yes |
| Coder A exclusive | 41 | No |
| Coder B exclusive | 40 | No |
| **Total** | **101** | — |

## Output columns (human_decisions.csv)

| Column | Values |
|--------|--------|
| job_id | posting identifier |
| coder_id | Coder A / Coder B |
| human_label | 0 / 1 / 2 |
| decision_type | accept_modal / override / requery_accept / requery_override |
| override_reason | structured category |
| note | free text |
| requery_label | LLM score after note injection (or blank) |
| requery_note | note used for re-query |
| timestamp | ISO datetime |
