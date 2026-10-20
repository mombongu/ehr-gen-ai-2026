import os
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None

st.set_page_config(
    page_title="Clinical Patient Explorer",
    page_icon="🔎",
    layout="wide",
)

DATA_DIR = Path(__file__).parent / "0.9" / "1_omop_data_csv"


@st.cache_data
def load_table(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def normalize_patient_ids(series):
    values = pd.to_numeric(series, errors="coerce")
    return sorted(values.dropna().astype(int).unique().tolist())


def get_condition_profile(df, patient_id):
    rows = df[df["person_id"] == patient_id]
    if rows.empty or "condition_concept_id" not in rows.columns:
        return set()
    concept_ids = pd.to_numeric(rows["condition_concept_id"], errors="coerce")
    return set(concept_ids[concept_ids.notna() & (concept_ids != 0)].astype(int).tolist())


def similarity(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def summarize_therapy(df, patient_ids):
    if df.empty or df is None:
        return pd.DataFrame(columns=["Therapy", "Similar patients"])

    candidate = df[df["person_id"].isin(patient_ids)].copy()
    if candidate.empty:
        return pd.DataFrame(columns=["Therapy", "Similar patients"])

    label_column = "drug_source_value" if "drug_source_value" in candidate.columns else "drug_concept_id"
    candidate["therapy_label"] = candidate[label_column].fillna("Unknown").astype(str)
    summary = (
        candidate.groupby("therapy_label", as_index=False)["person_id"]
        .nunique()
        .rename(columns={"therapy_label": "Therapy", "person_id": "Similar patients"})
    )
    return summary.sort_values("Similar patients", ascending=False).head(10)


def generate_ai_summary(selected_patient_id, similarity_df, therapy_summary, metric_summary):
    if OpenAI is None:
        return "OpenAI support is not installed. Add the `openai` package and set `OPENAI_API_KEY` to enable this feature."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Set the `OPENAI_API_KEY` environment variable to enable the optional AI summary."

    top_matches = similarity_df.head(5).copy()
    top_matches["Similarity"] = top_matches["Similarity"].round(3)

    context = {
        "selected_patient_id": selected_patient_id,
        "similarity": top_matches.to_dict(orient="records"),
        "therapy_summary": therapy_summary.head(5).to_dict(orient="records"),
        "outcome_metrics": metric_summary,
    }

    prompt = f"""
    You are a clinical research analyst. Based on the following OMOP-style patient cohort context,
    write a brief, clinician-friendly summary for Patient {selected_patient_id}.
    Keep it concise, avoid making definitive diagnoses, and clearly label this as a research-oriented summary.

    Context:
    {context}
    """

    client = OpenAI(api_key=api_key)

    try:
        if hasattr(client, "responses"):
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()
            if hasattr(response, "output") and response.output:
                message = response.output[0].content[0].text.value
                return message.strip()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # pragma: no cover - depends on external API
        return f"AI summary could not be generated: {exc}"


person = load_table("person.csv")
conditions = load_table("condition_occurrence.csv")
drugs = load_table("drug_exposure.csv")
procedures = load_table("procedure_occurrence.csv")
measurements = load_table("measurement.csv")

st.title("Clinical Patient Explorer")
st.caption("Explore a patient, compare them with a similar cohort, and review their treatment and outcome context.")

if person.empty or conditions.empty:
    st.error(f"Could not find required files in {DATA_DIR}")
    st.stop()

patient_ids = normalize_patient_ids(person["person_id"])
if not patient_ids:
    st.warning("No valid patient IDs were found in `person.csv`.")
    st.stop()

profiles = {pid: get_condition_profile(conditions, pid) for pid in patient_ids}

selected_id = st.selectbox("Patient selector", patient_ids, format_func=lambda pid: f"Patient {pid}")
selected_profile = profiles.get(selected_id, set())

similarity_rows = []
for pid in patient_ids:
    if pid == selected_id:
        continue
    other = profiles.get(pid, set())
    shared = len(selected_profile & other)
    score = similarity(selected_profile, other)
    similarity_rows.append(
        {
            "Patient ID": pid,
            "Similarity": round(score, 3),
            "Shared condition concepts": shared,
        }
    )

similarity_df = pd.DataFrame(similarity_rows)
if not similarity_df.empty:
    similarity_df = similarity_df.sort_values(
        ["Similarity", "Shared condition concepts"],
        ascending=[False, False],
    ).reset_index(drop=True)

left_col, right_col = st.columns([1.1, 1.4])

with left_col:
    st.subheader("Similarity chart")
    if not similarity_df.empty:
        chart_df = similarity_df.head(8).sort_values("Similarity", ascending=True).copy()
        chart_df["Patient ID"] = chart_df["Patient ID"].astype(str)
        st.bar_chart(chart_df.set_index("Patient ID")["Similarity"])
    else:
        st.info("No similar patients were found for this selection.")

with right_col:
    st.subheader("Therapy summary")
    similar_patient_ids = similarity_df["Patient ID"].head(10).tolist() if not similarity_df.empty else []
    therapy_summary = summarize_therapy(drugs, similar_patient_ids)
    if therapy_summary.empty:
        st.info("Drug exposure data for the similar cohort is not available.")
    else:
        st.dataframe(therapy_summary, use_container_width=True, hide_index=True)

    st.subheader("OpenAI summary")
    openai_ready = OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
    if not openai_ready:
        st.caption("Optional: install the `openai` package and set `OPENAI_API_KEY` to enable this summary.")

    if st.button("Generate AI summary", type="primary", use_container_width=True, disabled=not openai_ready):
        with st.spinner("Generating a brief cohort summary..."):
            metric_summary = {
                "selected_patient_id": int(selected_id),
                "unique_conditions": len(selected_profile),
                "drug_exposure_records": len(drugs[drugs["person_id"] == selected_id]) if not drugs.empty else 0,
                "procedure_records": len(procedures[procedures["person_id"] == selected_id]) if not procedures.empty else 0,
                "measurement_records": len(measurements[measurements["person_id"] == selected_id]) if not measurements.empty else 0,
                "avg_cohort_similarity": round(similarity_df["Similarity"].mean(), 3) if not similarity_df.empty else 0.0,
            }
            summary = generate_ai_summary(selected_id, similarity_df, therapy_summary, metric_summary)
        st.markdown(summary)

st.subheader("Outcome metrics")
metric_cols = st.columns(4)
metric_cols[0].metric("Unique conditions", len(selected_profile))
metric_cols[1].metric(
    "Drug exposure records",
    len(drugs[drugs["person_id"] == selected_id]) if not drugs.empty else 0,
)
metric_cols[2].metric(
    "Procedure records",
    len(procedures[procedures["person_id"] == selected_id]) if not procedures.empty else 0,
)
metric_cols[3].metric(
    "Measurement records",
    len(measurements[measurements["person_id"] == selected_id]) if not measurements.empty else 0,
)

st.subheader("Similar cohort")
if similarity_df.empty:
    st.info("No comparable patients are available for this patient profile.")
else:
    table_df = similarity_df.head(10).copy()
    table_df["Similarity"] = table_df["Similarity"].round(3)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

st.subheader("Patient detail")
medication_col, procedure_col, measurement_col = st.columns(3)

with medication_col:
    st.caption("Medication records")
    if not drugs.empty:
        rows = drugs[drugs["person_id"] == selected_id].copy()
        cols = [c for c in ["drug_exposure_start_date", "drug_source_value", "drug_concept_id", "quantity"] if c in rows.columns]
        if cols:
            st.dataframe(rows[cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.info("Medication fields are not available for this dataset.")
    else:
        st.info("Drug exposure data not found.")

with procedure_col:
    st.caption("Procedure records")
    if not procedures.empty:
        rows = procedures[procedures["person_id"] == selected_id].copy()
        st.dataframe(rows.head(25), use_container_width=True, hide_index=True)
    else:
        st.info("Procedure data not found.")

with measurement_col:
    st.caption("Measurement records")
    if not measurements.empty:
        rows = measurements[measurements["person_id"] == selected_id].copy()
        cols = [c for c in ["measurement_date", "measurement_source_value", "measurement_concept_id", "value_as_number", "unit_source_value"] if c in rows.columns]
        if cols:
            st.dataframe(rows[cols].head(25), use_container_width=True, hide_index=True)
        else:
            st.info("Measurement fields are not available for this dataset.")
    else:
        st.info("Measurement data not found.")

st.caption("Research prototype. Similarity is based on condition overlap and is not a validated clinical matching method.")
