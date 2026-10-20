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


def find_data_dir():
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / "1_omop_data_csv",
        base_dir / "1_omop_Data_csv",
        base_dir / "0.9" / "1_omop_data_csv",
        base_dir / "0.9" / "1_omop_Data_csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base_dir / "1_omop_data_csv"


DATA_DIR = find_data_dir()


def find_csv_file(data_dir, patterns):
    if not data_dir.exists():
        return None
    files = sorted(p for p in data_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    for pattern in patterns:
        for candidate in files:
            if candidate.name.lower() == pattern.lower():
                return candidate
    for pattern in patterns:
        lowered = pattern.lower()
        for candidate in files:
            if lowered in candidate.name.lower():
                return candidate
    return None


def required_csv_map(data_dir):
    return {
        "person": find_csv_file(data_dir, ["person.csv", "person_*.csv"]),
        "condition": find_csv_file(data_dir, ["condition_occurrence.csv", "condition*.csv"]),
        "drug": find_csv_file(data_dir, ["drug_exposure.csv", "drug*.csv"]),
        "procedure": find_csv_file(data_dir, ["procedure_occurrence.csv", "procedure*.csv"]),
        "measurement": find_csv_file(data_dir, ["measurement.csv", "measurement*.csv"]),
    }


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


def summarize_common_field(df, patient_ids, label_columns, label_name, value_name):
    if df is None or df.empty or not patient_ids:
        return pd.DataFrame(columns=[label_name, value_name])

    cohort = df[df["person_id"].isin(patient_ids)].copy()
    if cohort.empty:
        return pd.DataFrame(columns=[label_name, value_name])

    label_col = None
    for candidate in label_columns:
        if candidate in cohort.columns:
            label_col = candidate
            break
    if label_col is None:
        return pd.DataFrame(columns=[label_name, value_name])

    cohort["label_value"] = cohort[label_col].fillna("Unknown").astype(str)
    summary = (
        cohort.groupby("label_value", as_index=False)["person_id"]
        .nunique()
        .rename(columns={"label_value": label_name, "person_id": value_name})
    )
    return summary.sort_values(value_name, ascending=False).head(10).reset_index(drop=True)


def build_case_summary(selected_id, similar_df, drug_summary, procedure_summary, measurement_summary):
    top_drug = drug_summary.iloc[0]["Drug"] if not drug_summary.empty else "no common drug pattern"
    top_proc = procedure_summary.iloc[0]["Procedure"] if not procedure_summary.empty else "no common procedure pattern"
    top_meas = measurement_summary.iloc[0]["Measurement"] if not measurement_summary.empty else "no common lab pattern"

    return (
        f"Based on the {len(similar_df)} patients most similar to Patient {selected_id}, the most common patterns include "
        f"{top_drug}, {top_proc}, and {top_meas}. This retrospective view highlights the treatment and procedure patterns "
        f"typically observed in comparable OMOP cohorts, but it should be interpreted as exploratory clinical research rather than a verified care recommendation."
    )


def generate_ai_summary(selected_patient_id, similarity_df, therapy_summary, procedure_summary, measurement_summary, metric_summary):
    if OpenAI is None:
        return "OpenAI support is not installed. Add the `openai` package and set `OPENAI_API_KEY` to enable this feature."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Set the `OPENAI_API_KEY` environment variable to enable the optional AI summary."

    top_matches = similarity_df.head(5).copy()
    if not top_matches.empty:
        top_matches["Similarity"] = top_matches["Similarity"].round(3)

    context = {
        "selected_patient_id": selected_patient_id,
        "similarity": top_matches.to_dict(orient="records"),
        "therapy_summary": therapy_summary.head(5).to_dict(orient="records"),
        "procedure_summary": procedure_summary.head(5).to_dict(orient="records"),
        "measurement_summary": measurement_summary.head(5).to_dict(orient="records"),
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


required_files = required_csv_map(DATA_DIR)
available_csvs = sorted(
    [p.name for p in DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".csv"]
) if DATA_DIR.exists() else []

if DATA_DIR.exists() and not required_files["person"]:
    st.error(
        f"Missing required OMOP files in {DATA_DIR}. "
        f"Expected at least `person.csv` (and `condition_occurrence.csv`, `drug_exposure.csv`, `procedure_occurrence.csv`, `measurement.csv`). "
        f"Files found: {available_csvs[:20]}"
    )
    st.stop()

person = load_table(required_files["person"].name if required_files["person"] else "person.csv")
conditions = load_table(required_files["condition"].name if required_files["condition"] else "condition_occurrence.csv")
drugs = load_table(required_files["drug"].name if required_files["drug"] else "drug_exposure.csv")
procedures = load_table(required_files["procedure"].name if required_files["procedure"] else "procedure_occurrence.csv")
measurements = load_table(required_files["measurement"].name if required_files["measurement"] else "measurement.csv")

st.title("Clinical Patient Explorer")
st.caption("Explore a patient, compare them with a similar cohort, and review the treatment patterns used in prior comparable cases.")

if person.empty or conditions.empty:
    st.error(
        f"Could not find the required OMOP patient tables in {DATA_DIR}. "
        f"Expected `person.csv` and `condition_occurrence.csv`; checked folder contents: {available_csvs[:20]}"
    )
    st.stop()

patient_ids = normalize_patient_ids(person["person_id"])
if not patient_ids:
    st.warning("No valid patient IDs were found in the person table.")
    st.stop()

profiles = {pid: get_condition_profile(conditions, pid) for pid in patient_ids}

st.sidebar.header("Patient and cohort filters")
selected_id = st.sidebar.selectbox("Patient selector", patient_ids, format_func=lambda pid: f"Patient {pid}")
selected_profile = profiles.get(selected_id, set())

similarity_limit = st.sidebar.slider("Minimum similarity score", min_value=0.0, max_value=0.5, value=0.0, step=0.01)
top_n_similar = st.sidebar.slider("Top similar patients to show", min_value=5, max_value=50, value=10, step=1)

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
            "Shared conditions": shared,
        }
    )

similarity_df = pd.DataFrame(similarity_rows)
if not similarity_df.empty:
    similarity_df = similarity_df[
        similarity_df["Similarity"] >= similarity_limit
    ].sort_values(["Similarity", "Shared conditions"], ascending=[False, False]).reset_index(drop=True)

similar_ids = similarity_df["Patient ID"].head(top_n_similar).tolist() if not similarity_df.empty else []

st.subheader("Patient overview")
metric_cols = st.columns(4)
metric_cols[0].metric("Unique conditions", len(selected_profile))
metric_cols[1].metric("Similar patients", len(similar_ids))
metric_cols[2].metric(
    "Drug exposure records",
    len(drugs[drugs["person_id"] == selected_id]) if not drugs.empty else 0,
)
metric_cols[3].metric(
    "Procedure records",
    len(procedures[procedures["person_id"] == selected_id]) if not procedures.empty else 0,
)

left_col, right_col = st.columns([1.1, 1.4])

with left_col:
    st.subheader("Similarity chart")
    if similarity_df.empty:
        st.info("No similar patients are available for this patient profile.")
    else:
        chart_df = similarity_df.head(8).copy().sort_values("Similarity", ascending=True)
        chart_df["Patient ID"] = chart_df["Patient ID"].astype(str)
        st.bar_chart(chart_df.set_index("Patient ID")["Similarity"])

with right_col:
    st.subheader("What has been done before in similar cases?")
    if not similar_ids:
        st.info("No comparable patients meet the current similarity threshold.")
    else:
        drug_summary = summarize_common_field(
            drugs,
            similar_ids,
            ["drug_source_value", "drug_concept_id"],
            "Drug",
            "Patients",
        )
        procedure_summary = summarize_common_field(
            procedures,
            similar_ids,
            ["procedure_source_value", "procedure_concept_id"],
            "Procedure",
            "Patients",
        )
        measurement_summary = summarize_common_field(
            measurements,
            similar_ids,
            ["measurement_source_value", "measurement_concept_id"],
            "Measurement",
            "Patients",
        )

        evidence_text = build_case_summary(selected_id, similarity_df.head(top_n_similar), drug_summary, procedure_summary, measurement_summary)
        st.markdown(f"_Research-oriented summary:_ {evidence_text}")
        st.caption("Patterns below are derived from the similar cohort only and are intended for retrospective exploration.")
        st.dataframe(drug_summary, use_container_width=True, hide_index=True)

st.subheader("Similar cohort")
if similarity_df.empty:
    st.info("No comparable patients are available for this patient profile.")
else:
    table_df = similarity_df.head(top_n_similar).copy()
    table_df["Similarity"] = table_df["Similarity"].round(3)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

if similar_ids:
    drug_summary = summarize_common_field(
        drugs,
        similar_ids,
        ["drug_source_value", "drug_concept_id"],
        "Drug",
        "Patients",
    )
    procedure_summary = summarize_common_field(
        procedures,
        similar_ids,
        ["procedure_source_value", "procedure_concept_id"],
        "Procedure",
        "Patients",
    )
    measurement_summary = summarize_common_field(
        measurements,
        similar_ids,
        ["measurement_source_value", "measurement_concept_id"],
        "Measurement",
        "Patients",
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Common medications")
        if drug_summary.empty:
            st.info("No medication patterns found in the similar cohort.")
        else:
            st.dataframe(drug_summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Common procedures")
        if procedure_summary.empty:
            st.info("No procedure patterns found in the similar cohort.")
        else:
            st.dataframe(procedure_summary, use_container_width=True, hide_index=True)

    with col3:
        st.subheader("Common measurements")
        if measurement_summary.empty:
            st.info("No measurement patterns found in the similar cohort.")
        else:
            st.dataframe(measurement_summary, use_container_width=True, hide_index=True)

    st.subheader("OpenAI summary")
    openai_ready = OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
    if not openai_ready:
        st.caption("Optional: install the `openai` package and set `OPENAI_API_KEY` to enable this summary.")

    if st.button("Generate AI summary", type="primary", use_container_width=True, disabled=not openai_ready):
        with st.spinner("Generating a research-oriented summary..."):
            metric_summary = {
                "selected_patient_id": int(selected_id),
                "unique_conditions": len(selected_profile),
                "drug_exposure_records": len(drugs[drugs["person_id"] == selected_id]) if not drugs.empty else 0,
                "procedure_records": len(procedures[procedures["person_id"] == selected_id]) if not procedures.empty else 0,
                "measurement_records": len(measurements[measurements["person_id"] == selected_id]) if not measurements.empty else 0,
                "avg_cohort_similarity": round(similarity_df["Similarity"].mean(), 3) if not similarity_df.empty else 0.0,
            }
            summary = generate_ai_summary(
                selected_id,
                similarity_df,
                drug_summary,
                procedure_summary,
                measurement_summary,
                metric_summary,
            )
        st.markdown(summary)

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
