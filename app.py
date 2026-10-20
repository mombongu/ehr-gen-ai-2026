import os
from datetime import datetime

import pandas as pd
import streamlit as st

from ai_summary import generate_ai_summary, generate_patient_follow_up
from data_loader import (
    DATA_DIR,
    find_data_dir,
    get_condition_profile,
    load_concept_lookup,
    load_table,
    map_concept_name,
    normalize_patient_ids,
    prepare_patient_condition_table,
    prepare_patient_medication_table,
    required_csv_map,
)
from evidence import build_case_summary, summarize_common_field
from similarity import build_similarity_table

st.set_page_config(
    page_title="Clinical Patient Explorer",
    page_icon="🔎",
    layout="wide",
)


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
concept_lookup = load_concept_lookup()

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

similarity_df = build_similarity_table(selected_id, patient_ids, profiles, similarity_limit=similarity_limit)
similar_ids = similarity_df["Patient ID"].head(top_n_similar).tolist() if not similarity_df.empty else []

selected_patient_row = person[person["person_id"] == selected_id].iloc[0].to_dict() if not person[person["person_id"] == selected_id].empty else {}
selected_conditions = prepare_patient_condition_table(selected_id, conditions, concept_lookup)
selected_medications = prepare_patient_medication_table(selected_id, drugs, concept_lookup)

st.subheader("Selected patient")
patient_meta = st.columns([1.3, 1.3, 1.3, 1.3, 1.3])
patient_meta[0].metric("Patient ID", selected_id)
patient_meta[1].metric("Conditions", len(selected_conditions))
patient_meta[2].metric("Medications", len(selected_medications))
patient_meta[3].metric("Similar patients", len(similar_ids))

age_value = selected_patient_row.get("year_of_birth")
if pd.notna(age_value) and str(age_value).strip():
    age_str = f"{datetime.now().year - int(float(age_value))}"
else:
    age_str = "Unknown"
patient_meta[4].metric("Recorded age", age_str)

sex_value = selected_patient_row.get("gender_source_value")
race_value = selected_patient_row.get("race_source_value")
ethnicity_value = selected_patient_row.get("ethnicity_source_value")

info_cols = st.columns(4)
info_cols[0].markdown(f"**Sex:** {sex_value if pd.notna(sex_value) and str(sex_value).strip() else 'Unknown'}")
info_cols[1].markdown(f"**Race:** {race_value if pd.notna(race_value) and str(race_value).strip() else 'Unknown'}")
info_cols[2].markdown(f"**Ethnicity:** {ethnicity_value if pd.notna(ethnicity_value) and str(ethnicity_value).strip() else 'Unknown'}")
info_cols[3].markdown(f"**Condition ID count:** {len(selected_profile)}")

st.subheader("Patient condition list")
if selected_conditions.empty:
    st.info("No condition records were found for this patient.")
else:
    st.dataframe(selected_conditions.head(50), use_container_width=True, hide_index=True)

st.subheader("Patient medication list")
if selected_medications.empty:
    st.info("No medication records were found for this patient.")
else:
    st.dataframe(selected_medications.head(50), use_container_width=True, hide_index=True)

st.subheader("Ask about this patient")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Ask me about this patient’s conditions, medications, age, or overall chart overview."}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask a follow-up question about this patient")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = generate_patient_follow_up(selected_id, selected_patient_row, selected_conditions, selected_medications, prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        st.write(response)

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
    openai_ready = os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY").strip()
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
            rows["Medication"] = rows["drug_concept_id"].apply(lambda cid: map_concept_name(cid, concept_lookup))
            unknown_mask = rows["Medication"].eq("Unknown")
            if unknown_mask.any():
                rows.loc[unknown_mask, "Medication"] = rows.loc[unknown_mask, "drug_source_value"].fillna("Unknown")
            st.dataframe(rows[["drug_exposure_start_date", "Medication", "quantity"] if "quantity" in rows.columns else ["drug_exposure_start_date", "Medication"]].head(25), use_container_width=True, hide_index=True)
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
