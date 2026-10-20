import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Clinical Patient Explorer",
    page_icon="🔎",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "0.9" / "1_omop_data_csv"

@st.cache_data
def load_table(filename):
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)

person = load_table("person.csv")
conditions = load_table("condition_occurrence.csv")
drugs = load_table("drug_exposure.csv")
procedures = load_table("procedure_occurrence.csv")
measurements = load_table("measurement.csv")

st.title("Clinical Patient Explorer")
st.write("Explore a patient and compare records across patients.")

if person.empty or conditions.empty:
    st.error(f"Could not find required files in {DATA_DIR}")
    st.stop()

patient_ids = sorted(person["person_id"].dropna().unique(), key=str)

def get_profile(pid):
    rows = conditions[conditions["person_id"] == pid]
    if "condition_concept_id" not in rows.columns:
        return set()
    ids = pd.to_numeric(rows["condition_concept_id"], errors="coerce")
    return set(ids[ids.notna() & (ids != 0)].astype(int))

profiles = {pid: get_profile(pid) for pid in patient_ids}

def similarity(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0

selected_id = st.selectbox(
    "Select a patient",
    patient_ids,
    format_func=lambda pid: f"Patient {pid}"
)

selected_profile = profiles[selected_id]
ranked = []

for pid in patient_ids:
    if pid == selected_id:
        continue
    other = profiles[pid]
    ranked.append({
        "Patient ID": pid,
        "Similarity": round(similarity(selected_profile, other), 3),
        "Shared condition concepts": len(selected_profile & other)
    })

ranked.sort(key=lambda x: x["Similarity"], reverse=True)

c1, c2, c3 = st.columns(3)
c1.metric("Patients", len(patient_ids))
c2.metric("Condition concepts", len(selected_profile))
c3.metric(
    "Drug exposure records",
    len(drugs[drugs["person_id"] == selected_id]) if not drugs.empty else 0
)

st.subheader("Most similar patients")
st.dataframe(pd.DataFrame(ranked[:10]), use_container_width=True, hide_index=True)

st.subheader("Medication records")
if not drugs.empty:
    rows = drugs[drugs["person_id"] == selected_id]
    cols = [c for c in [
        "drug_exposure_start_date", "drug_source_value",
        "drug_concept_id", "quantity"
    ] if c in rows.columns]
    st.dataframe(rows[cols].head(100), use_container_width=True, hide_index=True)
else:
    st.info("Drug exposure data not found.")

st.subheader("Procedure records")
if not procedures.empty:
    rows = procedures[procedures["person_id"] == selected_id]
    st.dataframe(rows.head(100), use_container_width=True, hide_index=True)
else:
    st.info("Procedure data not found.")

st.subheader("Measurement records")
if not measurements.empty:
    rows = measurements[measurements["person_id"] == selected_id]
    cols = [c for c in [
        "measurement_date", "measurement_source_value",
        "measurement_concept_id", "value_as_number", "unit_source_value"
    ] if c in rows.columns]
    st.dataframe(rows[cols].head(100), use_container_width=True, hide_index=True)
else:
    st.info("Measurement data not found.")

st.caption(
    "Research prototype. Similarity is based on condition overlap "
    "and is not a validated clinical matching method."
)
