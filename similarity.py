import pandas as pd


def normalize_patient_ids(series):
    values = pd.to_numeric(series, errors="coerce")
    return sorted(values.dropna().astype(int).unique().tolist())


def get_condition_profile(df, patient_id):
    rows = df[df["person_id"] == patient_id]
    if rows.empty or "condition_concept_id" not in rows.columns:
        return set()
    concept_ids = pd.to_numeric(rows["condition_concept_id"], errors="coerce")
    return set(concept_ids[concept_ids.notna() & (concept_ids != 0)].astype(int).tolist())


def map_concept_name(concept_id, concept_lookup):
    if pd.isna(concept_id):
        return "Unknown"
    try:
        value = int(float(concept_id))
    except (TypeError, ValueError):
        return str(concept_id)
    if value == 0:
        return "Unknown"
    return concept_lookup.get(value, str(concept_id))


def prepare_patient_condition_table(selected_patient_id, conditions, concept_lookup):
    if conditions.empty:
        return pd.DataFrame(columns=["Condition", "Start date", "Source value"])

    rows = conditions[conditions["person_id"] == selected_patient_id].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Condition", "Start date", "Source value"])

    rows["condition_concept_id"] = pd.to_numeric(rows.get("condition_concept_id", pd.Series(dtype=float)), errors="coerce")
    rows["Condition"] = rows["condition_concept_id"].apply(lambda cid: map_concept_name(cid, concept_lookup))
    rows["Condition"] = rows["Condition"].replace({
        "Unknown": rows.get("condition_source_value", pd.Series(["Unknown"] * len(rows))).fillna("Unknown")
    })
    rows["Start date"] = rows.get("condition_start_date", pd.Series([None] * len(rows)))
    rows["Source value"] = rows.get("condition_source_value", pd.Series([None] * len(rows)))
    rows = rows[["Condition", "Start date", "Source value"]]
    rows = rows.drop_duplicates().sort_values(["Condition", "Start date"], na_position="last").reset_index(drop=True)
    return rows


def prepare_patient_medication_table(selected_patient_id, drugs, concept_lookup):
    if drugs.empty:
        return pd.DataFrame(columns=["Medication", "Start date", "Quantity", "Source value"])

    rows = drugs[drugs["person_id"] == selected_patient_id].copy()
    if rows.empty:
        return pd.DataFrame(columns=["Medication", "Start date", "Quantity", "Source value"])

    rows["drug_concept_id"] = pd.to_numeric(rows.get("drug_concept_id", pd.Series(dtype=float)), errors="coerce")
    rows["Medication"] = rows["drug_concept_id"].apply(lambda cid: map_concept_name(cid, concept_lookup))
    rows["Medication"] = rows["Medication"].replace({
        "Unknown": rows.get("drug_source_value", pd.Series(["Unknown"] * len(rows))).fillna("Unknown")
    })
    rows["Start date"] = rows.get("drug_exposure_start_date", pd.Series([None] * len(rows)))
    rows["Quantity"] = rows.get("quantity", pd.Series([None] * len(rows)))
    rows["Source value"] = rows.get("drug_source_value", pd.Series([None] * len(rows)))
    rows = rows[["Medication", "Start date", "Quantity", "Source value"]]
    rows = rows.drop_duplicates().sort_values(["Medication", "Start date"], na_position="last").reset_index(drop=True)
    return rows
