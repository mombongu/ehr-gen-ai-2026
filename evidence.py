import pandas as pd


def similarity(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_similarity_table(selected_id, patient_ids, profiles, similarity_limit=0.0):
    selected_profile = profiles.get(selected_id, set())
    rows = []
    for pid in patient_ids:
        if pid == selected_id:
            continue
        other = profiles.get(pid, set())
        shared = len(selected_profile & other)
        score = similarity(selected_profile, other)
        rows.append({
            "Patient ID": pid,
            "Similarity": round(score, 3),
            "Shared conditions": shared,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df[df["Similarity"] >= similarity_limit].sort_values(
        ["Similarity", "Shared conditions"], ascending=[False, False]
    ).reset_index(drop=True)
    return df


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
