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
