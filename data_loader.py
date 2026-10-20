from pathlib import Path

import pandas as pd
import streamlit as st


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
        "concept": find_csv_file(data_dir, ["2b_concept.csv", "concept.csv"]),
    }


@st.cache_data
def load_table(filename, data_dir=None):
    target_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    path = target_dir / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


@st.cache_data
def load_concept_lookup(data_dir=None):
    target_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    concept_path = target_dir / "2b_concept.csv"
    if not concept_path.exists():
        concept_path = target_dir / "concept.csv"
    if not concept_path.exists():
        return {}

    concept_df = pd.read_csv(concept_path, low_memory=False)
    if "concept_id" not in concept_df.columns or "concept_name" not in concept_df.columns:
        return {}

    concept_df = concept_df[["concept_id", "concept_name"]].copy()
    concept_df["concept_id"] = pd.to_numeric(concept_df["concept_id"], errors="coerce")
    concept_df = concept_df.dropna(subset=["concept_id"]).drop_duplicates(subset=["concept_id"])
    concept_df["concept_name"] = concept_df["concept_name"].fillna("Unknown")
    return concept_df.set_index("concept_id")["concept_name"].to_dict()
