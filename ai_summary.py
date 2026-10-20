import os
from datetime import datetime

import pandas as pd

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


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


def build_local_patient_answer(prompt, patient_row, condition_rows, medication_rows):
    q = (prompt or "").lower()
    condition_names = [str(value) for value in condition_rows["Condition"].dropna().tolist() if str(value).strip()][:20]
    med_names = [str(value) for value in medication_rows["Medication"].dropna().tolist() if str(value).strip()][:20]

    if any(term in q for term in ["condition", "diagnosis", "diagnoses", "problems", "problem"]):
        if not condition_names:
            return "This patient currently has no condition entries in the available OMOP record set."
        return "Current conditions: " + "; ".join(condition_names)

    if any(term in q for term in ["med", "medication", "drug", "prescription", "treatment", "therapy"]):
        if not med_names:
            return "This patient has no medication records in the available dataset."
        return "Current medications: " + "; ".join(med_names)

    if any(term in q for term in ["age", "years old", "birth"]):
        year_of_birth = patient_row.get("year_of_birth")
        if pd.notna(year_of_birth):
            age = datetime.now().year - int(year_of_birth)
            return f"Patient {patient_row.get('person_id')} is approximately {age} years old."
        return "The dataset does not include a usable year of birth for this patient."

    if any(term in q for term in ["sex", "gender"]):
        gender = patient_row.get("gender_source_value")
        if pd.notna(gender):
            return f"This patient is recorded as {gender}."
        return "Sex/gender is not clearly recorded in the available patient data."

    if any(term in q for term in ["summary", "overview", "tell me about", "what stands out"]):
        condition_text = ", ".join(condition_names) if condition_names else "no recorded conditions"
        med_text = ", ".join(med_names) if med_names else "no recorded medications"
        return (
            f"Patient {patient_row.get('person_id')} has {len(condition_names)} recorded conditions "
            f"({condition_text}) and {len(med_names)} medication entries ({med_text})."
        )

    if any(term in q for term in ["what is their name", "name", "patient id"]):
        return f"This is Patient {patient_row.get('person_id')} from the selected OMOP record set."

    return (
        "I can answer questions about this patient’s conditions, medications, age, sex, and overall chart summary. "
        f"Current conditions: {condition_names[:5] if condition_names else 'none'}; current medications: {med_names[:5] if med_names else 'none'}."
    )


def generate_patient_follow_up(selected_patient_id, patient_row, condition_rows, medication_rows, prompt):
    if OpenAI is not None and os.getenv("OPENAI_API_KEY"):
        context = {
            "patient_id": int(selected_patient_id),
            "gender": patient_row.get("gender_source_value", "Unknown"),
            "race": patient_row.get("race_source_value", "Unknown"),
            "year_of_birth": patient_row.get("year_of_birth"),
            "conditions": condition_rows[["Condition", "Start date"]].head(20).to_dict(orient="records"),
            "medications": medication_rows[["Medication", "Start date"]].head(20).to_dict(orient="records"),
            "question": prompt,
        }
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            response = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {
                        "role": "system",
                        "content": "You are a clinical chart assistant. Use only the provided patient information. Be careful, do not make clinical diagnoses. If a fact is missing, say it is not available.",
                    },
                    {"role": "user", "content": f"Here is the patient chart context: {context}\n\nUser question: {prompt}"},
                ],
            )
            if hasattr(response, "output_text") and response.output_text:
                return response.output_text.strip()
            if hasattr(response, "output") and response.output:
                return response.output[0].content[0].text.value.strip()
        except Exception:
            pass

    return build_local_patient_answer(prompt, patient_row, condition_rows, medication_rows)
