import re
import PyPDF2
from transformers import pipeline

from app.services.skills_data import SKILLS_LIST

# Load the NER pipeline once (this downloads a small model the first time it runs)
_ner_pipeline = None


def get_ner_pipeline():
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF resume file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def clean_text_for_ner(text: str) -> str:
    """Add spacing around common PDF extraction issues (glued words, punctuation)."""
    # Add space between a lowercase letter followed by an uppercase letter (e.g. "CoimbatoreDec" -> "Coimbatore Dec")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    # Add space around pipe/bullet characters
    text = re.sub(r"[|•]", " ", text)
    # Collapse multiple spaces/newlines into one space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> list[str]:
    """Match known skills against the resume text (case-insensitive)."""
    text_lower = text.lower()
    found_skills = []
    for skill in SKILLS_LIST:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)
    return found_skills


def extract_organizations(text: str) -> list[str]:
    """Use Hugging Face NER to extract organization names from resume text."""
    ner = get_ner_pipeline()
    cleaned = clean_text_for_ner(text)
    entities = ner(cleaned[:2000])
    orgs = set()
    for ent in entities:
        word = ent["word"].strip()
        if (
            ent["entity_group"] == "ORG"
            and ent["score"] > 0.75          # higher confidence threshold
            and len(word) > 2                # drop single/double-letter fragments
            and not word.startswith("##")    # drop leftover sub-word pieces
        ):
            orgs.add(word)
    return list(orgs)
def extract_experience_years(text: str) -> float:
    """
    Estimate years of experience by looking for date ranges like
    'Dec 2024' ... 'Jun 2025', or explicit patterns like '2 years experience'.
    This is a simple heuristic, not exact.
    """
    # Look for explicit "X years" mentions first
    explicit = re.search(r"(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience", text, re.IGNORECASE)
    if explicit:
        return float(explicit.group(1))

    # Fallback: find all 4-digit years mentioned, estimate span
    years = [int(y) for y in re.findall(r"\b(20\d{2})\b", text)]
    if len(years) >= 2:
        return float(max(years) - min(years))

    return 0.0


if __name__ == "__main__":
    sample_path = "data/sample/sample_resume.pdf"
    extracted = extract_text_from_pdf(sample_path)

    skills = extract_skills(extracted)
    print(f"--- Skills Found ({len(skills)}) ---")
    print(skills)

    orgs = extract_organizations(extracted)
    print(f"\n--- Organizations Found ({len(orgs)}) ---")
    print(orgs)

    experience = extract_experience_years(extracted)
    print(f"\n--- Estimated Experience (years) ---")
    print(experience)