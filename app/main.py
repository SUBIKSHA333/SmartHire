import shutil
import os
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db, engine, Base
from app import models
from app.services.resume_parser import (
    extract_text_from_pdf,
    extract_skills,
    extract_experience_years,
)
from app.services.llm_service import generate_candidate_feedback, generate_job_summary

# Create tables on startup if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="SmartHire API")

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load trained ML models once at startup
fit_model = joblib.load("ml/models/fit_model.joblib")
salary_model = joblib.load("ml/models/salary_model.joblib")
salary_feature_cols = joblib.load("ml/models/salary_feature_columns.joblib")

LOCATIONS = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Remote"]
ML_SKILLS = {"machine learning", "deep learning", "nlp", "tensorflow", "pytorch"}


@app.get("/")
def home():
    return {"message": "SmartHire API is running!"}


@app.post("/candidates/upload-resume")
def upload_resume(name: str, email: str, location: str, db: Session = Depends(get_db)):
    """Note: actual file upload handled in next endpoint; this is a placeholder for clarity."""
    raise HTTPException(status_code=400, detail="Use /candidates/upload-resume-file instead")


@app.post("/candidates/upload-resume-file")
async def upload_resume_file(
    name: str,
    email: str,
    location: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a PDF resume, parse it, and save the candidate to the database."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    resume_text = extract_text_from_pdf(file_path)
    skills = extract_skills(resume_text)
    experience_years = extract_experience_years(resume_text)

    candidate = models.Candidate(
        name=name,
        email=email,
        resume_text=resume_text,
        skills=",".join(skills),
        experience_years=experience_years,
        location=location,
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    return {
        "candidate_id": candidate.id,
        "skills": skills,
        "experience_years": experience_years,
    }


@app.get("/candidates")
def list_candidates(db: Session = Depends(get_db)):
    return db.query(models.Candidate).all()


@app.post("/jobs")
def create_job(
    title: str,
    description: str,
    required_skills: str,
    min_experience: float,
    location: str,
    db: Session = Depends(get_db),
):
    """Create a job description. required_skills should be comma-separated."""
    job = models.JobDescription(
        title=title,
        description=description,
        required_skills=required_skills,
        min_experience=min_experience,
        location=location,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return {"job_id": job.id, "title": job.title}


@app.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    return db.query(models.JobDescription).all()


@app.post("/screen")
def screen_candidate(candidate_id: int, job_id: int, db: Session = Depends(get_db)):
    """Score a candidate against a job and generate LLM feedback."""
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id).first()
    job = db.query(models.JobDescription).filter(models.JobDescription.id == job_id).first()

    if not candidate or not job:
        raise HTTPException(status_code=404, detail="Candidate or job not found")

    candidate_skills = set(candidate.skills.split(",")) if candidate.skills else set()
    job_skills = set(job.required_skills.split(",")) if job.required_skills else set()

    skill_match_pct = len(candidate_skills & job_skills) / len(job_skills) if job_skills else 0
    experience_diff = (candidate.experience_years or 0) - (job.min_experience or 0)
    location_match = 1 if candidate.location == job.location else 0

    features = pd.DataFrame([{
        "skill_match_pct": skill_match_pct,
        "experience_diff": experience_diff,
        "location_match": location_match,
    }])

    fit_proba = fit_model.predict_proba(features)[0][1]
    fit_label = "Good Fit" if fit_proba >= 0.5 else "Not a Fit"

    feedback = generate_candidate_feedback(
        candidate_skills=list(candidate_skills),
        job_required_skills=list(job_skills),
        fit_score=fit_proba,
    )

    result = models.ScreeningResult(
        candidate_id=candidate.id,
        job_id=job.id,
        fit_score=float(fit_proba),
        fit_label=fit_label,
        feedback_report=feedback,
    )
    db.add(result)
    db.commit()
    db.refresh(result)

    return {
        "fit_score": round(float(fit_proba), 3),
        "fit_label": fit_label,
        "feedback": feedback,
    }


@app.post("/predict-salary/{candidate_id}")
def predict_salary(candidate_id: int, db: Session = Depends(get_db)):
    """Predict salary for a candidate based on their profile."""
    candidate = db.query(models.Candidate).filter(models.Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    skills = candidate.skills.split(",") if candidate.skills else []
    num_skills = len(skills)
    has_ml_skills = int(any(s in ML_SKILLS for s in skills))

    # Build feature row matching training columns (one-hot location)
    row = {
        "experience_years": candidate.experience_years or 0,
        "num_skills": num_skills,
        "has_ml_skills": has_ml_skills,
    }
    for loc in LOCATIONS:
        row[f"loc_{loc}"] = 1 if candidate.location == loc else 0

    features = pd.DataFrame([row])[salary_feature_cols]  # ensure correct column order
    predicted_salary = salary_model.predict(features)[0]

    prediction = models.SalaryPrediction(
        candidate_id=candidate.id,
        predicted_salary=float(predicted_salary),
        currency="INR",
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)

    return {
        "candidate_id": candidate.id,
        "predicted_salary_inr": round(float(predicted_salary), -3),
    }


@app.post("/jobs/{job_id}/summary")
def get_job_summary(job_id: int, db: Session = Depends(get_db)):
    """Generate an LLM summary for a job description."""
    job = db.query(models.JobDescription).filter(models.JobDescription.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    summary = generate_job_summary(job.description)
    return {"job_id": job.id, "summary": summary}