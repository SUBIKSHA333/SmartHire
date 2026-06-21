import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from app.database import SessionLocal
from app import models
from app.services.resume_parser import extract_text_from_pdf, extract_skills, extract_experience_years
from app.services.llm_service import generate_candidate_feedback, generate_job_summary

st.set_page_config(page_title="SmartHire", page_icon="💼", layout="wide")

st.title("💼 SmartHire — AI Job Screening & Salary Prediction")
st.caption("XGBoost + LLM-powered candidate screening and salary prediction")

# Load ML models once
fit_model = joblib.load("ml/models/fit_model.joblib")
salary_model = joblib.load("ml/models/salary_model.joblib")
salary_feature_cols = joblib.load("ml/models/salary_feature_columns.joblib")

LOCATIONS = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Remote"]
ML_SKILLS = {"machine learning", "deep learning", "nlp", "tensorflow", "pytorch"}

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

page = st.sidebar.radio("Navigate", ["📄 Screen a Candidate", "💼 Job Postings", "📊 Analytics"])


def get_db():
    return SessionLocal()


# ---------------- SCREEN CANDIDATE PAGE ----------------
if page == "📄 Screen a Candidate":
    st.header("Upload Resume & Screen Against a Job")

    db = get_db()
    jobs = db.query(models.JobDescription).all()

    if not jobs:
        st.warning("No job postings yet. Create one in the 'Job Postings' tab first.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Candidate Name")
            email = st.text_input("Candidate Email")
            location = st.selectbox("Candidate Location", LOCATIONS)
        with col2:
            job_choice = st.selectbox("Select Job", [f"{j.id} - {j.title}" for j in jobs])
            resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

        if st.button("Screen Candidate"):
            if not (name and email and resume_file):
                st.error("Please fill in all fields and upload a resume.")
            else:
                with st.spinner("Parsing resume and analyzing fit..."):
                    file_path = os.path.join(UPLOAD_DIR, resume_file.name)
                    with open(file_path, "wb") as f:
                        f.write(resume_file.getbuffer())

                    resume_text = extract_text_from_pdf(file_path)
                    skills = extract_skills(resume_text)
                    experience_years = extract_experience_years(resume_text)

                    candidate = models.Candidate(
                        name=name, email=email, resume_text=resume_text,
                        skills=",".join(skills), experience_years=experience_years,
                        location=location,
                    )
                    db.add(candidate)
                    db.commit()
                    db.refresh(candidate)

                    job_id = int(job_choice.split(" - ")[0])
                    job = db.query(models.JobDescription).filter(models.JobDescription.id == job_id).first()

                    candidate_skills = set(skills)
                    job_skills = set(job.required_skills.split(",")) if job.required_skills else set()
                    skill_match_pct = len(candidate_skills & job_skills) / len(job_skills) if job_skills else 0
                    experience_diff = experience_years - (job.min_experience or 0)
                    location_match = 1 if location == job.location else 0

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

                    # Salary prediction
                    num_skills = len(skills)
                    has_ml_skills = int(any(s in ML_SKILLS for s in skills))
                    row = {"experience_years": experience_years, "num_skills": num_skills, "has_ml_skills": has_ml_skills}
                    for loc in LOCATIONS:
                        row[f"loc_{loc}"] = 1 if location == loc else 0
                    salary_features = pd.DataFrame([row])[salary_feature_cols]
                    predicted_salary = salary_model.predict(salary_features)[0]

                    result = models.ScreeningResult(
                        candidate_id=candidate.id, job_id=job.id,
                        fit_score=float(fit_proba), fit_label=fit_label, feedback_report=feedback,
                    )
                    db.add(result)
                    db.commit()

                st.success("Screening complete!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Fit Score", f"{fit_proba*100:.1f}%")
                col2.metric("Verdict", fit_label)
                col3.metric("Predicted Salary", f"₹{predicted_salary:,.0f}")

                st.subheader("Extracted Skills")
                st.write(", ".join(skills) if skills else "No skills detected")

                st.subheader("AI Feedback Report")
                st.write(feedback)

    db.close()

# ---------------- JOB POSTINGS PAGE ----------------
elif page == "💼 Job Postings":
    st.header("Manage Job Postings")

    with st.expander("➕ Create New Job"):
        title = st.text_input("Job Title")
        description = st.text_area("Job Description")
        required_skills = st.text_input("Required Skills (comma-separated, e.g. python,sql,machine learning)")
        min_experience = st.number_input("Minimum Experience (years)", min_value=0.0, step=0.5)
        job_location = st.selectbox("Job Location", LOCATIONS, key="job_loc")

        if st.button("Create Job"):
            if title and description and required_skills:
                db = get_db()
                job = models.JobDescription(
                    title=title, description=description,
                    required_skills=required_skills.lower(),
                    min_experience=min_experience, location=job_location,
                )
                db.add(job)
                db.commit()
                db.close()
                st.success(f"Job '{title}' created!")
            else:
                st.error("Please fill in title, description, and required skills.")

    st.subheader("Existing Jobs")
    db = get_db()
    jobs = db.query(models.JobDescription).all()
    db.close()

    if jobs:
        for job in jobs:
            with st.expander(f"{job.title} (ID: {job.id})"):
                st.write(f"**Location:** {job.location}")
                st.write(f"**Min Experience:** {job.min_experience} years")
                st.write(f"**Required Skills:** {job.required_skills}")
                st.write(f"**Description:** {job.description}")
                if st.button(f"Generate AI Summary", key=f"summary_{job.id}"):
                    with st.spinner("Generating summary..."):
                        summary = generate_job_summary(job.description)
                    st.info(summary)
    else:
        st.info("No jobs created yet.")

# ---------------- ANALYTICS PAGE ----------------
elif page == "📊 Analytics":
    st.header("Hiring Analytics")

    db = get_db()
    candidates = db.query(models.Candidate).all()
    screenings = db.query(models.ScreeningResult).all()
    salary_preds = db.query(models.SalaryPrediction).all()
    db.close()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Candidates", len(candidates))
    col2.metric("Total Screenings", len(screenings))
    col3.metric("Salary Predictions", len(salary_preds))

    if screenings:
        df = pd.DataFrame([{"fit_label": s.fit_label, "fit_score": s.fit_score} for s in screenings])
        fig = px.histogram(df, x="fit_label", title="Fit Outcomes Distribution")
        st.plotly_chart(fig, use_container_width=True)

    if candidates:
        all_skills = []
        for c in candidates:
            if c.skills:
                all_skills.extend(c.skills.split(","))
        if all_skills:
            skill_df = pd.DataFrame(all_skills, columns=["skill"])
            skill_counts = skill_df["skill"].value_counts().reset_index()
            skill_counts.columns = ["skill", "count"]
            fig2 = px.bar(skill_counts.head(15), x="skill", y="count", title="Top Skills Across Candidates")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No candidate data yet. Screen some candidates first!")