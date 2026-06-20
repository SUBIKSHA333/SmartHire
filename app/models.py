from sqlalchemy import Column, Integer, String, Float, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    resume_text = Column(Text, nullable=True)          # raw extracted text
    skills = Column(Text, nullable=True)                # comma-separated skills
    experience_years = Column(Float, nullable=True)
    location = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    screenings = relationship("ScreeningResult", back_populates="candidate")


class JobDescription(Base):
    __tablename__ = "job_descriptions"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    required_skills = Column(Text, nullable=True)       # comma-separated skills
    min_experience = Column(Float, nullable=True)
    location = Column(String(200), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    screenings = relationship("ScreeningResult", back_populates="job")


class ScreeningResult(Base):
    __tablename__ = "screening_results"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    job_id = Column(Integer, ForeignKey("job_descriptions.id"), nullable=False)
    fit_score = Column(Float, nullable=True)            # 0-1 probability from XGBoost classifier
    fit_label = Column(String(50), nullable=True)        # e.g. "Good Fit" / "Not a Fit"
    feedback_report = Column(Text, nullable=True)        # LLM-generated feedback
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    candidate = relationship("Candidate", back_populates="screenings")
    job = relationship("JobDescription", back_populates="screenings")


class SalaryPrediction(Base):
    __tablename__ = "salary_predictions"

    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    predicted_salary = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    candidate = relationship("Candidate")