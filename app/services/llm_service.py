import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(prompt: str, system_message: str = "You are a helpful HR assistant.") -> str:
    """Send a prompt to the Groq LLM and return the response text."""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=600,
    )
    return response.choices[0].message.content


def generate_candidate_feedback(candidate_skills: list, job_required_skills: list, fit_score: float) -> str:
    """Generate a personalized feedback report for a candidate based on their screening result."""
    missing_skills = [s for s in job_required_skills if s not in candidate_skills]
    matched_skills = [s for s in job_required_skills if s in candidate_skills]

    prompt = f"""A candidate was screened for a job with the following results:
- Matched skills: {', '.join(matched_skills) if matched_skills else 'None'}
- Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}
- Fit score: {fit_score * 100:.0f}%

Write a short, encouraging, professional feedback report (3-4 sentences) for this candidate. 
Mention their strengths first, then what skills would help them improve their fit for similar roles."""

    return ask_llm(prompt, system_message="You are a supportive HR assistant giving constructive candidate feedback.")


def generate_job_summary(job_description: str) -> str:
    """Generate a concise summary of a job description."""
    prompt = f"""Summarize the following job description in 2-3 clear, concise sentences, 
highlighting the role, key requirements, and what makes it stand out:

{job_description}"""

    return ask_llm(prompt, system_message="You are an HR assistant who writes clear, concise job summaries.")


if __name__ == "__main__":
    answer = ask_llm("Say hello and confirm you're working as SmartHire's AI assistant.")
    print("--- LLM Response ---")
    print(answer)

    print("\n--- Sample Candidate Feedback ---")
    feedback = generate_candidate_feedback(
        candidate_skills=["python", "sql", "pandas"],
        job_required_skills=["python", "sql", "machine learning", "docker"],
        fit_score=0.55,
    )
    print(feedback)

    print("\n--- Sample Job Summary ---")
    summary = generate_job_summary(
        "We are looking for a Data Scientist with 2+ years experience in Python, SQL, "
        "and machine learning. The ideal candidate has worked with XGBoost or scikit-learn, "
        "is comfortable with cloud deployment, and can communicate insights to stakeholders. "
        "Remote-friendly role based in Bangalore."
    )
    print(summary) 