import random
import pandas as pd

random.seed(42)

SKILLS_POOL = [
    "python", "sql", "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "docker", "aws", "react", "java", "data visualization",
]

LOCATIONS = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Remote"]


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    rows = []
    for _ in range(n_samples):
        # Candidate profile
        candidate_skills = set(random.sample(SKILLS_POOL, random.randint(2, 8)))
        candidate_exp = round(random.uniform(0, 10), 1)
        candidate_loc = random.choice(LOCATIONS)

        # Job profile
        required_skills = set(random.sample(SKILLS_POOL, random.randint(3, 6)))
        min_experience = round(random.uniform(0, 6), 1)
        job_loc = random.choice(LOCATIONS)

        # Features
        skill_overlap = candidate_skills & required_skills
        skill_match_pct = len(skill_overlap) / len(required_skills) if required_skills else 0
        experience_diff = candidate_exp - min_experience
        location_match = 1 if candidate_loc == job_loc else 0

        # Rule-based label (simulates a recruiter's judgment), with added noise
        # to mimic real-world inconsistency in human hiring decisions
        rule_based_fit = skill_match_pct >= 0.6 and experience_diff >= -1
        is_good_fit = int(rule_based_fit)

        # 12% chance to flip the label - simulates noisy/inconsistent recruiter judgment
        if random.random() < 0.12:
            is_good_fit = 1 - is_good_fit

        rows.append({
            "skill_match_pct": round(skill_match_pct, 2),
            "experience_diff": experience_diff,
            "location_match": location_match,
            "is_good_fit": is_good_fit,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/fit_scoring_data.csv", index=False)
    print(f"Generated {len(df)} rows.")
    print(df.head())
    print(f"\nClass balance:\n{df['is_good_fit'].value_counts()}")