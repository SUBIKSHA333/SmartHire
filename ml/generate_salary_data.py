import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

LOCATIONS = ["Bangalore", "Chennai", "Hyderabad", "Pune", "Remote"]

# Base salary multiplier per location (cost of living / market demand factor)
LOCATION_MULTIPLIER = {
    "Bangalore": 1.25,
    "Hyderabad": 1.15,
    "Pune": 1.10,
    "Chennai": 1.05,
    "Remote": 1.00,
}

ML_SKILLS = {"machine learning", "deep learning", "nlp", "tensorflow", "pytorch"}
SKILLS_POOL = [
    "python", "sql", "machine learning", "deep learning", "nlp",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "docker", "aws", "react", "java", "data visualization",
]


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    rows = []
    for _ in range(n_samples):
        experience_years = round(random.uniform(0, 12), 1)
        skills = set(random.sample(SKILLS_POOL, random.randint(2, 8)))
        num_skills = len(skills)
        has_ml_skills = int(len(skills & ML_SKILLS) > 0)
        location = random.choice(LOCATIONS)

        # Base salary formula (in INR, annual, in lakhs for readability internally then scaled)
        base = 3.5  # entry-level base in LPA (lakhs per annum)
        exp_component = experience_years * 1.8
        skill_component = num_skills * 0.3
        ml_bonus = 2.5 if has_ml_skills else 0

        salary_lpa = base + exp_component + skill_component + ml_bonus
        salary_lpa *= LOCATION_MULTIPLIER[location]

        # Add realistic noise (+/- 15%)
        noise_factor = np.random.normal(1.0, 0.12)
        salary_lpa *= max(noise_factor, 0.6)

        salary_inr = round(salary_lpa * 100000, -3)  # round to nearest 1000

        rows.append({
            "experience_years": experience_years,
            "num_skills": num_skills,
            "has_ml_skills": has_ml_skills,
            "location": location,
            "salary": salary_inr,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    # One-hot encode location for the model
    df_encoded = pd.get_dummies(df, columns=["location"], prefix="loc")
    df_encoded.to_csv("data/salary_data.csv", index=False)
    print(f"Generated {len(df)} rows.")
    print(df.head())
    print(f"\nSalary stats (INR):")
    print(df["salary"].describe())