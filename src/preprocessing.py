import re
from collections import Counter

import spacy

# Load the spaCy language model for tokenisation and NER
nlp = spacy.load("en_core_web_sm")


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def clean_text(text):
    """Strips URLs, emails, special chars, and extra whitespace."""
    if not text:
        return ""

    text = re.sub(r'http\S+', '', text)       # drop URLs
    text = re.sub(r'\S+@\S+', '', text)       # drop email addresses
    text = re.sub(r'[^\w\s]', ' ', text)       # keep only alphanumeric + spaces
    text = ' '.join(text.split())              # collapse whitespace

    return text.lower()


def preprocess_text(text):
    """Lemmatises tokens and removes stop words / punctuation."""
    if not nlp:
        return text.split()  # basic fallback when spaCy unavailable

    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Skills Database & Extraction
# ---------------------------------------------------------------------------

# Curated list of common tech and soft skills for keyword matching
COMMON_SKILLS = {
    # Programming languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "html", "css",
    "go", "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "dart",
    # Web frameworks
    "react", "angular", "vue", "node.js", "express", "django", "flask", "fastapi",
    "next.js", "spring boot", "tailwind css",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "nosql", "graphql",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "ci/cd", "terraform", "github actions", "jenkins",
    # Data & ML
    "machine learning", "deep learning", "nlp", "computer vision", "data science",
    "tensorflow", "pytorch", "keras", "scikit-learn",
    "pandas", "numpy", "matplotlib", "seaborn", "tableau",
    "power bi", "excel", "data visualization", "statistics",
    # Big data
    "spark", "hadoop", "airflow", "dbt", "kafka",
    # Mobile
    "android", "ios", "flutter", "react native",
    # Testing & tools
    "selenium", "testing", "rest apis", "microservices", "system design",
    # Soft skills
    "communication", "teamwork", "leadership", "agile", "scrum", "project management",
}

# Maps abbreviations and synonyms to canonical skill names
SKILL_ALIASES = {
    "js": "javascript", "ts": "typescript", "py": "python",
    "ml": "machine learning", "dl": "deep learning", "ai": "artificial intelligence",
    "cpp": "c++", "csharp": "c#", "golang": "go",
    "k8s": "kubernetes", "tf": "tensorflow", "sklearn": "scikit-learn",
    "reactjs": "react", "nodejs": "node.js", "expressjs": "express",
    "postgres": "postgresql", "mongo": "mongodb",
    "natural language processing": "nlp", "cv": "computer vision",
}


def extract_skills(text):
    """Finds known skills in the text using whole-word regex matching."""
    if not text:
        return []

    text_lower = text.lower()
    found_skills = []

    for skill in COMMON_SKILLS:
        # \b ensures we match "python" but not "pythonic"
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_skills.append(skill)

    return list(set(found_skills))


# ---------------------------------------------------------------------------
# Named Entity Recognition
# ---------------------------------------------------------------------------

def extract_entities(text):
    """Pulls out organisations, people, and locations via spaCy NER."""
    if not nlp:
        return {}

    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PERSON", "GPE"]:
            entities.setdefault(ent.label_, []).append(ent.text)

    return entities
