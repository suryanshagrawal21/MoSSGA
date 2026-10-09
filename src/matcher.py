import logging
import os

# Suppress HuggingFace symlinks warning on Windows before any HF imports
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Cache the SBERT model so we only load it once across the whole session
_sbert_model = None


def get_sbert_model():
    """Loads (or returns cached) Sentence-BERT model."""
    global _sbert_model
    if _sbert_model is None:
        try:
            logger.info("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.error("Failed to load SBERT model: %s", e)
            return None
    return _sbert_model


def calculate_similarity(cleaned_resumes, cleaned_jd):
    """Computes TF-IDF cosine similarity between each resume and the JD."""
    # Build a single corpus: JD first, then all resumes
    corpus = [cleaned_jd] + cleaned_resumes
    tfidf_matrix = TfidfVectorizer().fit_transform(corpus)

    jd_vector = tfidf_matrix[0]
    resume_vectors = tfidf_matrix[1:]

    scores = cosine_similarity(jd_vector, resume_vectors).flatten()
    return scores


def calculate_sbert_similarity(cleaned_resumes, cleaned_jd):
    """Computes semantic similarity using Sentence-BERT embeddings."""
    model = get_sbert_model()
    if model is None:
        return [0.0] * len(cleaned_resumes)  # graceful fallback

    jd_embedding = model.encode([cleaned_jd])
    resume_embeddings = model.encode(cleaned_resumes)

    scores = cosine_similarity(jd_embedding, resume_embeddings).flatten()
    return scores


def find_missing_skills(resume_text, jd_text, all_skills):
    """Returns skills that appear in the JD but are absent from the resume."""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    # First figure out which skills the JD actually asks for
    jd_required = [skill for skill in all_skills if skill in jd_lower]

    # Then check which of those the resume doesn't mention
    missing = [skill for skill in jd_required if skill not in resume_lower]

    return list(set(missing))
