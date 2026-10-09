"""
MoSSGA — GitHub Profile Analyzer
=================================
Fetches a GitHub user's public repositories via the GitHub REST API
and extracts programming languages, technologies, project complexity,
and practical skills for multi-modal fusion with resume data.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict

import requests

logger = logging.getLogger(__name__)

# Suppress HuggingFace symlinks warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# GitHub REST API base URL
GITHUB_API = "https://api.github.com"

# Optional: set GITHUB_TOKEN env var to raise rate limit from 60 to 5000 req/hr
_GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

# Map GitHub-reported languages to standardised skill names
LANGUAGE_SKILL_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "c++": "c++",
    "c#": "c#",
    "c": "c",
    "go": "go",
    "rust": "rust",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin",
    "scala": "scala",
    "r": "r",
    "dart": "dart",
    "shell": "bash",
    "html": "html",
    "css": "css",
    "jupyter notebook": "python",
    "hcl": "terraform",
    "dockerfile": "docker",
    "makefile": "linux",
}

# Map common GitHub repo topics to skill names
TOPIC_SKILL_MAP = {
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "artificial-intelligence": "machine learning",
    "data-science": "data science",
    "natural-language-processing": "nlp",
    "computer-vision": "computer vision",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "scikit-learn": "scikit-learn",
    "react": "react",
    "reactjs": "react",
    "nextjs": "next.js",
    "vue": "vue",
    "vuejs": "vue",
    "angular": "angular",
    "nodejs": "node.js",
    "express": "express",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "docker": "docker",
    "kubernetes": "kubernetes",
    "aws": "aws",
    "azure": "azure",
    "gcp": "gcp",
    "devops": "devops",
    "ci-cd": "ci/cd",
    "mongodb": "mongodb",
    "postgresql": "postgresql",
    "mysql": "mysql",
    "sql": "sql",
    "graphql": "graphql",
    "rest-api": "rest apis",
    "api": "rest apis",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "tableau": "tableau",
    "blockchain": "blockchain",
    "web-development": "web development",
    "android": "android",
    "ios": "ios",
    "flutter": "flutter",
    "react-native": "react native",
    "selenium": "selenium",
    "testing": "testing",
    "agile": "agile",
    "microservices": "microservices",
    "spring-boot": "spring boot",
}

# Infer technologies from README / description keywords
README_TECH_PATTERNS = {
    "docker": r"\bdocker\b",
    "kubernetes": r"\bkubernetes\b|\bk8s\b",
    "aws": r"\baws\b|\bamazon web services\b",
    "gcp": r"\bgcp\b|\bgoogle cloud\b",
    "azure": r"\bazure\b",
    "ci/cd": r"\bci/cd\b|\bgithub actions\b|\bjenkins\b|\bcircle ?ci\b",
    "mongodb": r"\bmongodb\b|\bmongoose\b",
    "postgresql": r"\bpostgresql\b|\bpostgres\b",
    "redis": r"\bredis\b",
    "graphql": r"\bgraphql\b",
    "rest apis": r"\brest ?api\b",
    "tensorflow": r"\btensorflow\b|\btf\b",
    "pytorch": r"\bpytorch\b",
    "scikit-learn": r"\bscikit-learn\b|\bsklearn\b",
    "react": r"\breact\b|\breactjs\b",
    "next.js": r"\bnext\.?js\b",
    "vue": r"\bvue\b|\bvuejs\b",
    "angular": r"\bangular\b",
    "node.js": r"\bnode\.?js\b",
    "express": r"\bexpress\.?js\b|\bexpress\b",
    "django": r"\bdjango\b",
    "flask": r"\bflask\b",
    "fastapi": r"\bfastapi\b",
    "selenium": r"\bselenium\b",
    "pandas": r"\bpandas\b",
    "numpy": r"\bnumpy\b",
    "machine learning": r"\bmachine learning\b|\bml\b",
    "deep learning": r"\bdeep learning\b|\bdl\b",
    "nlp": r"\bnlp\b|\bnatural language processing\b",
}


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _extract_username(github_input: str) -> str:
    """Extracts a GitHub username from a URL or plain string."""
    if not github_input:
        return ""

    github_input = github_input.strip().rstrip("/")

    # Handle full URLs: https://github.com/username or https://github.com/username/repo
    match = re.match(r"https?://github\.com/([a-zA-Z0-9_-]+)", github_input)
    if match:
        return match.group(1)

    # Handle github.com/username (without protocol)
    match = re.match(r"github\.com/([a-zA-Z0-9_-]+)", github_input)
    if match:
        return match.group(1)

    # Assume it's just a username
    if re.match(r"^[a-zA-Z0-9_-]+$", github_input):
        return github_input

    return ""


def _fetch_json(url: str, params: dict = None) -> dict | list | None:
    """Makes a GET request and returns parsed JSON, or None on failure."""
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "MoSSGA-SkillAnalyzer/1.0",
    }
    if _GITHUB_TOKEN:
        headers["Authorization"] = f"token {_GITHUB_TOKEN}"

    try:
        resp = requests.get(url, params=params, timeout=10, headers=headers)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            logger.warning("GitHub API rate-limit hit (403). Try again later or set GITHUB_TOKEN env var.")
            return None
        elif resp.status_code == 404:
            logger.warning("GitHub user/resource not found (404).")
            return None
        else:
            logger.warning("GitHub API returned %d for %s", resp.status_code, url)
            return None
    except requests.RequestException as e:
        logger.error("GitHub API request failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Core analysis functions
# ---------------------------------------------------------------------------

def fetch_user_profile(username: str) -> dict | None:
    """Fetches basic profile data for a GitHub user."""
    data = _fetch_json(f"{GITHUB_API}/users/{username}")
    if data is None:
        return None

    return {
        "username": data.get("login", username),
        "name": data.get("name", ""),
        "bio": data.get("bio", ""),
        "public_repos": data.get("public_repos", 0),
        "followers": data.get("followers", 0),
        "following": data.get("following", 0),
        "created_at": data.get("created_at", ""),
        "avatar_url": data.get("avatar_url", ""),
        "html_url": data.get("html_url", ""),
    }


def fetch_repos(username: str, max_repos: int = 30) -> list:
    """Fetches public repos for a user, sorted by most recently updated."""
    repos = _fetch_json(
        f"{GITHUB_API}/users/{username}/repos",
        params={"sort": "updated", "direction": "desc", "per_page": min(max_repos, 100)},
    )
    if not repos:
        return []

    parsed = []
    for repo in repos[:max_repos]:
        if repo.get("fork"):
            continue  # skip forks for authenticity

        parsed.append({
            "name": repo.get("name", ""),
            "description": repo.get("description", "") or "",
            "language": repo.get("language", ""),
            "topics": repo.get("topics", []),
            "stargazers_count": repo.get("stargazers_count", 0),
            "forks_count": repo.get("forks_count", 0),
            "size": repo.get("size", 0),
            "created_at": repo.get("created_at", ""),
            "updated_at": repo.get("updated_at", ""),
            "html_url": repo.get("html_url", ""),
        })

    return parsed


def fetch_repo_languages(username: str, repo_name: str) -> dict:
    """Fetches language breakdown (bytes) for a specific repo."""
    data = _fetch_json(f"{GITHUB_API}/repos/{username}/{repo_name}/languages")
    return data if data else {}


# ---------------------------------------------------------------------------
# Skill extraction from GitHub data
# ---------------------------------------------------------------------------

def extract_languages_and_skills(username: str, repos: list) -> dict:
    """
    Aggregates language usage across repos and maps to standardised skills.

    Returns:
        {
            "language_bytes": {"Python": 125000, ...},
            "language_percentages": {"Python": 45.2, ...},
            "inferred_skills": ["python", "javascript", ...],
        }
    """
    total_bytes = defaultdict(int)

    for repo in repos[:15]:  # limit API calls
        langs = fetch_repo_languages(username, repo["name"])
        for lang, byte_count in langs.items():
            total_bytes[lang] += byte_count

    grand_total = sum(total_bytes.values()) or 1

    language_pct = {
        lang: round(count / grand_total * 100, 1)
        for lang, count in sorted(total_bytes.items(), key=lambda x: -x[1])
    }

    # Map to skills
    inferred_skills = set()
    for lang in total_bytes:
        skill = LANGUAGE_SKILL_MAP.get(lang.lower())
        if skill:
            inferred_skills.add(skill)

    return {
        "language_bytes": dict(total_bytes),
        "language_percentages": language_pct,
        "inferred_skills": list(inferred_skills),
    }


def extract_topic_skills(repos: list) -> list:
    """Extracts skills from GitHub repo topics."""
    topic_skills = set()
    for repo in repos:
        for topic in repo.get("topics", []):
            topic_lower = topic.lower()
            if topic_lower in TOPIC_SKILL_MAP:
                topic_skills.add(TOPIC_SKILL_MAP[topic_lower])
    return list(topic_skills)


def extract_description_skills(repos: list) -> list:
    """Infers technologies from repo descriptions using regex patterns."""
    desc_skills = set()
    for repo in repos:
        desc = (repo.get("description", "") or "").lower()
        name = repo.get("name", "").lower().replace("-", " ").replace("_", " ")
        combined = desc + " " + name

        for skill, pattern in README_TECH_PATTERNS.items():
            if re.search(pattern, combined, re.IGNORECASE):
                desc_skills.add(skill)

    return list(desc_skills)


def calculate_project_complexity(repos: list) -> list:
    """
    Scores each repo's complexity based on multiple signals.

    Complexity = w1*log(size) + w2*stars + w3*forks + w4*topic_count
    """
    import math

    scored_repos = []
    for repo in repos:
        size = max(repo.get("size", 1), 1)
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        topics = len(repo.get("topics", []))

        complexity = (
            0.3 * min(math.log(size, 10), 5)  # size in log scale, cap at 5
            + 0.3 * min(stars, 50) / 50         # star contribution, cap at 50
            + 0.2 * min(forks, 20) / 20          # fork contribution, cap at 20
            + 0.2 * min(topics, 5) / 5           # topic diversity, cap at 5
        ) * 100

        scored_repos.append({
            **repo,
            "complexity_score": round(complexity, 1),
        })

    scored_repos.sort(key=lambda r: r["complexity_score"], reverse=True)
    return scored_repos


# ---------------------------------------------------------------------------
# Main analysis entry point
# ---------------------------------------------------------------------------

def analyze_github_profile(github_input: str) -> dict | None:
    """
    Full GitHub profile analysis pipeline.

    Args:
        github_input: GitHub URL or username.

    Returns:
        Structured analysis dict, or None if profile cannot be fetched.
    """
    username = _extract_username(github_input)
    if not username:
        logger.warning("Could not parse GitHub username from: %s", github_input)
        return None

    # Fetch profile
    profile = fetch_user_profile(username)
    if profile is None:
        return None

    # Fetch repos
    repos = fetch_repos(username, max_repos=30)
    if not repos:
        return {
            "profile": profile,
            "repos_analyzed": 0,
            "languages": {},
            "skills": [],
            "projects": [],
            "complexity_scores": [],
            "contribution_metrics": {
                "total_repos": 0,
                "total_stars": 0,
                "language_diversity": 0,
                "avg_project_complexity": 0,
            },
            "status": "no_repos",
        }

    # Extract skills from multiple signals
    lang_data = extract_languages_and_skills(username, repos)
    topic_skills = extract_topic_skills(repos)
    desc_skills = extract_description_skills(repos)

    # Merge all skills (deduplicated)
    all_skills = list(set(
        lang_data["inferred_skills"] + topic_skills + desc_skills
    ))

    # Score project complexity
    scored_repos = calculate_project_complexity(repos)

    # Contribution metrics
    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    total_forks = sum(r.get("forks_count", 0) for r in repos)
    avg_complexity = (
        sum(r["complexity_score"] for r in scored_repos) / len(scored_repos)
        if scored_repos else 0
    )

    return {
        "profile": profile,
        "repos_analyzed": len(repos),
        "languages": lang_data["language_percentages"],
        "language_bytes": lang_data["language_bytes"],
        "skills": all_skills,
        "skill_sources": {
            "from_languages": lang_data["inferred_skills"],
            "from_topics": topic_skills,
            "from_descriptions": desc_skills,
        },
        "projects": scored_repos[:10],  # top 10 by complexity
        "contribution_metrics": {
            "total_repos": len(repos),
            "total_stars": total_stars,
            "total_forks": total_forks,
            "language_diversity": len(lang_data["language_percentages"]),
            "avg_project_complexity": round(avg_complexity, 1),
        },
        "status": "success",
    }
