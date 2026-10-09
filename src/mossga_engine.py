"""
MoSSGA — Main Orchestrator Engine
===================================
Top-level pipeline that ties together all MoSSGA components:
  1. Resume parsing → text, skills, education, experience, projects
  2. GitHub analysis → languages, projects, technologies
  3. Multi-modal fusion → unified skill profile
  4. Semantic skill matching against JD
  5. Skill gap detection with severity
  6. Recommendation generation
"""

import logging

from src.parser import extract_text
from src.preprocessing import clean_text, extract_skills, preprocess_text
from src.github_analyzer import analyze_github_profile
from src.semantic_skill_matcher import (
    semantic_skill_match,
    calculate_semantic_gap_score,
    resolve_skill_aliases,
)
from src.multimodal_fusion import (
    fuse_skill_profiles,
    generate_fusion_summary,
    compute_weighted_skill_score,
)
from src.career_intelligence import (
    extract_education_from_text,
    extract_experience_from_text,
    extract_projects_from_text,
    estimate_level,
    generate_skill_gap_and_roadmap,
    generate_career_guidance,
    generate_job_recommendations,
)
from src.hybrid_scorer import hybrid_scorer
from src.explainability import explainability_engine
from src.workforce_module import workforce_engine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Course / Resource Recommendation Database
# ---------------------------------------------------------------------------

COURSE_RECOMMENDATIONS = {
    "python": [
        {"title": "Python for Everybody", "platform": "Coursera", "level": "Beginner",
         "url": "https://www.coursera.org/specializations/python"},
        {"title": "Automate the Boring Stuff", "platform": "Udemy", "level": "Beginner",
         "url": "https://automatetheboringstuff.com/"},
    ],
    "javascript": [
        {"title": "The Complete JavaScript Course", "platform": "Udemy", "level": "Beginner",
         "url": "https://www.udemy.com/course/the-complete-javascript-course/"},
        {"title": "JavaScript.info Tutorial", "platform": "Free", "level": "Intermediate",
         "url": "https://javascript.info/"},
    ],
    "react": [
        {"title": "React — The Complete Guide", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/react-the-complete-guide/"},
        {"title": "Official React Tutorial", "platform": "Free", "level": "Beginner",
         "url": "https://react.dev/learn"},
    ],
    "machine learning": [
        {"title": "Machine Learning Specialization", "platform": "Coursera", "level": "Intermediate",
         "url": "https://www.coursera.org/specializations/machine-learning-introduction"},
        {"title": "Fast.ai Practical ML", "platform": "Free", "level": "Intermediate",
         "url": "https://www.fast.ai/"},
    ],
    "deep learning": [
        {"title": "Deep Learning Specialization", "platform": "Coursera", "level": "Advanced",
         "url": "https://www.coursera.org/specializations/deep-learning"},
        {"title": "PyTorch Lightning Tutorials", "platform": "Free", "level": "Intermediate",
         "url": "https://pytorch.org/tutorials/"},
    ],
    "docker": [
        {"title": "Docker Mastery", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/docker-mastery/"},
        {"title": "Docker Getting Started", "platform": "Free", "level": "Beginner",
         "url": "https://docs.docker.com/get-started/"},
    ],
    "kubernetes": [
        {"title": "Kubernetes for Developers", "platform": "Udemy", "level": "Advanced",
         "url": "https://www.udemy.com/course/kubernetes-certified-application-developer/"},
        {"title": "Kubernetes Basics", "platform": "Free", "level": "Intermediate",
         "url": "https://kubernetes.io/docs/tutorials/"},
    ],
    "sql": [
        {"title": "SQL for Data Science", "platform": "Coursera", "level": "Beginner",
         "url": "https://www.coursera.org/learn/sql-for-data-science"},
        {"title": "SQLBolt Interactive Lessons", "platform": "Free", "level": "Beginner",
         "url": "https://sqlbolt.com/"},
    ],
    "aws": [
        {"title": "AWS Cloud Practitioner Essentials", "platform": "AWS", "level": "Beginner",
         "url": "https://aws.amazon.com/training/learn-about/cloud-practitioner/"},
    ],
    "tensorflow": [
        {"title": "TensorFlow Developer Certificate", "platform": "Coursera", "level": "Intermediate",
         "url": "https://www.coursera.org/professional-certificates/tensorflow-in-practice"},
    ],
    "pytorch": [
        {"title": "PyTorch for Deep Learning", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/pytorch-for-deep-learning/"},
    ],
    "nlp": [
        {"title": "NLP Specialization", "platform": "Coursera", "level": "Advanced",
         "url": "https://www.coursera.org/specializations/natural-language-processing"},
    ],
    "node.js": [
        {"title": "The Complete Node.js Developer Course", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/the-complete-nodejs-developer-course-2/"},
    ],
    "typescript": [
        {"title": "Understanding TypeScript", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/understanding-typescript/"},
    ],
    "git": [
        {"title": "Git & GitHub Crash Course", "platform": "Free", "level": "Beginner",
         "url": "https://www.youtube.com/watch?v=RGOj5yH7evk"},
    ],
    "ci/cd": [
        {"title": "GitHub Actions — The Complete Guide", "platform": "Udemy", "level": "Intermediate",
         "url": "https://www.udemy.com/course/github-actions-the-complete-guide/"},
    ],
    "data visualization": [
        {"title": "Data Visualization with Python", "platform": "Coursera", "level": "Intermediate",
         "url": "https://www.coursera.org/learn/python-for-data-visualization"},
    ],
}

# Fallback recommendation for skills not in our database
DEFAULT_COURSE = {
    "title": "Search on Coursera / Udemy / YouTube",
    "platform": "Various",
    "level": "Beginner",
    "url": "https://www.coursera.org/",
}


# ---------------------------------------------------------------------------
# Recommendation Generator
# ---------------------------------------------------------------------------

def generate_mossga_recommendations(
    gap_result: dict,
    fusion_result: dict,
    target_role: str = "",
) -> dict:
    """
    Generates personalized recommendations based on MoSSGA analysis.

    Returns:
        {
            "skill_recommendations": [
                {"skill": str, "importance": str, "courses": [...], "action": str},
            ],
            "career_steps": [str],
            "improvement_areas": [str],
        }
    """
    missing_detailed = gap_result.get("missing_with_severity", [])

    skill_recs = []
    for item in missing_detailed[:8]:  # top 8 most important
        skill = item["skill"]
        severity = item["severity"]

        courses = COURSE_RECOMMENDATIONS.get(
            skill.lower(),
            [DEFAULT_COURSE.copy()]
        )

        # Personalise the action based on fusion data
        sources = fusion_result.get("skill_sources", {})
        if skill.lower() in {s.lower() for s in sources}:
            action = f"Strengthen your existing {skill} knowledge — it was found but doesn't match JD requirements closely enough."
        else:
            action = f"Learn {skill} — it's a {severity.lower()} requirement for this role."

        skill_recs.append({
            "skill": skill,
            "importance": severity,
            "courses": courses[:2],
            "action": action,
        })

    # Career improvement steps
    gap_severity = gap_result.get("gap_severity", "Medium")
    career_steps = _generate_career_steps(gap_severity, missing_detailed, target_role)

    # Improvement areas
    improvement_areas = _identify_improvement_areas(fusion_result, gap_result)

    return {
        "skill_recommendations": skill_recs,
        "career_steps": career_steps,
        "improvement_areas": improvement_areas,
    }


def _generate_career_steps(severity: str, missing: list, role: str) -> list:
    """Creates actionable career improvement steps."""
    role_name = role or "your target role"
    steps = []

    if severity == "Critical":
        steps.extend([
            f"🚨 Significant skill gap detected for {role_name}. Start with foundational skills first.",
            "📚 Dedicate 2-3 hours daily to structured online courses for the top 3 missing skills.",
            "🔨 Build 2-3 portfolio projects that demonstrate practical use of missing skills.",
            "🌐 Contribute to open-source projects using the required technologies.",
            "📝 Update your resume and GitHub profile as you acquire each new skill.",
        ])
    elif severity == "High":
        steps.extend([
            f"⚠️ Notable gaps exist for {role_name}, but your foundation is solid.",
            "📚 Focus on the top 3 critical missing skills through targeted courses.",
            "🔨 Build at least 1 end-to-end project showcasing the missing technologies.",
            "🌐 Star and contribute to GitHub repos using the required stack.",
            "💼 Apply to adjacent roles while upskilling towards your target.",
        ])
    elif severity == "Medium":
        steps.extend([
            f"🎯 Good progress towards {role_name}! A few targeted skills to bridge.",
            "📚 Take intermediate-level courses for 2-3 missing skills.",
            "🔨 Extend an existing project with the missing technologies.",
            "📝 Highlight relevant experience more prominently in your resume.",
        ])
    else:  # Low
        steps.extend([
            f"✅ Strong alignment with {role_name}! Minor refinements needed.",
            "📚 Explore advanced topics in your strongest skill areas.",
            "🏆 Pursue certifications for key technologies (AWS, Google Cloud, etc.).",
            "💼 Focus on interview preparation — your skills are competitive.",
        ])

    return steps


def _identify_improvement_areas(fusion_result: dict, gap_result: dict) -> list:
    """Identifies strategic improvement areas."""
    areas = []

    val = fusion_result.get("validation_summary", {})
    resume_only = val.get("resume_only", [])
    github_only = val.get("github_only", [])

    if resume_only:
        areas.append(
            f"🔗 **Portfolio Gap:** {len(resume_only)} skills listed on resume lack GitHub evidence. "
            f"Build projects to demonstrate: {', '.join(list(resume_only)[:3])}."
        )

    if github_only:
        areas.append(
            f"📄 **Resume Gap:** {len(github_only)} GitHub skills missing from resume. "
            f"Add these to your resume: {', '.join(list(github_only)[:3])}."
        )

    gap_score = gap_result.get("gap_score", 0)
    if gap_score > 50:
        areas.append(
            "⚡ **Skill Acquisition:** Focus on high-priority missing skills. "
            "Consider bootcamps or intensive courses for faster learning."
        )
    elif gap_score > 20:
        areas.append(
            "📈 **Skill Refinement:** Target specific advanced topics to close remaining gaps."
        )

    if not areas:
        areas.append(
            "🌟 **Excellent Position:** Your multi-modal profile is strong. "
            "Focus on deepening expertise and pursuing leadership opportunities."
        )

    return areas


# ---------------------------------------------------------------------------
# Full MoSSGA Pipeline
# ---------------------------------------------------------------------------

def run_mossga_pipeline(
    resume_text: str = "",
    github_input: str = "",
    jd_text: str = "",
    target_role: str = "",
) -> dict:
    """
    Runs the complete MoSSGA analysis pipeline.

    Accepts any combination of inputs (resume, GitHub, JD).
    At minimum one of resume_text or github_input must be provided.

    Returns:
        Comprehensive MoSSGAReport dict with all analysis results.
    """
    report = {
        "status": "success",
        "inputs": {
            "has_resume": bool(resume_text),
            "has_github": bool(github_input),
            "has_jd": bool(jd_text),
        },
        "resume_analysis": None,
        "github_analysis": None,
        "fusion_result": None,
        "fusion_summary": "",
        "semantic_match": None,
        "gap_analysis": None,
        "hybrid_scoring": None,
        "explainable_insights": None,
        "career_path": None,
        "recommendations": None,
        "mossga_score": 0.0,
    }

    # ── Step 1: Resume Analysis ─────────────────────────────────
    resume_skills = []
    if resume_text:
        resume_skills = extract_skills(resume_text)
        education = extract_education_from_text(resume_text)
        experience = extract_experience_from_text(resume_text)
        projects = extract_projects_from_text(resume_text)

        report["resume_analysis"] = {
            "skills": resume_skills,
            "education": education,
            "experience": experience,
            "projects": projects,
            "skill_count": len(resume_skills),
        }

    # ── Step 2: GitHub Analysis ─────────────────────────────────
    github_skills = []
    if github_input:
        gh_analysis = analyze_github_profile(github_input)
        if gh_analysis and gh_analysis.get("status") == "success":
            github_skills = gh_analysis.get("skills", [])
            report["github_analysis"] = gh_analysis
        elif gh_analysis:
            report["github_analysis"] = gh_analysis  # no_repos status
        else:
            report["github_analysis"] = {"status": "failed", "skills": []}

    # ── Step 3: Multi-Modal Fusion ──────────────────────────────
    # Default all_skills to union of both sources before fusion
    all_skills = list(set(resume_skills + github_skills))

    if resume_skills or github_skills:
        fusion = fuse_skill_profiles(resume_skills, github_skills)
        fusion_summary = generate_fusion_summary(fusion)
        all_skills = fusion["fused_skills"]

        report["fusion_result"] = fusion
        report["fusion_summary"] = fusion_summary
    else:
        # Create a minimal fusion result for compatibility
        report["fusion_result"] = {
            "fused_skills": [],
            "skill_sources": {},
            "confidence_scores": {},
            "validation_summary": {"both_sources": [], "resume_only": [], "github_only": []},
            "fusion_stats": {
                "total_unique": 0, "validated_count": 0,
                "validation_rate": 0, "resume_skill_count": 0, "github_skill_count": 0,
            },
        }

    # ── Step 4: Semantic Skill Matching ─────────────────────────
    jd_skills = extract_skills(jd_text) if jd_text else []

    if all_skills and jd_skills:
        match_result = semantic_skill_match(all_skills, jd_skills)
        report["semantic_match"] = match_result

        # ── Step 5: Skill Gap Analysis ──────────────────────────
        gap_result = calculate_semantic_gap_score(match_result)
        report["gap_analysis"] = gap_result

        # ── Step 5.5: Hybrid Scoring ────────────────────────────
        cand_skill_set = set(all_skills)
        jd_skill_set = set(jd_skills)
        hybrid_results = hybrid_scorer.compute_hybrid_score(cand_skill_set, jd_skill_set, match_result)
        report["hybrid_scoring"] = hybrid_results
        report["mossga_score"] = hybrid_results["hybrid_score"]
        
        # ── Step 5.6: Explainable AI Insights ───────────────────
        insights = explainability_engine.generate_explanations(
            gap_result.get("missing_with_severity", []), 
            cand_skill_set, 
            target_role or "Target Role"
        )
        report["explainable_insights"] = insights
        
        # ── Step 5.7: Workforce Informatics (Career Path) ───────
        career_path = workforce_engine.predict_career_path(cand_skill_set, target_role)
        progression = workforce_engine.model_skill_progression(list(jd_skill_set))
        report["career_path"] = {
            "next_skills": career_path,
            "progression_model": progression
        }

        # ── Step 6: Recommendations ─────────────────────────────
        recommendations = generate_mossga_recommendations(
            gap_result,
            report["fusion_result"],
            target_role=target_role,
        )
        report["recommendations"] = recommendations

    elif all_skills:
        # No JD — still provide basic analysis
        level = estimate_level(all_skills)
        role = target_role or "General"
        missing, roadmap = generate_skill_gap_and_roadmap(all_skills, role)
        recs = generate_job_recommendations(all_skills, role)

        report["gap_analysis"] = {
            "gap_score": 0,
            "gap_severity": "N/A",
            "missing_with_severity": [],
        }
        report["recommendations"] = {
            "skill_recommendations": [],
            "career_steps": [generate_career_guidance(level, role)],
            "improvement_areas": [],
        }

    return report
