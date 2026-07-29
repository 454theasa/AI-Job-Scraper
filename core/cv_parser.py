import os
import json
import PyPDF2
from google import genai
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

def extract_text_from_pdf(pdf_file) -> str:
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

import hashlib

# --- Quality Gate Constraints ---
BANNED_STANDALONE_WORDS = {
    "process", "operations", "supply chain", "engineering",
    "management", "logistics", "production", "quality"
}

BANNED_SENIORITY_WORDS = {
    "junior", "senior", "manager", "lead", "head",
    "director", "intern", "trainee", "assistant"
}

FALLBACK_MUST_HAVES = [
    "Process Optimization",
    "Lean Manufacturing",
    "Supply Chain Coordination",
    "Quality Management"
]

# Simple in-memory cache keyed by MD5 of CV text
_SKILL_CACHE: Dict[str, Dict[str, Any]] = {}

def validate_skill_phrase(skill: str) -> bool:
    """Validates a single skill phrase against quality rules:
    - 2 to 4 words.
    - Not a standalone vague single word.
    - No seniority words anywhere in the phrase.
    """
    if not skill or not isinstance(skill, str):
        return False
    cleaned = skill.strip()
    words = cleaned.split()
    # Rule 1: Must be 2 to 4 words
    if len(words) < 2 or len(words) > 4:
        return False
    # Rule 2: Standalone vague single words are banned (handled by len < 2, but check lowercase)
    if cleaned.lower() in BANNED_STANDALONE_WORDS:
        return False
    # Rule 3: Seniority words are banned anywhere inside a skill phrase
    for w in words:
        if w.lower() in BANNED_SENIORITY_WORDS:
            return False
    return True

def sanitize_and_validate_skills(skills: List[str]) -> List[str]:

    """Filters a list of skills, returning only valid 2-4 word skill phrases for search."""
    valid_skills = []
    for s in skills or []:
        if validate_skill_phrase(s):
            valid_skills.append(s.strip())
    # De-duplicate while preserving order
    return list(dict.fromkeys(valid_skills))

class SkillExtractionSchema(BaseModel):
    must_have: List[str] = Field(description="4-6 Scoring-facing core functional domain phrases (2-4 words each).")
    nice_to_have: List[str] = Field(description="6-10 Scoring-facing tools, frameworks, software, languages, and technical competencies.")
    search_phrases: List[str] = Field(description="4-5 Search-facing natural job titles with seniority baked in (e.g., 'Supply Chain Intern').")

FALLBACK_MUST_HAVES = [
    "Process Optimization",
    "Lean Manufacturing",
    "Supply Chain Coordination",
    "Quality Management"
]

FALLBACK_NICE_TO_HAVES = [
    "Power BI",
    "SAP",
    "Six Sigma",
    "AutoCAD",
    "SQL",
    "Python",
    "ERP Systems",
    "Excel"
]

FALLBACK_SEARCH_PHRASES = [
    "Process Engineer",
    "Manufacturing Engineer",
    "Industrial Engineer",
    "Continuous Improvement Specialist",
    "Production Planner",
    "Quality Engineer"
]

def sanitize_scoring_keywords(keywords: List[str]) -> List[str]:
    """Sanitizes scoring keywords (nice_to_have).
    Allows single-word tools/technologies (e.g. Python, SAP, CAD, SQL).
    Validates basic non-emptiness and string formatting.
    """
    clean = []
    for kw in keywords or []:
        if isinstance(kw, str) and len(kw.strip()) >= 2:
            clean.append(kw.strip())
    # Preserves order while de-duplicating
    return list(dict.fromkeys(clean))

def validate_search_phrase(phrase: str, seniority: str) -> bool:
    if not phrase or not isinstance(phrase, str):
        return False
    cleaned = phrase.strip()
    words = cleaned.split()
    if len(words) < 2 or len(words) > 5:
        return False
    banned_attrs = {"experience", "background", "knowledge", "skills", "ability", "understanding"}
    if any(w.lower() in banned_attrs for w in words):
        return False
    if seniority in ["Internship", "Graduate", "Junior"]:
        seniority_markers = {"intern", "internship", "praktikum", "stage", "junior", "trainee", "entry", "berufseinsteiger", "absolvent", "student", "werkstudent"}
        if not any(w.lower() in seniority_markers for w in words):
            return False
    return True

def sanitize_search_phrases(phrases: List[str], seniority: str) -> List[str]:
    valid = []
    for p in phrases or []:
        if validate_search_phrase(p, seniority):
            valid.append(p.strip())
    return list(dict.fromkeys(valid))

def analyze_cv_with_gemini(cv_text: str, target_country: str, english_only: bool = False,
                           seniority_level: str = "Junior", graduate_junior_ratio: int = 50, api_key: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No Gemini API key configured. Add it in the sidebar '🔑 AI Settings' or set GEMINI_API_KEY.")
    model = model or "gemini-2.5-flash"

    # --- Cache Check ---
    cache_key_string = f"{cv_text}_{target_country}_{english_only}_{seniority_level}_{graduate_junior_ratio}"
    cv_hash = hashlib.md5(cache_key_string.encode('utf-8')).hexdigest()
    if cv_hash in _SKILL_CACHE:
        print(f"[DEBUG] Returning cached CV skills for hash {cv_hash[:8]}")
        return _SKILL_CACHE[cv_hash]

    client = genai.Client(api_key=api_key)
    
    language_instruction = "English phrases only" if english_only else f"A mix of local language for {target_country} and English phrases"
    prompt = f"""
    You are an expert technical recruiter. Your task is to extract Scoring-facing competencies and compose Search-facing phrases based on the candidate's CV and target seniority level.

    TARGET SENIORITY LEVEL: {seniority_level}
    LANGUAGE MODE: {language_instruction}

    1. 'must_have' (SCORING-FACING — 4 to 6 entries):
       - Each MUST be a precise 2-4 word functional skill phrase.
       - CRITICAL: Extract the pure functional domain! If the CV says 'Supply Chain Management', extract 'Supply Chain' or 'Logistics'. Do NOT append words like 'Support', 'Assistant', or 'Administration' to try and make it sound entry-level.
       - NEVER include seniority levels (BANNED: junior, senior, manager, lead, head, director, intern, trainee, assistant).
       - NEVER output single vague words as standalone skills.

    2. 'nice_to_have' (SCORING-FACING — 6 to 10 entries):
       - Specific tools, software, platforms, frameworks, methodologies, and technical competencies (e.g., 'Power BI', 'SAP', 'Python').
       - Single words are EXCELLENT here (e.g. 'Python', 'SAP', 'CAD', 'SQL'). DO NOT pad them into long phrases.

    3. 'search_phrases' (SEARCH-FACING — 4 to 5 highly fitting entries):
       - Generate natural job titles appropriate for the '{seniority_level}' level.
       - IF the level is 'Graduate', generate exactly {100 - graduate_junior_ratio}% '... Intern' titles AND {graduate_junior_ratio}% 'Junior...' titles.
       - CRITICAL: Do NOT double-up seniority terms (e.g., no 'Junior Intern').
       - CRITICAL: Do NOT use the word 'Graduate' as a title suffix (use 'Junior' or standard entry-level phrasing instead).
       - No attribute phrases (e.g., no 'Python Experience').

    CV Text:
    {cv_text}
    """
    
    from google.genai import types

    def _call_ai() -> Dict[str, Any]:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SkillExtractionSchema,
                temperature=0.0
            ),
        )
        parsed_data = response.parsed
        return {
            "must_have": parsed_data.must_have if parsed_data else [],
            "nice_to_have": parsed_data.nice_to_have if parsed_data else [],
            "search_phrases": parsed_data.search_phrases if parsed_data else []
        }

    # Attempt 1: Call AI with temp=0.0
    try:
        raw_res = _call_ai()
        valid_must = sanitize_and_validate_skills(raw_res.get("must_have", []))
        valid_nice = sanitize_scoring_keywords(raw_res.get("nice_to_have", []))
        valid_search = sanitize_search_phrases(raw_res.get("search_phrases", []), seniority_level)
        if len(valid_must) >= 2 and len(valid_nice) >= 4 and len(valid_search) >= 2:
            final_res = {"must_have": valid_must, "nice_to_have": valid_nice, "search_phrases": valid_search}
            if cv_hash:
                _SKILL_CACHE[cv_hash] = final_res
            return final_res
    except Exception as e:
        print(f"[WARN] First AI skill extraction attempt failed: {e}")

    # Retry 1: Second attempt if validation failed
    print("[WARN] Retrying AI skill extraction (Attempt 2)...")
    try:
        raw_res = _call_ai()
        valid_must = sanitize_and_validate_skills(raw_res.get("must_have", []))
        valid_nice = sanitize_scoring_keywords(raw_res.get("nice_to_have", []))
        valid_search = sanitize_search_phrases(raw_res.get("search_phrases", []), seniority_level)
        if len(valid_must) >= 2 and len(valid_nice) >= 4 and len(valid_search) >= 2:
            final_res = {"must_have": valid_must, "nice_to_have": valid_nice, "search_phrases": valid_search}
            if cv_hash:
                _SKILL_CACHE[cv_hash] = final_res
            return final_res
    except Exception as e:
        print(f"[WARN] Second AI skill extraction attempt failed: {e}")

    # Fallback: Hardcoded, known-good defaults for both search & scoring
    print("[WARN] Validation failed twice. Using hardcoded fallback skills.")
    fallback_res = {
        "must_have": FALLBACK_MUST_HAVES,
        "nice_to_have": FALLBACK_NICE_TO_HAVES,
        "search_phrases": FALLBACK_SEARCH_PHRASES
    }
    if cv_hash:
        _SKILL_CACHE[cv_hash] = fallback_res
    return fallback_res


