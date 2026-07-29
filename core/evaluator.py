import os
import json
import re
import hashlib
import numpy as np
from sqlmodel import Session, select
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Optional
from google import genai
from google.genai import types
from .models import JobListing, JobStatus, Keyword, KeywordCategory
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize local embedding model for Phase 4
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# --- Tunable batching constants ---
LANGUAGE_TRAP_CHUNK_SIZE = 20      # jobs per Gemini call
LANGUAGE_TRAP_TEXT_LIMIT = 4000    # chars of each posting sent for the language check
RELEVANCE_RESCUE_CHUNK_SIZE = 20   # rejected jobs per Gemini relevance-rescue call
RELEVANCE_RESCUE_TEXT_LIMIT = 4000 # chars of each posting sent for the relevance check
AI_RESCUE_SCORE = 10.0             # fixed score assigned to jobs recovered by the AI relevance rescue
MIN_MATCH_SCORE = 35.0             # floor under the OR scoring scale (~best must-have similarity 0.5); below -> REJECTED

# Exact diagnostic string for the ONLY rejection type eligible for the AI rescue step.
# (Hard-nos, language traps, gate-1 junk and aggregator trash must never be rescued.)
NO_MUST_HAVE_REASON = "Rejected: Did not contain any core functional domains (Must Haves) via vector matching."

class ExpatFilterSchema(BaseModel):
    requires_local_language: bool = Field(description="True if the role enforces a strict local language requirement.")
    reasoning_snippet: str = Field(description="A brief explanation.")

class LanguageTrapItem(BaseModel):
    job_id: int = Field(description="The ID of the job posting being analyzed.")
    requires_local_language: bool = Field(description="True ONLY if the posting explicitly demands a local (non-English) language skill.")
    evidence: str = Field(description="The exact sentence quoted from the posting that imposes the language requirement. Empty string if none.")

class LanguageTrapBatch(BaseModel):
    results: List[LanguageTrapItem]

# --- Free local language heuristic (no API cost) ---
_STOPWORDS = {
    "en": {"the", "and", "with", "you", "your", "we", "are", "for", "our", "this", "that", "will", "have", "from", "who", "what", "they", "their"},
    "de": {"der", "die", "das", "und", "mit", "wir", "sie", "für", "ist", "den", "dem", "des", "sich", "werden", "ihre", "bei", "von", "zu", "auch", "als"},
    "fr": {"le", "la", "les", "des", "et", "avec", "nous", "vous", "pour", "est", "une", "dans", "qui", "sur", "par", "être", "au", "aux", "cette"},
    "nl": {"het", "een", "met", "wij", "voor", "zijn", "van", "op", "aan", "dat", "ook", "als", "naar", "bij", "uit", "deze"},
    "tr": {"bir", "için", "ile", "olarak", "olan", "gibi", "çok", "daha", "kadar", "sonra", "önce", "ancak"},
}

def detect_language_hint(text: str) -> str:
    """Free local heuristic. Returns 'en' when the text is clearly English,
    otherwise 'ambiguous' (send to Gemini for a real check)."""
    if not text:
        return "ambiguous"
    words = re.findall(r"[a-zA-ZÀ-ÿğüşöçıİ]+", text[:2000].lower())
    if not words:
        return "ambiguous"
    counts = {lang: sum(1 for w in words if w in sw) for lang, sw in _STOPWORDS.items()}
    en = counts["en"]
    others_max = max((c for lang, c in counts.items() if lang != "en"), default=0)
    if en >= 5 and en > others_max:
        return "en"
    return "ambiguous"

def truncate_head_tail(text: str, limit: int) -> str:
    """First half + last half of the text at the same token cost as a plain
    truncation. Requirement sections (language, experience, degree, location)
    almost always sit at the END of a posting — a head-only cut blinds every
    AI gate to them. (Root cause of the valuemize miss, 2026-07-24.)"""
    if not text or len(text) <= limit:
        return text or ""
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]

_CONFLICTING_TITLE_DOMAINS = [
    (
        [r"\bnetwork\s+(?:engineer|administrator|technician|specialist)\b", r"\bit\s+network\b", r"\btelecom\b", r"\bcisco\b"],
        ["network", "telecom", "cisco"],
        "IT/Network Engineering"
    ),
    (
        [r"\bsales\s+(?:representative|manager|specialist|executive|consultant)\b", r"\baccount\s+manager\b"],
        ["sales", "account management"],
        "Sales / Account Management"
    ),
    (
        [r"\bfinancial\s+accountant\b", r"\btax\s+specialist\b", r"\bauditor\b"],
        ["accounting", "tax", "audit"],
        "Finance / Accounting"
    ),
    (
        [r"\bnurse\b", r"\bphysician\b", r"\bmedical\s+doctor\b"],
        ["nursing", "medicine", "healthcare"],
        "Clinical Healthcare"
    )
]

def check_title_domain_mismatch(title: str, candidate_keywords: List[str]) -> Tuple[bool, str]:
    """Checks if a job title explicitly belongs to a domain that conflicts with the candidate's keywords."""
    if not title:
        return False, ""
    title_lower = title.lower()
    kw_str = " ".join(candidate_keywords).lower()
    for patterns, req_kws, domain_name in _CONFLICTING_TITLE_DOMAINS:
        if not any(rk in kw_str for rk in req_kws):
            for pat in patterns:
                if re.search(pat, title_lower):
                    return True, f"Title domain mismatch ({domain_name} title for non-{domain_name} candidate)"
    return False, ""

def gate_1_metadata_check(job: JobListing) -> bool:
    """Deterministic check for expired or invalid content."""
    text = job.full_text.lower() if job.full_text else ""
    if "404 not found" in text or "security check" in text or "cloudflare" in text or "access denied" in text:
        return False
    return True


_LANGUAGE_TRAP_RULES = """
You are checking job postings for STRICT local-language requirements.

For EACH posting below, decide whether it EXPLICITLY demands proficiency in a local (non-English) language.

Rules:
1. A posting merely WRITTEN in German, French, Dutch or Turkish is NOT a language requirement. Many international companies post locally but work in English.
2. Only an explicit demand counts (e.g. "fluent German required", "German at least B2", "verhandlungssicheres Deutsch", "excellent command of French, mandatory"). Vague nice-to-haves ("German is a plus", "French an advantage") do NOT count.
3. Ignore programming languages and geographical location names.
4. If you set requires_local_language to true, you MUST quote the exact sentence from the posting in 'evidence'. If you cannot quote one, set it to false.
5. When in doubt, set requires_local_language to false.
"""

SURVIVAL_LANGUAGE_RULES = """
You are checking job postings for language requirements. The candidate speaks English and basic local language (A2-B1) and wants jobs they can actually do.

For EACH posting decide: does it EXPLICITLY demand a local (non-English) language at a level ABOVE B1?

Rules:
1. No language requirement mentioned at all -> requires_local_language = false.
2. Basic requirements are FINE: "A1", "A2", "B1", "Grundkenntnisse", "basic German", "German is a plus", "erste Deutschkenntnisse" -> false.
3. Only STRONG demands count: "fluent German required", "C1/C2", "verhandlungssicheres Deutsch", "sehr gute Deutschkenntnisse in Wort und Schrift", "native speaker", "excellent command of French, mandatory" -> true.
4. A posting merely WRITTEN in German, French, Dutch or Turkish is NOT a language requirement by itself.
5. Ignore programming languages and geographical location names.
6. If you set requires_local_language to true, you MUST quote the exact sentence from the posting in 'evidence'. If you cannot quote one, set it to false.
7. When in doubt, set requires_local_language to false.
"""

def gate_2_language_trap(job_text: str, target_country: str, api_key: Optional[str] = None, model: Optional[str] = None) -> Tuple[bool, str]:
    """Single-posting language check (used by the Laboratory tab fallback)."""
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return False, "Skipped language trap due to missing API key."
    client = genai.Client(api_key=api_key)
    model = model or "gemini-2.5-flash"

    prompt = _LANGUAGE_TRAP_RULES + f"""
The posting below is a single job. Treat its JOB_ID as 0.

Job Text:
{truncate_head_tail(job_text, LANGUAGE_TRAP_TEXT_LIMIT)}
"""

    for attempt in range(2):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LanguageTrapBatch,
                    temperature=0.1
                ),
            )
            data = response.parsed
            for item in data.results:
                return item.requires_local_language, (item.evidence or "No explicit evidence quoted.")
            return False, "No language requirement found."
        except Exception as e:
            if attempt == 0:
                prompt += f"\n\nCorrection Request: Previous attempt failed with {str(e)}. Ensure the output strictly matches the schema."
            else:
                # Never auto-reject on API failure (would kill good jobs on transient errors)
                return False, "Language check unavailable (API error); allowed by default."

def batch_language_trap(client, jobs: List[JobListing], chunk_size: int = LANGUAGE_TRAP_CHUNK_SIZE, model: str = "gemini-2.5-flash", rules: Optional[str] = None) -> Dict[int, Tuple[bool, str]]:
    """Batch language check: one Gemini call per `chunk_size` postings.
    Returns {job_id: (requires_local_language, reason)}.
    `rules` overrides the default strict prompt (survival mode passes SURVIVAL_LANGUAGE_RULES)."""
    results: Dict[int, Tuple[bool, str]] = {}
    if not jobs:
        return results

    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        postings_text = ""
        for job in chunk:
            truncated = truncate_head_tail(job.full_text, LANGUAGE_TRAP_TEXT_LIMIT)
            postings_text += f"JOB_ID: {job.id}\n{truncated}\n---\n"

        prompt = (rules or _LANGUAGE_TRAP_RULES) + f"\nPostings:\n{postings_text}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LanguageTrapBatch,
                    temperature=0.1
                ),
            )
            data = response.parsed
            returned_ids = set()
            rejected_count = 0
            allowed_count = 0
            if data and data.results:
                for item in data.results:
                    returned_ids.add(item.job_id)
                    results[item.job_id] = (item.requires_local_language, item.evidence or "No explicit evidence quoted.")
                    if item.requires_local_language:
                        rejected_count += 1
                    else:
                        allowed_count += 1
            for job in chunk:
                if job.id not in returned_ids:
                    results[job.id] = (False, "Not analyzed by model; allowed by default.")
                    allowed_count += 1
            print(f"    -> [DEBUG] language gate: {rejected_count} rejected / {allowed_count} allowed / verdicts parsed OK")
        except Exception as e:
            print(f"    -> [!] Batch language trap failed: {e}")
            for job in chunk:
                results[job.id] = (False, "Language check unavailable (API error); allowed by default.")

    return results


# --- Entry-level gate (runs only when UserContext.entry_level_only) ---
ENTRY_LEVEL_TRAP_CHUNK_SIZE = 20   # postings per Gemini call
ENTRY_LEVEL_TRAP_TEXT_LIMIT = 4000 # chars of each posting sent for the experience check

class EntryLevelTrapItem(BaseModel):
    job_id: int = Field(description="The ID of the job posting being analyzed.")
    requires_experience: bool = Field(description="True ONLY if the posting explicitly demands 2+ years of professional experience as a requirement.")
    requires_enrollment: bool = Field(default=False, description="True ONLY if the posting explicitly demands current university enrollment (e.g. 'currently pursuing a degree', 'must be enrolled'). False when unknown.")
    evidence: str = Field(description="The exact sentence quoted from the posting that imposes the experience or enrollment requirement. Empty string if none.")

class EntryLevelTrapBatch(BaseModel):
    results: List[EntryLevelTrapItem]

# --- On-demand country/location filter ---
LOCATION_TRAP_CHUNK_SIZE = 20   # postings per Gemini call
LOCATION_TRAP_TEXT_LIMIT = 4000 # chars of each posting sent for the location check

class LocationTrapItem(BaseModel):
    job_id: int = Field(description="The ID of the job posting being analyzed.")
    location_mismatch: bool = Field(description="True ONLY if the posting's work location is clearly in a different country than the target.")
    evidence: str = Field(description="The exact sentence quoted from the posting that states the location. Empty string if none.")

class LocationTrapBatch(BaseModel):
    results: List[LocationTrapItem]

_LOCATION_TRAP_RULES = """
You are checking job postings for their WORK LOCATION.
The candidate's target country: {target_country}.

Each posting may include a STORED LOCATION line. Two kinds:
- board-reported city/region (trustworthy hint), or
- "query:X" = only the country term of the search query that found the posting (weak hint — the job itself may be elsewhere). The posting text always wins over any hint.

For EACH posting decide: is the work location clearly in a DIFFERENT country than the target?

Rules:
1. Set location_mismatch = true ONLY when the work location is clearly in another country (a foreign city, a different country name, "based in <other country>"). You KNOW world geography: a city implies its country (e.g. Munich -> Germany, Konstanz -> Germany, even if close to the target's border).
2. Accept (false) when: the location is in the target country; the job is remote/home-office without a contradicting country requirement; the location is not stated at all; or the text is ambiguous.
3. "Remote" is acceptable EVEN IF the company HQ is elsewhere - unless the posting explicitly demands residence or presence in another specific country.
4. Company headquarters or legal addresses alone do NOT define the work location; the actual place of work does.
5. If you set location_mismatch to true, you MUST quote the exact sentence from the posting in 'evidence'. If you cannot quote one, set it to false.
6. When in doubt, set location_mismatch to false - a false rejection loses a real job.
"""

_LOCATION_RESCUE_RULES = """
You are RE-CHECKING job postings that a first pass rejected for being in the wrong country.
The candidate's target country: {target_country}.

Each posting may include a STORED LOCATION line (board-reported location or search country hint). Treat it as a hint; the posting text wins if they disagree.

The first pass may have been too strict: it can confuse company HQ with the actual work location, or misread border cities.

For EACH posting decide: is the actual WORK LOCATION acceptable for the candidate?

ACCEPT (location_mismatch = false) when: the work location is in the target country, the job is remote/home-office, or the location is genuinely unclear.
REJECT (location_mismatch = true) ONLY when the work location is clearly and certainly in another country - and you MUST quote the exact sentence proving it in 'evidence'. If you cannot quote one, set location_mismatch to false.
"""

def batch_location_trap(client, jobs: List[JobListing], target_country: str, chunk_size: int = LOCATION_TRAP_CHUNK_SIZE, model: str = "gemini-2.5-flash", rules: Optional[str] = None) -> Dict[int, Tuple[bool, str]]:
    """Batch location check: one Gemini call per `chunk_size` postings.
    Free pre-pass: a stored location already containing the target country name is
    accepted without AI. Returns {job_id: (location_mismatch, reason)}.
    `rules` overrides the default prompt (the rescue button passes _LOCATION_RESCUE_RULES).
    Failures default to allowed."""
    results: Dict[int, Tuple[bool, str]] = {}
    if not jobs:
        return results

    prompt_rules = (rules or _LOCATION_TRAP_RULES).format(target_country=target_country)
    target_lower = target_country.strip().lower()

    # Free pre-pass: a BOARD-REPORTED stored location containing the target country
    # name is accepted without AI. "query:X" hints are only the search term, not the
    # job's real location — those always go to the AI.
    ai_jobs = []
    for job in jobs:
        board_reported = job.location and not job.location.startswith("query:")
        if board_reported and target_lower and target_lower in job.location.lower():
            results[job.id] = (False, f"Stored location '{job.location}' matches target (free pre-pass).")
        else:
            ai_jobs.append(job)

    for start in range(0, len(ai_jobs), chunk_size):
        chunk = ai_jobs[start:start + chunk_size]
        postings_text = ""
        for job in chunk:
            truncated = truncate_head_tail(job.full_text, LOCATION_TRAP_TEXT_LIMIT)
            stored = f"STORED LOCATION: {job.location}\n" if job.location else ""
            postings_text += f"JOB_ID: {job.id}\nTITLE: {job.title}\n{stored}{truncated}\n---\n"

        prompt = prompt_rules + f"\nPostings:\n{postings_text}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LocationTrapBatch,
                    temperature=0.1
                ),
            )
            data = response.parsed
            returned_ids = set()
            for item in data.results:
                returned_ids.add(item.job_id)
                results[item.job_id] = (item.location_mismatch, item.evidence or "No explicit evidence quoted.")
            # Jobs missing from the model's answer are allowed by default
            for job in chunk:
                if job.id not in returned_ids:
                    results[job.id] = (False, "Not analyzed by model; allowed by default.")
        except Exception as e:
            print(f"    -> [!] Batch location trap failed: {e}")
            for job in chunk:
                results[job.id] = (False, "Location check unavailable (API error); allowed by default.")

    return results


_YEARS_EXPERIENCE_RE = re.compile(
    r'(?:minimum|at least|mindestens)\s*(\d+)\s*\+?\s*(?:years|jahre)'
    r'|(\d+)\s*\+\s*(?:years|jahre)\s*(?:of\s*)?(?:professional\s*|relevant\s*)?(?:experience|erfahrung|berufserfahrung)',
    re.IGNORECASE
)

def has_clear_experience_demand(text: str) -> bool:
    """Free local heuristic: True only for an explicit 3+ years requirement written in
    the posting (never entry-level). Everything else goes to the AI check."""
    if not text:
        return False
    for m in _YEARS_EXPERIENCE_RE.finditer(text.lower()):
        years = int(m.group(1) or m.group(2))
        if years >= 3:
            return True
    return False

_ENTRY_LEVEL_TRAP_RULES = """
You are checking job postings for PROFESSIONAL EXPERIENCE requirements.
The candidate is a fresh graduate (degree already completed) with no professional experience, targeting entry-level roles.

For EACH posting decide: does it EXPLICITLY demand 2 or more years of professional experience as a REQUIREMENT?

Rules:
1. Only explicit demands count: "3+ years of experience required", "mindestens 2 Jahre Berufserfahrung", "proven track record of at least 5 years".
2. Nice-to-haves do NOT count: "ideally 1-2 years", "experience is a plus", "idealerweise erste Erfahrung".
3. Internships, working-student jobs, trainee programs and junior/entry-level roles are always fine ("first professional experience", "Berufseinsteiger willkommen", "0-2 years") -> requires_experience = false.
4. Education requirements (degree, field of study) do NOT count - the candidate has a degree. Only professional experience matters.
5. If you set requires_experience to true, you MUST quote the exact sentence from the posting in 'evidence'. If you cannot quote one, set it to false.
6. When in doubt, set requires_experience to false.
"""

_ENROLLMENT_RULE_ADDENDUM = """
Additionally, for EACH posting decide: does it EXPLICITLY demand CURRENT UNIVERSITY ENROLLMENT as a requirement?
{candidate_status_line}

Rules:
1. Only explicit demands count: "currently pursuing a Bachelor's/Master's degree", "must be enrolled at a university", "immatrikuliert", "Werkstudent (m/w/d) eingeschrieben".
2. Roles open to graduates do NOT count: "recent graduate", "Absolventen willkommen", "students or graduates", "degree completed or in final year" -> requires_enrollment = false.
3. If you set requires_enrollment to true, you MUST quote the exact sentence in 'evidence'. If you cannot quote one, set it to false.
4. When in doubt, set requires_enrollment to false.
"""

def _enrollment_addendum(candidate_status: str) -> str:
    """Status-aware enrollment rule block. Students never see this (their gate stays off):
    'must be enrolled' jobs are exactly what they want. Graduates reject strict
    enrollment demands but keep 'recent graduates welcome' roles; a Bachelor graduate
    additionally rejects EXCLUSIVE completed-Master (or higher) degree demands.
    Exclusivity matters: like seniority, only block when the posting demands what the
    candidate is NOT — OR-ed alternatives that include the candidate's level always pass
    ('Bachelor or Master student in mechanical engineering' fits a Bachelor graduate)."""
    if candidate_status == "graduate_bachelor":
        status_line = ("The candidate holds a COMPLETED Bachelor's degree and is NOT enrolled. "
                       "Explicit demands for a completed Master's degree or higher ALSO count as requires_enrollment, "
                       "but ONLY when EXCLUSIVE (e.g. 'Master's degree required', 'MSc only'). "
                       "OR-ed alternatives that include the Bachelor do NOT count: 'BSc or MSc', 'Bachelor or Master graduate', "
                       "'Bachelor graduate or Master student' -> requires_enrollment = false.")
    else:  # graduate_master, or legacy contexts with the flag on but no status stored
        status_line = ("The candidate holds a COMPLETED Master's degree and is NOT enrolled. "
                       "Degree-level requirements never count (the candidate already has a Master). "
                       "Current-enrollment demands count ONLY when EXCLUSIVE: 'students or graduates', "
                       "'enrolled or recently graduated', 'Bachelor or Master student' -> requires_enrollment = false.")
    return _ENROLLMENT_RULE_ADDENDUM.format(candidate_status_line=status_line)

def batch_entry_level_trap(client, jobs: List[JobListing], chunk_size: int = ENTRY_LEVEL_TRAP_CHUNK_SIZE, model: str = "gemini-2.5-flash", check_enrollment: bool = False, candidate_status: str = "") -> Dict[int, Tuple[bool, bool, str]]:
    """Batch experience-requirement check: one Gemini call per `chunk_size` postings.
    With check_enrollment=True, also flags explicit current-enrollment demands
    (rules shaped by candidate_status, see _enrollment_addendum).
    Returns {job_id: (requires_experience, requires_enrollment, evidence)}. Failures default to allowed."""
    results: Dict[int, Tuple[bool, bool, str]] = {}
    if not jobs:
        return results

    rules = _ENTRY_LEVEL_TRAP_RULES + (_enrollment_addendum(candidate_status) if check_enrollment else "")

    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        postings_text = ""
        for job in chunk:
            truncated = truncate_head_tail(job.full_text, ENTRY_LEVEL_TRAP_TEXT_LIMIT)
            postings_text += f"JOB_ID: {job.id}\nTITLE: {job.title}\n{truncated}\n---\n"

        prompt = rules + f"\nPostings:\n{postings_text}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=EntryLevelTrapBatch,
                    temperature=0.1
                ),
            )
            data = response.parsed
            returned_ids = set()
            for item in data.results:
                returned_ids.add(item.job_id)
                results[item.job_id] = (item.requires_experience, item.requires_enrollment, item.evidence or "No explicit evidence quoted.")
            # Jobs missing from the model's answer are allowed by default
            for job in chunk:
                if job.id not in returned_ids:
                    results[job.id] = (False, False, "Not analyzed by model; allowed by default.")
        except Exception as e:
            print(f"    -> [!] Batch entry-level trap failed: {e}")
            for job in chunk:
                results[job.id] = (False, False, "Experience check unavailable (API error); allowed by default.")

    return results

class RelevanceRescueItem(BaseModel):
    job_id: int = Field(description="The ID of the job posting being analyzed.")
    accept: bool = Field(description="True if the role is a real vacancy semantically related to the candidate's field.")
    reason: str = Field(description="Short reason for the decision.")

class RelevanceRescueBatch(BaseModel):
    results: List[RelevanceRescueItem]

_RELEVANCE_RESCUE_RULES = """
You are re-checking job postings that an automatic keyword filter rejected because NONE of the candidate's core keywords matched mathematically. Vector math misses synonyms and related roles, so a FEW of these may still be good fits - most are correctly rejected.

Candidate's core fields (must-have): {must_haves}
Candidate's secondary skills (nice-to-have): {nice_to_haves}

For EACH posting decide: is the posting's CORE DAILY WORK clearly inside the candidate's functional domain?

ACCEPT (accept=true) ONLY when:
- The activities described as the job's main responsibilities overlap directly with a must-have field. Your reason MUST name the concrete overlapping activity from the posting (e.g. "daily work is line optimization and waste reduction"), never the company or industry.
- Example that MUST be accepted: keyword "process optimisation" -> a posting for "Intern Process Quality" whose tasks are process mapping and improvement.

REJECT (accept=false) when ANY of these holds:
1. The link is to the company/industry but the role's daily work is something else (e.g. a marketing or business-development intern at a manufacturing company is NOT process optimisation work).
2. The content is an aggregator or listing page: a good job-like title, but the body is just a list of unrelated job links, navigation menus, or SEO filler instead of one concrete vacancy.
3. The role belongs to a clearly different field (e.g. the keywords are engineering but the posting is marketing, sales or finance).
4. The posting clearly demands senior/lead/manager level while the candidate targets intern/junior level.

When in doubt, REJECT - this is a second-chance review of already-rejected postings, and a wrong accept pollutes the match list.
Give a short reason for every decision.
"""

def batch_relevance_rescue(client, session: Session, user_context_id: int, jobs: List[JobListing], chunk_size: int = RELEVANCE_RESCUE_CHUNK_SIZE, model: str = "gemini-2.5-flash") -> Dict[int, Tuple[bool, str]]:
    """Batch relevance rescue: second-chance Gemini review for jobs rejected ONLY because
    no must-have keyword matched the vector filter. One call per `chunk_size` postings.
    Returns {job_id: (accept, reason)}. Anything not explicitly accepted stays rejected."""
    results: Dict[int, Tuple[bool, str]] = {}
    if not jobs:
        return results

    statement = select(Keyword).where(Keyword.user_context_id == user_context_id)
    keywords = session.exec(statement).all()
    must_haves = [k.word for k in keywords if k.category == KeywordCategory.MUST_HAVE.value]
    nice_to_haves = [k.word for k in keywords if k.category == KeywordCategory.NICE_TO_HAVE.value]

    prompt_rules = _RELEVANCE_RESCUE_RULES.format(must_haves=must_haves, nice_to_haves=nice_to_haves)

    for start in range(0, len(jobs), chunk_size):
        chunk = jobs[start:start + chunk_size]
        postings_text = ""
        for job in chunk:
            truncated = truncate_head_tail(job.full_text, RELEVANCE_RESCUE_TEXT_LIMIT)
            postings_text += f"JOB_ID: {job.id}\nTITLE: {job.title}\n{truncated}\n---\n"

        prompt = prompt_rules + f"\nPostings:\n{postings_text}"

        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RelevanceRescueBatch,
                    temperature=0.0
                ),
            )
            data = response.parsed
            returned_ids = set()
            for item in data.results:
                returned_ids.add(item.job_id)
                results[item.job_id] = (item.accept, item.reason)
            # Jobs missing from the model's answer stay rejected (rescue defaults closed)
            for job in chunk:
                if job.id not in returned_ids:
                    results[job.id] = (False, "Not analyzed by model; stays rejected.")
        except Exception as e:
            print(f"    -> [!] Batch relevance rescue failed: {e}")
            for job in chunk:
                results[job.id] = (False, "Relevance check unavailable (API error); stays rejected.")

    return results

# --- Must-have synonym expansion (SCORING ONLY — never discovery) ---
# Adding these to the keyword table would multiply the DDG query matrix, so they
# live here as a tunable. Key = lowercase must-have keyword; values = equivalent
# phrasings (incl. German) treated as the SAME functional area. Keywords not in
# the map simply form their own single-phrase area.
MUST_HAVE_SYNONYMS = {
    "industrial engineering": ["manufacturing engineering", "production engineering", "produktionstechnik", "fertigungstechnik"],
    "process optimization": ["process improvement", "prozessoptimierung", "continuous improvement", "lean management", "operational excellence", "kaizen"],
    "operational analysis & efficiency": ["prozessanalyse", "effizienzsteigerung", "business operations", "operations analyst"],
    "logistics & supply chain management": ["logistik", "supply chain", "supply chain management", "materialfluss", "operations strategy", "logistics management"],
    "logistics management": ["logistik", "supply chain", "supply chain management", "materialfluss", "operations strategy"],
    "data analysis & automation": ["datenanalyse", "data analytics", "automatisierung", "process automation", "prozessautomatisierung"],
    "data-driven automation": ["datenanalyse", "data analytics", "automatisierung", "process automation", "prozessautomatisierung"],
    "data analysis": ["datenanalyse", "data analytics", "business analytics"],
}
SECOND_AREA_CREDIT = 10.0  # bonus when a SECOND distinct must-have area also matches (>0.4)

def calculate_vector_score(job_text: str, keywords: List[Keyword]) -> Tuple[float, str]:
    """Vector-embedding scoring with OR semantics.

    Keywords are alternative job PROFILES, not a checklist: one strong must-have
    match is enough. score = must_best*70 + nice_bonus - soft_penalty + E_boost.
    """
    if not job_text:
        return 0.0, "Empty job text."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(job_text)

    if not chunks:
        return 0.0, "No chunks generated."

    chunk_embeddings = embedder.encode(chunks)

    must_haves = [k.word for k in keywords if k.category == KeywordCategory.MUST_HAVE.value]
    nice_to_haves = [k.word for k in keywords if k.category == KeywordCategory.NICE_TO_HAVE.value]
    soft_nos = [k.word for k in keywords if k.category == KeywordCategory.SOFT_NO.value]
    hard_nos = [k.word for k in keywords if k.category == KeywordCategory.HARD_NO.value]

    def max_similarities(words) -> np.ndarray:
        """Per-word max cosine similarity vs the job chunks (vectors are L2-normalized)."""
        if not words:
            return np.array([])
        word_embeddings = embedder.encode(words)
        assert chunk_embeddings.shape[1] == word_embeddings.shape[1], "Fatal mathematical error: Vector dimension mismatch."
        similarity_matrix = np.dot(word_embeddings, chunk_embeddings.T)  # (num_words, num_chunks)
        return np.max(similarity_matrix, axis=1)  # (num_words,)

    # Check hard nos (unchanged: any match > 0.5 rejects outright)
    if hard_nos:
        hard_no_embeddings = embedder.encode(hard_nos)
        assert chunk_embeddings.shape[1] == hard_no_embeddings.shape[1], "Vector dimension mismatch."
        sim_matrix = np.dot(hard_no_embeddings, chunk_embeddings.T)
        if np.any(sim_matrix > 0.5):
            return 0.0, "Rejected due to Hard No vector match."

    E_boost = 0.0
    lower_job_text = job_text.lower()
    if "english" in lower_job_text or "international team" in lower_job_text or "visa sponsorship" in lower_job_text or "relocation" in lower_job_text:
        E_boost = 10.0

    # Group must-haves into AREAS (keyword + its synonyms = one area); per area the
    # best phrase similarity counts. OR semantics across areas: the best area drives
    # the score, a second DISTINCT matching area earns SECOND_AREA_CREDIT (ranking).
    area_results = []  # (best_sim, area_label, winning_phrase)
    for w in must_haves:
        phrases = [w] + [p for p in MUST_HAVE_SYNONYMS.get(w.lower(), []) if p.lower() != w.lower()]
        sims = max_similarities(phrases)
        if sims.size > 0:
            best_i = int(np.argmax(sims))
            area_results.append((float(sims[best_i]), w, phrases[best_i]))

    area_results.sort(key=lambda r: -r[0])
    if area_results and area_results[0][0] > 0.4:
        must_best, win_label, win_phrase = area_results[0]
    else:
        must_best, win_label, win_phrase = 0.0, "-", "-"

    if must_best == 0.0 and len(must_haves) > 0:
        return 0.0, NO_MUST_HAVE_REASON

    second_area_sim = area_results[1][0] if len(area_results) > 1 else 0.0
    second_credit = SECOND_AREA_CREDIT if second_area_sim > 0.4 else 0.0

    nice_sims = max_similarities(nice_to_haves)
    soft_sims = max_similarities(soft_nos)
    nice_bonus = min(20.0, 5.0 * int(np.sum(nice_sims > 0.4))) if nice_sims.size > 0 else 0.0
    soft_penalty = min(20.0, 10.0 * int(np.sum(soft_sims > 0.4))) if soft_sims.size > 0 else 0.0

    final_score = must_best * 70.0 + second_credit + nice_bonus - soft_penalty + E_boost
    final_score = max(0.0, min(100.0, final_score))

    return final_score, (f"Vector Match Score: {final_score:.2f} (best area '{win_label}' via '{win_phrase}' sim {must_best:.2f}, "
                         f"second-area credit: +{second_credit:.0f}, nice bonus: +{nice_bonus:.0f}, "
                         f"soft-no penalty: -{soft_penalty:.0f}, English boost: +{E_boost:.0f})")

# --- Cross-board duplicate detection (flag only; the job stays MATCHED) ---
DUPLICATE_SIM_THRESHOLD = 0.88  # cosine on normalized titles; company must also be compatible
SIMHASH_HAMMING_THRESHOLD = 8   # body fingerprint distance: <=8 = same posting, different board chrome.
                                # Calibrated on live DB 2026-07-27: real dups at d=0, same-title d=12,
                                # unrelated jobs d>=17 (p1 of all pairs = 22) -> 8 keeps 2x margin to noise.
                                # Synthetic head+tail board chrome test: d=8.

def _normalize_body(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', (text or '').lower())

def body_hash(text: str) -> str:
    """MD5 of the fully normalized body — catches byte-identical reposts under new URLs."""
    return hashlib.md5(_normalize_body(text).encode()).hexdigest()

def text_simhash(text: str) -> int:
    """64-bit simhash over word 3-grams. Near-duplicate bodies (same posting with
    different board boilerplate) differ by only a few bits. 0 when text too short."""
    words = re.sub(r'[^a-z0-9 ]', ' ', (text or '').lower()).split()
    if len(words) < 3:
        return 0
    v = [0] * 64
    for i in range(len(words) - 2):
        h = int.from_bytes(hashlib.md5(" ".join(words[i:i + 3]).encode()).digest()[:8], "big")
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    fp = 0
    for b in range(64):
        if v[b] > 0:
            fp |= (1 << b)
    return fp

def simhash_distance(a: int, b: int) -> int:
    return bin((a ^ b) & 0xFFFFFFFFFFFFFFFF).count("1")

def simhash_to_db(h: int) -> int:
    """SQLite INTEGER is signed 64-bit — fold the unsigned fingerprint into range."""
    return h - (1 << 64) if h >= (1 << 63) else h

def simhash_from_db(v: int) -> int:
    return v + (1 << 64) if v < 0 else v

_COMPANY_NOISE_TOKENS = {"ag", "gmbh", "sa", "se", "ltd", "llc", "inc", "co", "group", "holding", "technologies", "technology", "technik", "solutions", "international", "the"}

def _normalize_title(title: str) -> str:
    """Lowercase, drop the jobspy ' at Company' suffix, '(m/w/d)'-style tags and punctuation."""
    t = (title or "").lower()
    t = re.sub(r'\s+at\s+.+$', '', t)
    t = re.sub(r'\(.*?\)', ' ', t)
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def _extract_company(title: str) -> str:
    """Company part of jobspy-style 'Title at Company'. Empty string when unknown."""
    m = re.search(r'\s+at\s+(.+)$', title or "")
    if not m:
        return ""
    c = re.sub(r'[^a-z0-9 ]', ' ', m.group(1).lower())
    tokens = [t for t in re.sub(r'\s+', ' ', c).strip().split() if t not in _COMPANY_NOISE_TOKENS]
    return " ".join(tokens)

def _company_compatible(a: str, b: str) -> bool:
    """True when either company is unknown, or they share a meaningful token
    ('helbling' vs 'helbling technik' -> True; 'hilti' vs 'medtronic' -> False)."""
    if not a or not b:
        return True
    return bool(set(a.split()) & set(b.split()))

def find_duplicate_of(job: JobListing, recent_matches: List[JobListing]) -> Optional[int]:
    """Returns the id of an already-matched job that is the same vacancy on another
    board, or None. Three layers, strongest signal first:
    (1) exact normalized body hash — byte-identical repost under a new URL;
    (2) body simhash Hamming distance — same posting with different board chrome
        (boards copy the employer's text verbatim but rewrite titles freely);
    (3) exact normalized title key, then title embedding cosine — both behind the
        company-compatibility gate.
    Body layers need no company check: identical text is proof on its own.
    Side effect: stores the job's own fingerprint in job.text_simhash for future runs."""
    body = job.full_text or ""
    if len(body) >= 200:  # shorter texts produce unstable fingerprints
        new_hash = body_hash(body)
        new_sh = text_simhash(body)
        job.text_simhash = simhash_to_db(new_sh)
        for m in recent_matches:
            if m.id == job.id or len(m.full_text or "") < 200:
                continue
            if body_hash(m.full_text) == new_hash:
                return m.id
            other_sh = simhash_from_db(m.text_simhash) if m.text_simhash is not None else text_simhash(m.full_text)
            if new_sh and other_sh and simhash_distance(new_sh, other_sh) <= SIMHASH_HAMMING_THRESHOLD:
                return m.id

    new_key = _normalize_title(job.title)
    new_company = _extract_company(job.title)
    if not new_key:
        return None

    candidates = []
    for m in recent_matches:
        if m.id == job.id:
            continue
        if not _company_compatible(new_company, _extract_company(m.title)):
            continue
        if new_key == _normalize_title(m.title):
            return m.id  # exact normalized key match
        candidates.append(m)

    if candidates:
        embs = embedder.encode([new_key] + [_normalize_title(m.title) for m in candidates])
        sims = np.dot(embs[1:], embs[0])
        best = int(np.argmax(sims))
        if sims[best] >= DUPLICATE_SIM_THRESHOLD:
            return candidates[best].id
    return None

SURVIVAL_MATCH_SCORE = 50.0  # fixed score for survival-mode matches (no keyword scoring there)

def _recent_matched_jobs(session: Session, ctx) -> List[JobListing]:
    """Last 100 MATCHED jobs, scoped to the same user when possible (for duplicate checks)."""
    if ctx is not None and ctx.user_id is not None:
        from .models import UserContext
        statement_matched = (
            select(JobListing)
            .join(UserContext, JobListing.user_context_id == UserContext.id)
            .where(JobListing.status == JobStatus.MATCHED.value)
            .where(UserContext.user_id == ctx.user_id)
            .order_by(JobListing.id.desc())
            .limit(100)
        )
    else:
        statement_matched = select(JobListing).where(JobListing.status == JobStatus.MATCHED.value).order_by(JobListing.id.desc()).limit(100)
    return session.exec(statement_matched).all()

def evaluate_job(session: Session, job: JobListing, user_context_id: int, language_result: Optional[Tuple[bool, str]] = None, entry_result: Optional[Tuple[bool, bool, str]] = None, location_result: Optional[Tuple[bool, str]] = None):
    from .models import UserContext
    ctx = session.get(UserContext, user_context_id)
    target_country = ctx.target_country if ctx else "Unknown"

    statement = select(Keyword).where(Keyword.user_context_id == user_context_id)
    keywords = session.exec(statement).all()

    if not gate_1_metadata_check(job):
        job.status = JobStatus.REJECTED.value
        job.score = 0.0
        job.diagnostic_report = "Rejected: Invalid content (Security check, 404, or generic page)."
        return

    must_haves = [k.word for k in keywords if k.category == KeywordCategory.MUST_HAVE.value]
    if not getattr(ctx, 'survival_mode', False) and must_haves:
        is_mismatch, mismatch_reason = check_title_domain_mismatch(job.title, must_haves)
        if is_mismatch:
            job.status = JobStatus.REJECTED.value
            job.score = 0.0
            job.diagnostic_report = f"Rejected: Early Title Domain Guard ({mismatch_reason})"
            return


    # Language gate. Survival mode uses its own batched check (allows up to B1);
    # a missing result (e.g. Laboratory tab) means allow.
    if getattr(ctx, 'survival_mode', False):
        if language_result is not None:
            is_trap, reason = language_result
            if is_trap:
                job.status = JobStatus.REJECTED.value
                job.score = 0.0
                job.diagnostic_report = f"Rejected: Language above B1 required (survival mode). Reason: {reason}"
                return
    elif ctx and ctx.english_only:
        if language_result is not None:
            is_trap, reason = language_result
        else:
            from .ai import resolve_ai_config
            api_key, model = resolve_ai_config(session, user_context_id)
            is_trap, reason = gate_2_language_trap(job.full_text, target_country, api_key=api_key, model=model)
        if is_trap:
            job.status = JobStatus.REJECTED.value
            job.score = 0.0
            job.diagnostic_report = f"Rejected: Language Trap. Reason: {reason}"
            return


    # Entry-level gate: reject explicit professional-experience demands (entry_level_only)
    # and/or explicit current-enrollment demands (reject_enrollment_required).
    # Each flag acts only on its own half of the result tuple.
    if entry_result is not None and (getattr(ctx, 'entry_level_only', False) or getattr(ctx, 'reject_enrollment_required', False)):
        requires_exp, requires_enroll, exp_reason = entry_result
        if requires_exp and getattr(ctx, 'entry_level_only', False):
            job.status = JobStatus.REJECTED.value
            job.score = 0.0
            job.diagnostic_report = f"Rejected: Experience requirement (entry-level mode). Reason: {exp_reason}"
            return
        if requires_enroll and getattr(ctx, 'reject_enrollment_required', False):
            job.status = JobStatus.REJECTED.value
            job.score = 0.0
            job.diagnostic_report = f"Rejected: Current enrollment required (graduate mode). Reason: {exp_reason}"
            return

    # Survival mode: no keyword/vector scoring at all. Any real vacancy that passed
    # gate-1 and the language gate (up to B1 accepted) is a match. Duplicates flagged.
    if getattr(ctx, 'survival_mode', False):
        if location_result is not None and location_result[0]:
            job.status = JobStatus.REJECTED.value
            job.diagnostic_report = f"Rejected: Location mismatch (target: {target_country}). Evidence: {location_result[1]}"
            return
        duplicate_of = find_duplicate_of(job, _recent_matched_jobs(session, ctx))
        job.status = JobStatus.MATCHED.value
        job.score = SURVIVAL_MATCH_SCORE
        report = "Survival mode: passed language gate (no language need or up to B1)."
        job.diagnostic_report = f"[DUPLICATE of #{duplicate_of}] {report}" if duplicate_of is not None else report
        return

    score, report = calculate_vector_score(job.full_text, keywords)

    if score == 0.0:
        job.status = JobStatus.REJECTED.value
        job.score = 0.0
        job.diagnostic_report = report
    elif score < MIN_MATCH_SCORE:
        # Weak single-keyword hit below the floor. NOT rescue-eligible (rescue is
        # only for zero-match NO_MUST_HAVE_REASON rejections).
        job.status = JobStatus.REJECTED.value
        job.score = score
        job.diagnostic_report = f"Rejected: Below minimum match score ({score:.2f} < {MIN_MATCH_SCORE}). {report}"
    else:
        # Location gate AFTER scoring: a rejected job keeps its real score so the
        # location rescue button can restore it intact if this was a false positive.
        if location_result is not None and location_result[0]:
            job.status = JobStatus.REJECTED.value
            job.score = score
            job.diagnostic_report = f"Rejected: Location mismatch (target: {target_country}). Evidence: {location_result[1]}"
            return

        recent_matches = _recent_matched_jobs(session, ctx)

        duplicate_of = find_duplicate_of(job, recent_matches)

        job.status = JobStatus.MATCHED.value
        job.score = score
        if duplicate_of is not None:
            job.diagnostic_report = f"[DUPLICATE of #{duplicate_of}] {report}"
        else:
            job.diagnostic_report = report
