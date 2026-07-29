from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING_SCRAPE = "PENDING_SCRAPE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    REJECTED = "REJECTED"
    MATCHED = "MATCHED"
    EXPIRED = "EXPIRED"

class KeywordCategory(str, Enum):
    MUST_HAVE = "must_have"
    NICE_TO_HAVE = "nice_to_have"
    SOFT_NO = "soft_no"
    HARD_NO = "hard_no"
    SEARCH_PHRASE = "search_phrase"

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True, index=True)
    gemini_api_key: Optional[str] = None
    gemini_model: str = Field(default="gemini-2.5-flash")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class UserContext(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    target_country: str
    english_only: bool = Field(default=False)
    entry_level_only: bool = Field(default=False)
    reject_enrollment_required: bool = Field(default=False)  # derived from enrollment_status (graduate_*)
    enrollment_status: str = Field(default="")  # "", student_bachelor, student_master, graduate_bachelor, graduate_master
    seniority_level: str = Field(default="Junior")  # Internship, Graduate, Junior, Mid-Level, Senior, Lead
    graduate_junior_ratio: int = Field(default=50)  # 0 = All Internships, 100 = All Junior
    jobspy_ratio: int = Field(default=80)  # Percentage of jobs to pull from JobSpy (rest from DDGS)
    survival_mode: bool = Field(default=False)
    survival_areas: str = Field(default="")  # comma-separated keys from SURVIVAL_JOB_AREAS
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Keyword(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    word: str = Field(index=True)
    category: str # KeywordCategory
    user_context_id: Optional[int] = Field(default=None, foreign_key="usercontext.id")

class JobListing(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url_hash: str = Field(unique=True, index=True)
    url: str
    title: str
    location: Optional[str] = None  # board-reported location (JobSpy) or the country term of the DDG query that found it
    snippet: Optional[str] = None
    full_text: Optional[str] = None
    text_simhash: Optional[int] = None  # 64-bit body fingerprint (signed) for near-duplicate detection
    status: str = Field(default=JobStatus.PENDING_SCRAPE.value)
    score: float = Field(default=0.0)
    diagnostic_report: Optional[str] = None
    application_status: str = Field(default="NOT_APPLIED")
    user_context_id: Optional[int] = Field(default=None, foreign_key="usercontext.id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ScrapeQueue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="joblisting.id")
    url: str
    accept_language: str
    user_context_id: int
    status: str = Field(default="PENDING")
    retry_count: int = Field(default=0)
    
class DeadLetterQueue(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    job_id: int
    url: str
    error_message: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
