from sqlmodel import SQLModel, create_engine, Session, select
from .models import JobListing
import hashlib
import json
import os
import sqlite3
import time

sqlite_file_name = "jobs.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=False)

def _archive_legacy_db_if_needed():
    """If an old-schema jobs.db exists, rename it aside and let a fresh DB
    be created. No data migration. Checks the columns added after v6."""
    if not os.path.exists(sqlite_file_name):
        return
    try:
        conn = sqlite3.connect(sqlite_file_name)
        def cols(table):
            try:
                return [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            except sqlite3.Error:
                return []
        jl_cols = cols("joblisting")
        user_cols = cols("user")
        conn.close()
    except sqlite3.Error:
        return  # Not a valid sqlite file; leave it alone

    schema_ok = (
        jl_cols and "user_context_id" in jl_cols
        and user_cols and "gemini_model" in user_cols
    )
    if jl_cols and not schema_ok:
        legacy_name = f"{sqlite_file_name}.legacy"
        if os.path.exists(legacy_name):
            legacy_name = f"{sqlite_file_name}.legacy.{int(time.time())}"
        os.rename(sqlite_file_name, legacy_name)
        print(f"[DB] Legacy schema detected. Moved old database to: {legacy_name}")

def _migrate_columns_if_needed():
    """Lightweight additive migrations for existing DBs (the fresh-start guard only
    covers breaking legacy schemas). Adds new columns with defaults; never drops data."""
    if not os.path.exists(sqlite_file_name):
        return
    try:
        conn = sqlite3.connect(sqlite_file_name)
        cols = [row[1] for row in conn.execute("PRAGMA table_info(usercontext)").fetchall()]
        if cols and "entry_level_only" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN entry_level_only BOOLEAN DEFAULT 0")
            conn.commit()
            print("[DB] Migrated: added usercontext.entry_level_only")
        if cols and "survival_mode" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN survival_mode BOOLEAN DEFAULT 0")
            conn.execute("ALTER TABLE usercontext ADD COLUMN survival_areas VARCHAR DEFAULT ''")
            conn.commit()
            print("[DB] Migrated: added usercontext.survival_mode + survival_areas")
        if cols and "reject_enrollment_required" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN reject_enrollment_required BOOLEAN DEFAULT 0")
            conn.commit()
            print("[DB] Migrated: added usercontext.reject_enrollment_required")
        if cols and "enrollment_status" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN enrollment_status VARCHAR DEFAULT ''")
            conn.commit()
            print("[DB] Migrated: added usercontext.enrollment_status")
        if cols and "seniority_level" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN seniority_level VARCHAR DEFAULT 'Junior'")
            conn.commit()
            print("[DB] Migrated: added usercontext.seniority_level")
        if cols and "graduate_junior_ratio" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN graduate_junior_ratio INTEGER DEFAULT 50")
            conn.commit()
            print("[DB] Migrated: added usercontext.graduate_junior_ratio")
        if cols and "jobspy_ratio" not in cols:
            conn.execute("ALTER TABLE usercontext ADD COLUMN jobspy_ratio INTEGER DEFAULT 80")
            conn.commit()
            print("[DB] Migrated: added usercontext.jobspy_ratio")
        jl_cols = [row[1] for row in conn.execute("PRAGMA table_info(joblisting)").fetchall()]
        if jl_cols and "location" not in jl_cols:
            conn.execute("ALTER TABLE joblisting ADD COLUMN location VARCHAR")
            conn.commit()
            print("[DB] Migrated: added joblisting.location")
        if jl_cols and "text_simhash" not in jl_cols:
            conn.execute("ALTER TABLE joblisting ADD COLUMN text_simhash INTEGER")
            conn.commit()
            print("[DB] Migrated: added joblisting.text_simhash")
        conn.close()
    except sqlite3.Error:
        pass

def create_db_and_tables():
    _archive_legacy_db_if_needed()
    _migrate_columns_if_needed()
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session

def generate_url_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()

def is_job_exists(session: Session, url: str) -> bool:
    url_hash = generate_url_hash(url)
    statement = select(JobListing).where(JobListing.url_hash == url_hash)
    result = session.exec(statement).first()
    return result is not None

def dump_database_to_json(session: Session, file_path: str = "jobs_dump.json", user_context_ids=None):
    statement = select(JobListing)
    if user_context_ids is not None:
        statement = statement.where(JobListing.user_context_id.in_(user_context_ids))
    jobs = session.exec(statement).all()
    data = []
    
    for job in jobs:
        data.append({
            "id": job.id,
            "title": job.title,
            "url": job.url,
            "status": job.status,
            "score": job.score,
            "diagnostic_report": job.diagnostic_report,
            "snippet": job.snippet
        })
        
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
