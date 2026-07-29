import streamlit as st
import pandas as pd
import os
import re
from sqlmodel import select, delete
from core.database import create_db_and_tables, get_session
from core.models import User, UserContext, Keyword, KeywordCategory, JobListing, JobStatus, ScrapeQueue, DeadLetterQueue
from core.cv_parser import extract_text_from_pdf, analyze_cv_with_gemini
from core.discovery import discover_jobs
from core.scraper import scrape_and_evaluate_jobs
from core.ai import resolve_ai_config_for_user

st.set_page_config(page_title="Job Search Agent", layout="wide")

@st.cache_resource
def init_db():
    create_db_and_tables()

init_db()

# =====================
# PROFILE SELECTOR (per-user data pools)
# =====================
def get_all_users():
    session = next(get_session())
    try:
        return session.exec(select(User).order_by(User.id)).all()
    finally:
        session.close()

st.sidebar.title("👤 Profile")

users = get_all_users()
user_names = [u.name for u in users]

new_name = st.sidebar.text_input("Create new profile:", placeholder="e.g. Adil, Ayşe...")
if st.sidebar.button("➕ Create Profile"):
    if new_name.strip():
        if new_name.strip() in user_names:
            st.sidebar.warning("This profile name already exists.")
        else:
            session = next(get_session())
            try:
                user = User(name=new_name.strip())
                session.add(user)
                session.commit()
                st.session_state['active_user_id'] = user.id
                st.rerun()
            finally:
                session.close()
    else:
        st.sidebar.warning("Please enter a name.")

users = get_all_users()
if not users:
    st.info("Welcome! Create a profile in the sidebar to get started.")
    st.stop()

user_names = [u.name for u in users]
if 'active_user_id' not in st.session_state or st.session_state['active_user_id'] not in [u.id for u in users]:
    st.session_state['active_user_id'] = users[0].id

active_index = [u.id for u in users].index(st.session_state['active_user_id'])
selected_name = st.sidebar.selectbox("Active profile:", user_names, index=active_index)
st.session_state['active_user_id'] = next(u.id for u in users if u.name == selected_name)
active_user_id = st.session_state['active_user_id']

st.sidebar.caption("Jobs, keywords and scores are kept separate per profile.")

# ---- AI SETTINGS (per profile) ----
with st.sidebar.expander("🔑 AI Settings", expanded=False):
    session = next(get_session())
    try:
        _u = session.get(User, active_user_id)
        current_model = (_u.gemini_model if _u and _u.gemini_model else "gemini-2.5-flash")
        has_key = bool(_u and _u.gemini_api_key)
    finally:
        session.close()

    if has_key:
        st.caption("Key status: ✅ saved for this profile")
    elif os.environ.get("GEMINI_API_KEY"):
        st.caption("Key status: ⚙️ using GEMINI_API_KEY environment variable")
    else:
        st.caption("Key status: ❌ not set — the AI features will not run")

    new_key = st.text_input("Gemini API key:", type="password", placeholder="Paste key for this profile")
    new_model = st.text_input("Gemini model:", value=current_model,
                              help="e.g. gemini-2.5-flash (cheap, default) or gemini-2.5-pro (smarter, pricier)")
    if st.button("💾 Save AI Settings"):
        session = next(get_session())
        try:
            u = session.get(User, active_user_id)
            if new_key.strip():
                u.gemini_api_key = new_key.strip()
            if new_model.strip():
                u.gemini_model = new_model.strip()
            session.add(u)
            session.commit()
        finally:
            session.close()
        st.success("AI settings saved for this profile.")
        st.rerun()

# ---- DANGER ZONE (data deletion) ----
def _wipe_jobs_for_contexts(session, ctx_ids):
    """Delete jobs + queue entries for the given context ids. Returns deleted job count."""
    if not ctx_ids:
        return 0
    job_ids = session.exec(select(JobListing.id).where(JobListing.user_context_id.in_(ctx_ids))).all()
    if job_ids:
        session.exec(delete(ScrapeQueue).where(ScrapeQueue.job_id.in_(job_ids)))
        session.exec(delete(DeadLetterQueue).where(DeadLetterQueue.job_id.in_(job_ids)))
        session.exec(delete(JobListing).where(JobListing.id.in_(job_ids)))
    return len(job_ids)

with st.sidebar.expander("🗑 Danger Zone", expanded=False):
    st.caption("These actions cannot be undone.")
    confirm_delete = st.checkbox("I understand — enable the delete buttons")
    if confirm_delete:
        if st.button("🧹 Reset my jobs (keep profile, key & keywords)"):
            session = next(get_session())
            try:
                ctx_ids = session.exec(select(UserContext.id).where(UserContext.user_id == active_user_id)).all()
                n = _wipe_jobs_for_contexts(session, ctx_ids)
                session.commit()
            finally:
                session.close()
            st.success(f"Done — {n} jobs wiped for this profile.")
            st.rerun()

        if st.button("❌ Delete this profile completely"):
            session = next(get_session())
            try:
                ctx_ids = session.exec(select(UserContext.id).where(UserContext.user_id == active_user_id)).all()
                _wipe_jobs_for_contexts(session, ctx_ids)
                if ctx_ids:
                    session.exec(delete(Keyword).where(Keyword.user_context_id.in_(ctx_ids)))
                    session.exec(delete(UserContext).where(UserContext.id.in_(ctx_ids)))
                session.exec(delete(User).where(User.id == active_user_id))
                session.commit()
            finally:
                session.close()
            for k in ['active_user_id', 'current_user_context_id', 'gemini_keywords', 'keyword_df', 'google_captcha_blocked']:
                st.session_state.pop(k, None)
            st.success("Profile deleted.")
            st.rerun()

def get_active_user_context_ids():
    session = next(get_session())
    try:
        return session.exec(select(UserContext.id).where(UserContext.user_id == active_user_id)).all()
    finally:
        session.close()

st.title("🤖 Autonomous Job Agent (v7)")

tab1, tab2, tab_lab, tab_rej = st.tabs(["⚙️ Setup & CV", "🎯 Matched Jobs", "🧪 Laboratory", "❌ Rejected Jobs"])

with tab1:
    st.header("1. Target Country & CV Upload")
    target_country = st.selectbox(
        "Target Country:", 
        ["Germany", "Switzerland", "Netherlands", "United States", "United Kingdom", "Austria", "Turkey"],
        index=0
    )
    seniority_level = st.selectbox(
        "Seniority Level:",
        ["Internship", "Graduate", "Junior", "Mid-Level", "Senior", "Lead"],
        index=2,
        help="This informs the AI to generate appropriate search phrases and filter out overqualified roles."
    )
    
    graduate_junior_ratio = 50
    if seniority_level == "Graduate":
        graduate_junior_ratio = st.slider(
            "Graduate Role Mix (Intern vs. Junior)",
            min_value=0,
            max_value=100,
            value=50,
            step=10,
            format="%d%% Junior",
            help="0% = All Internships, 100% = All Junior roles. 50% = Equal mix."
        )
    english_only = st.checkbox("English-only Jobs (Disable local language translations)", value=False)
    
    st.subheader("Source Engine Blend")
    jobspy_ratio = st.slider(
        "JobSpy vs DuckDuckGo Mix",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        format="%d%% JobSpy",
        help="100% means purely native boards (LinkedIn/Indeed). Lower values force the agent to reserve a slice of the search for DuckDuckGo to find aggregator sites (like iagora)."
    )
    # Implicitly derive entry_level_only based on seniority selection
    entry_level_only = seniority_level in ["Internship", "Graduate", "Junior"]
    # Derive enrollment status from seniority level
    if seniority_level == "Internship":
        enrollment_status_key = "student_master"  # Keep 'must be enrolled' jobs
    else:
        enrollment_status_key = "graduate_master" # Reject 'must be enrolled' jobs
        
    reject_enrollment_required = enrollment_status_key.startswith("graduate")
    survival_mode = st.checkbox(
        "🆘 Survival jobs mode (any field, no experience/degree needed — only language matters)",
        value=False,
        help="Searches no-qualification jobs in the areas below. CV keywords are ignored; every real vacancy that needs no language or up to B1 local language is a match."
    )
    survival_area_keys = []
    if survival_mode:
        from core.discovery import get_survival_area_labels, get_survival_caption
        _area_labels = get_survival_area_labels(target_country)
        chosen = st.multiselect(
            "Job areas to search:",
            options=list(_area_labels.keys()),
            default=list(_area_labels.keys()),
            format_func=lambda k: _area_labels[k],
        )
        survival_area_keys = chosen
        st.caption(get_survival_caption(target_country))

        # Survival mode needs no CV/keywords: allow saving the context directly.
        if 'gemini_keywords' not in st.session_state:
            if st.button("Save & Continue (survival mode — no CV needed)"):
                session = next(get_session())
                ctx = UserContext(target_country=target_country, english_only=english_only, entry_level_only=entry_level_only,
                                  reject_enrollment_required=reject_enrollment_required, enrollment_status=enrollment_status_key,
                                  seniority_level=seniority_level, graduate_junior_ratio=graduate_junior_ratio,
                                  jobspy_ratio=jobspy_ratio,
                                  survival_mode=True, survival_areas=",".join(survival_area_keys), user_id=active_user_id)
                session.add(ctx)
                session.commit()
                session.refresh(ctx)
                st.session_state['current_user_context_id'] = ctx.id
                st.success("Survival mode saved. You can now start the agent below!")
    
    uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Analyze CV"):
            with st.spinner("Analyzing CV... (uses the API key from '🔑 AI Settings')"):
                try:
                    cfg_session = next(get_session())
                    try:
                        api_key, model_name = resolve_ai_config_for_user(cfg_session, active_user_id)
                    finally:
                        cfg_session.close()
                    cv_text = extract_text_from_pdf(uploaded_file)
                    result = analyze_cv_with_gemini(cv_text, target_country, english_only, seniority_level, graduate_junior_ratio, api_key=api_key, model=model_name)
                    st.session_state['gemini_keywords'] = result
                    st.success("CV analyzed! Please check the table below and click 'Save & Continue' to proceed.")
                except Exception as e:
                    st.error(f"Error: {e}")
                    
    if 'gemini_keywords' in st.session_state:
        st.subheader("2. Keyword Approval (Human-in-the-Loop)")
        st.write("Edit the keywords generated by AI if necessary. Make sure to choose the correct category!")
        
        # Initialize DataFrame in session state if it doesn't exist
        if 'keyword_df' not in st.session_state:
            data = []
            for cat, words in st.session_state['gemini_keywords'].items():
                if isinstance(words, list):
                    for w in words:
                        mapped_cat = KeywordCategory.MUST_HAVE.value
                        if "nice" in cat.lower():
                            mapped_cat = KeywordCategory.NICE_TO_HAVE.value
                        elif "search" in cat.lower():
                            mapped_cat = KeywordCategory.SEARCH_PHRASE.value
                        data.append({"Keyword": w, "Category": mapped_cat})
            
            for _ in range(2):
                data.append({"Keyword": "", "Category": KeywordCategory.HARD_NO.value})
            for _ in range(2):
                data.append({"Keyword": "", "Category": KeywordCategory.SOFT_NO.value})
                
            df = pd.DataFrame(data)
            
            # Custom sort logic (MUST_HAVE first, then NICE_TO_HAVE, SEARCH_PHRASE, SOFT_NO, HARD_NO)
            cat_order = {
                KeywordCategory.MUST_HAVE.value: 1,
                KeywordCategory.NICE_TO_HAVE.value: 2,
                KeywordCategory.SEARCH_PHRASE.value: 3,
                KeywordCategory.SOFT_NO.value: 4,
                KeywordCategory.HARD_NO.value: 5
            }
            df['_sort'] = df['Category'].map(cat_order)
            df = df.sort_values('_sort').drop('_sort', axis=1).reset_index(drop=True)
            st.session_state['keyword_df'] = df
            
        edited_df = st.data_editor(st.session_state['keyword_df'], num_rows="dynamic", column_config={
            "Category": st.column_config.SelectboxColumn(
                "Category",
                options=[e.value for e in KeywordCategory],
                required=True
            )
        })
        
        # Auto-sort and refresh if user makes a change
        if not edited_df.equals(st.session_state['keyword_df']):
            cat_order = {
                KeywordCategory.MUST_HAVE.value: 1,
                KeywordCategory.NICE_TO_HAVE.value: 2,
                KeywordCategory.SEARCH_PHRASE.value: 3,
                KeywordCategory.SOFT_NO.value: 4,
                KeywordCategory.HARD_NO.value: 5
            }
            edited_df['_sort'] = edited_df['Category'].map(cat_order)
            sorted_df = edited_df.sort_values(by=['_sort', 'Keyword']).drop('_sort', axis=1).reset_index(drop=True)
            st.session_state['keyword_df'] = sorted_df
            st.rerun()
        
        if st.button("Save & Continue"):
            session = next(get_session())
            ctx = UserContext(target_country=target_country, english_only=english_only, entry_level_only=entry_level_only,
                              reject_enrollment_required=reject_enrollment_required, enrollment_status=enrollment_status_key,
                              seniority_level=seniority_level, graduate_junior_ratio=graduate_junior_ratio,
                              jobspy_ratio=jobspy_ratio,
                              survival_mode=survival_mode, survival_areas=",".join(survival_area_keys), user_id=active_user_id)
            session.add(ctx)
            session.commit()
            session.refresh(ctx)
            
            for index, row in edited_df.iterrows():
                if pd.notna(row["Keyword"]) and str(row["Keyword"]).strip():
                    kw = Keyword(word=str(row["Keyword"]).strip(), category=row["Category"], user_context_id=ctx.id)
                    session.add(kw)
            
            session.commit()
            st.session_state['current_user_context_id'] = ctx.id
            st.success("Keywords saved to the database. You can now start the agent!")
            
    if 'current_user_context_id' in st.session_state:
        st.subheader("3. Start Agent")
        max_results = st.selectbox("Search Limit (Number of NEW jobs to scan)", [50, 100, 150, 200, 250, 300], index=0)
        
        from core.discovery import discover_jobs, get_default_job_terms

        session = next(get_session())
        ctx = session.get(UserContext, st.session_state['current_user_context_id'])

        
        progress_bar = st.empty()
        status_text = st.empty()

        def update_progress(current, total, message):
            progress = float(current) / float(total) if total > 0 else 0.0
            progress_bar.progress(min(progress, 1.0))
            status_text.text(f"[{current}/{total}] {message}")

        if st.button("Discover and Evaluate Jobs"):
            st.write("### 🧠 Batched Discovery & Semantic Evaluation Phase")

            with st.spinner(f"Searching with polite delays, then reviewing in AI batches (Target: {max_results} new jobs)..."):
                added_jobs, total_fetched, failed_sites = discover_jobs(
                    session,
                    st.session_state['current_user_context_id'],
                    max_results=max_results,
                    progress_callback=update_progress
                )

                st.write(f"🦆 **Total raw links fetched (JobSpy + DuckDuckGo):** {total_fetched}")
                st.write(f"🤖 **Jobs that passed AI pre-screening:** {len(added_jobs)}")
                if failed_sites:
                    # Split captcha blocks from ordinary failures
                    captcha = [s for s, e in failed_sites if e.startswith("CAPTCHA_BLOCKED")]
                    ordinary = [(s, e) for s, e in failed_sites if not e.startswith("CAPTCHA_BLOCKED")]
                    if ordinary:
                        st.warning("⚠️ These boards were unreachable and were skipped: " +
                                   ", ".join(f"**{site}** ({err[:60]})" for site, err in ordinary))
                    if captcha:
                        st.session_state['google_captcha_blocked'] = True

            # Otomatik JSON Export (scoped to the active profile)
            from core.database import dump_database_to_json
            import json
            dump_database_to_json(session, "jobs_dump.json", user_context_ids=get_active_user_context_ids())

            progress_bar.empty()
            status_text.empty()
            st.success("🎉 All processes completed! Check the other tabs for results.")

            with open("jobs_dump.json", "r", encoding="utf-8") as f:
                json_data = f.read()

            st.download_button(
                label="📥 Download AI Analysis Dump (JSON)",
                data=json_data,
                file_name="jobs_dump.json",
                mime="application/json",
                help="Download the complete database snapshot to analyze false positives/negatives with another AI."
            )

        # ---- CAPTCHA RESCUE (persists across Streamlit reruns) ----
        if st.session_state.get('google_captcha_blocked'):
            st.error("🤖 Google hit us with a captcha (IP rate-limit). You can unblock it manually:")
            st.markdown(
                "1. [👉 Open Google Jobs in your browser](https://www.google.com/search?q=jobs&udm=8) "
                "(same network/IP) and solve the captcha if one appears.\n"
                "2. Come back here and click **retry**. Solving it usually unflags the IP for a while."
            )
            col_retry, col_dismiss = st.columns(2)
            if col_retry.button("✅ I solved it — retry Google now"):
                from core.discovery import search_google_only
                from core.scraper import scrape_and_evaluate_jobs
                with st.spinner("Retrying Google gently (1-2 requests, with a pre-call pause)..."):
                    added_n, err = search_google_only(session, st.session_state['current_user_context_id'])
                if err:
                    if "sorry" in err.lower() or "429" in err:
                        st.error("Still blocked — Google hasn't released the IP yet. Wait a few hours and try again, or solve the captcha once more.")
                    else:
                        st.error(f"Retry failed: {err[:120]}")
                else:
                    st.session_state['google_captcha_blocked'] = False
                    with st.spinner(f"Google is back! Scraping & evaluating the {added_n} new jobs..."):
                        scrape_and_evaluate_jobs(session, st.session_state['current_user_context_id'], update_progress)
                    st.success(f"✅ Google unblocked — {added_n} new jobs added and evaluated!")
                    st.rerun()
            if col_dismiss.button("Dismiss (skip Google for now)"):
                st.session_state['google_captcha_blocked'] = False
                st.rerun()
with tab2:
    st.header("🎯 Matched Jobs (High Score)")
    # Rendered unconditionally on every rerun: if this list lives behind a button,
    # the status selectbox below can never commit — the rerun after a change
    # returns button=False and skips the write. (Bug reported 2026-07-24.)
    session = next(get_session())
    # Sort logic: NOT_APPLIED first (your to-do list on top), then score. Scoped to the active profile.
    ctx_ids = get_active_user_context_ids()
    statement = (
        select(JobListing)
        .where(JobListing.status == JobStatus.MATCHED.value)
        .where(JobListing.user_context_id.in_(ctx_ids))
        .order_by(JobListing.application_status != "NOT_APPLIED", JobListing.score.desc())
    )
    matched_jobs = session.exec(statement).all()

    # ---- On-demand country filter: boards return remote/neighboring-country jobs
    # despite the location hint, and nothing in the pipeline verifies location.
    # This second pass rejects only postings CLEARLY located in another country
    # (evidence-quoting; remote and unstated locations stay).
    from core.evaluator import batch_location_trap, MIN_MATCH_SCORE, AI_RESCUE_SCORE, SURVIVAL_MATCH_SCORE
    if matched_jobs and st.button(f"🌍 Country filter: re-check locations of {len(matched_jobs)} matched jobs"):
        with st.spinner(f"AI is checking locations in batches of 20..."):
            from core.ai import get_genai_client, MissingApiKeyError
            rejected_n, checked_n = 0, 0
            by_ctx = {}
            for j in matched_jobs:
                by_ctx.setdefault(j.user_context_id, []).append(j)
            for ctx_id, ctx_jobs in by_ctx.items():
                ctx = session.get(UserContext, ctx_id)
                if not ctx:
                    continue
                try:
                    client, model_name = get_genai_client(session, ctx_id)
                except MissingApiKeyError as e:
                    st.error(str(e))
                    continue
                results = batch_location_trap(client, ctx_jobs, ctx.target_country, model=model_name)
                for job in ctx_jobs:
                    mismatch, reason = results.get(job.id, (False, "Not analyzed by model; allowed by default."))
                    checked_n += 1
                    if mismatch:
                        job.status = JobStatus.REJECTED.value
                        # Score is kept (not zeroed) so the location rescue button can
                        # restore the job with its original score if this was a false positive.
                        job.diagnostic_report = f"Rejected: Location mismatch (target: {ctx.target_country}). Evidence: {reason}"
                        rejected_n += 1
            session.commit()
        st.success(f"Done: {checked_n} jobs checked — {rejected_n} moved to Rejected (wrong country), {checked_n - rejected_n} kept.")
        st.rerun()

    # ---- Re-verify matched: re-runs ALL current gates (language, experience/enrollment,
    # location) and re-scores with the current OR formula. Self-service cleanup after
    # gate/scoring improvements. AI gates fail open (missing key -> gates skipped,
    # local re-scoring still runs). One commit at the end.
    if matched_jobs and st.button(f"🔁 Re-verify {len(matched_jobs)} matched jobs (language, experience, location, re-score)"):
        with st.spinner("Re-verifying matched jobs against the current gates (batches of 20)..."):
            from core.ai import get_genai_client, MissingApiKeyError
            from core.evaluator import (batch_language_trap, batch_entry_level_trap, batch_location_trap,
                                        calculate_vector_score, detect_language_hint, has_clear_experience_demand,
                                        MIN_MATCH_SCORE, SURVIVAL_LANGUAGE_RULES)
            rejected_n, rescored_n, checked_n = 0, 0, 0
            by_ctx = {}
            for j in matched_jobs:
                by_ctx.setdefault(j.user_context_id, []).append(j)
            for ctx_id, ctx_jobs in by_ctx.items():
                ctx = session.get(UserContext, ctx_id)
                if not ctx:
                    continue
                survival = bool(getattr(ctx, 'survival_mode', False))
                try:
                    client, model_name = get_genai_client(session, ctx_id)
                except MissingApiKeyError as e:
                    st.error(f"Profile '{getattr(ctx, 'name', ctx_id)}': {e} — AI gates skipped, re-scoring only.")
                    client, model_name = None, None

                # (a) language gate
                lang_results = {}
                if client and (getattr(ctx, 'english_only', False) or survival):
                    if survival:
                        lang_results = batch_language_trap(client, ctx_jobs, model=model_name, rules=SURVIVAL_LANGUAGE_RULES)
                    else:
                        to_check = [j for j in ctx_jobs if detect_language_hint(j.full_text or "") == "ambiguous"]
                        if to_check:
                            lang_results = batch_language_trap(client, to_check, model=model_name)

                # (b) experience / enrollment gate (free regex first, AI batched)
                entry_results = {}
                check_enrollment = bool(getattr(ctx, 'reject_enrollment_required', False))
                if getattr(ctx, 'entry_level_only', False) or check_enrollment:
                    clear = [j for j in ctx_jobs if has_clear_experience_demand(j.full_text or "")]
                    clear_ids = {j.id for j in clear}
                    for j in clear:
                        entry_results[j.id] = (True, False, "Explicit 3+ years experience requirement (free pre-filter).")
                    ambiguous = [j for j in ctx_jobs if j.id not in clear_ids]
                    if client and ambiguous:
                        entry_results.update(batch_entry_level_trap(client, ambiguous, model=model_name, check_enrollment=check_enrollment,
                                                                    candidate_status=getattr(ctx, 'enrollment_status', '') or 'graduate_master'))

                # (c) location gate (always on)
                loc_results = {}
                if client and ctx.target_country:
                    loc_results = batch_location_trap(client, ctx_jobs, ctx.target_country, model=model_name)

                # (d) apply gates, then re-score survivors (survival contexts keep their fixed score)
                kw_rows = session.exec(select(Keyword).where(Keyword.user_context_id == ctx_id)).all()
                for job in ctx_jobs:
                    checked_n += 1
                    diag = None
                    is_trap, reason = lang_results.get(job.id, (False, ""))
                    if is_trap:
                        diag = f"Rejected: Language Trap (re-verify). Reason: {reason}"
                    if diag is None:
                        req_exp, req_enroll, ev = entry_results.get(job.id, (False, False, ""))
                        if req_exp and getattr(ctx, 'entry_level_only', False):
                            diag = f"Rejected: Experience requirement (re-verify). Reason: {ev}"
                        elif req_enroll and check_enrollment:
                            diag = f"Rejected: Current enrollment required (re-verify). Reason: {ev}"
                    if diag is None:
                        mismatch, reason = loc_results.get(job.id, (False, ""))
                        if mismatch:
                            diag = f"Rejected: Location mismatch (re-verify; target: {ctx.target_country}). Evidence: {reason}"
                    if diag is None and not survival:
                        new_score, report = calculate_vector_score(job.full_text, kw_rows)
                        if new_score < MIN_MATCH_SCORE:
                            diag = f"Rejected: Below minimum match score after re-scoring ({new_score:.2f} < {MIN_MATCH_SCORE}). {report}"
                        else:
                            if abs(new_score - (job.score or 0.0)) > 0.01:
                                rescored_n += 1
                            job.score = new_score
                            job.diagnostic_report = report
                    if diag is not None:
                        job.status = JobStatus.REJECTED.value
                        job.diagnostic_report = diag
                        rejected_n += 1
            session.commit()
        st.success(f"Done: {checked_n} matched jobs re-verified — {rejected_n} moved to Rejected, {checked_n - rejected_n} kept ({rescored_n} re-scored to the new scale).")
        st.rerun()

    if not matched_jobs:
        st.info("No matched jobs yet.")
    else:
        status_options = ["NOT_APPLIED", "APPLIED", "INTERVIEW", "REJECTED"]
        for job in matched_jobs:
            is_rep = "[POTENTIAL REPETITION]" in (job.diagnostic_report or "")
            title_prefix = ":orange[[REPETITION]] " if is_rep else ""

            with st.expander(f"{title_prefix}[{job.score} Pts] [{job.application_status}] {job.title}"):
                st.markdown(f"**URL:** [Go to Job]({job.url})")

                new_status = st.selectbox(
                    "Application Status",
                    status_options,
                    index=status_options.index(job.application_status),
                    key=f"status_{job.id}"
                )

                if new_status != job.application_status:
                    job.application_status = new_status
                    session.add(job)
                    session.commit()
                    st.rerun()

                colored_report = job.diagnostic_report or ""
                for tag in re.findall(r"\[(?:DUPLICATE of #\d+|POTENTIAL REPETITION)\]", colored_report):
                    colored_report = colored_report.replace(tag, f":orange[**{tag}**]")
                st.markdown(f"**Diagnostic Report:** {colored_report}")
                # Floor transparency: fixed scores (rescue/survival) are exempt from the
                # floor; formula scores show their margin (negative = re-verify would reject).
                if job.score == AI_RESCUE_SCORE or job.score == SURVIVAL_MATCH_SCORE:
                    st.caption(f"Fixed score ({job.score:.0f} pts — AI rescue / survival match): exempt from the {MIN_MATCH_SCORE:.0f}-pt rejection floor.")
                else:
                    st.caption(f"Margin to the {MIN_MATCH_SCORE:.0f}-pt rejection floor: {job.score - MIN_MATCH_SCORE:+.1f} pts.")
                if job.snippet:
                    st.markdown(f"**Snippet:** {job.snippet}")

with tab_lab:
    st.header("🧪 Laboratory (Testing Ground)")
    st.write("Manually test a specific URL to see why the system scored it the way it did.")
    
    test_url = st.text_input("Test URL:")
    
    if st.button("Test URL"):
        if 'current_user_context_id' not in st.session_state:
            st.error("You must complete the setup in Tab 1 first (User Context).")
        elif not test_url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Scraping and testing URL..."):
                try:
                    from core.scraper import extract_job_text
                    from core.evaluator import evaluate_job
                    from core.database import generate_url_hash

                    session = next(get_session())

                    dummy_job = JobListing(
                        url=test_url,
                        url_hash=generate_url_hash(test_url),
                        title="Test Job",
                        status=JobStatus.PENDING_SCRAPE.value
                    )

                    full_text = extract_job_text(test_url)
                    if not full_text:
                        st.error("Failed to scrape text from this URL. Access denied or page is not supported.")
                    else:
                        dummy_job.full_text = full_text
                        evaluate_job(session, dummy_job, st.session_state['current_user_context_id'])
                        
                        if dummy_job.status == JobStatus.REJECTED.value:
                            st.error(f"Test Result: REJECTED")
                            st.write(f"**Diagnostic Report:** {dummy_job.diagnostic_report}")
                        else:
                            st.success(f"Test Result: MATCHED")
                            st.write(f"**Score:** {dummy_job.score}")
                            st.write(f"**Diagnostic Report:** {dummy_job.diagnostic_report}")
                            
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab_rej:
    st.header("❌ Rejected Jobs")
    st.write("Jobs that failed the AI pre-screening or received a 'Hard No'.")

    # ---- AI Relevance Rescue: second chance for jobs rejected ONLY because no
    # must-have keyword matched the vector filter. Jobs already re-checked by the
    # AI (diagnostic has " | AI relevance check:" appended) are not eligible again,
    # so repeat clicks don't burn tokens. Aggregator trash is rejected by the prompt.
    from core.evaluator import batch_relevance_rescue, NO_MUST_HAVE_REASON, AI_RESCUE_SCORE

    ctx_ids = get_active_user_context_ids()
    session = next(get_session())
    try:
        rescuable_count = len(session.exec(
            select(JobListing)
            .where(JobListing.status == JobStatus.REJECTED.value)
            .where(JobListing.user_context_id.in_(ctx_ids))
            .where(JobListing.diagnostic_report == NO_MUST_HAVE_REASON)
        ).all())
    finally:
        session.close()

    if st.button(f"🛟 AI Rescue: re-check {rescuable_count} 'no keyword match' rejections", disabled=rescuable_count == 0):
        with st.spinner(f"AI is re-checking {rescuable_count} rejected jobs in batches of 20..."):
            from core.ai import get_genai_client, MissingApiKeyError
            rescued_total, checked_total = 0, 0
            session = next(get_session())
            try:
                jobs = session.exec(
                    select(JobListing)
                    .where(JobListing.status == JobStatus.REJECTED.value)
                    .where(JobListing.user_context_id.in_(ctx_ids))
                    .where(JobListing.diagnostic_report == NO_MUST_HAVE_REASON)
                ).all()

                # Group by context: each context has its own keywords and AI config
                by_ctx = {}
                for j in jobs:
                    by_ctx.setdefault(j.user_context_id, []).append(j)

                for ctx_id, ctx_jobs in by_ctx.items():
                    try:
                        client, model_name = get_genai_client(session, ctx_id)
                    except MissingApiKeyError as e:
                        st.error(str(e))
                        continue
                    results = batch_relevance_rescue(client, session, ctx_id, ctx_jobs, model=model_name)
                    for job in ctx_jobs:
                        accept, reason = results.get(job.id, (False, "Not analyzed by model; stays rejected."))
                        checked_total += 1
                        if accept:
                            job.status = JobStatus.MATCHED.value
                            job.score = AI_RESCUE_SCORE
                            job.diagnostic_report = f"Rescued by AI relevance check: {reason}"
                            rescued_total += 1
                        else:
                            job.diagnostic_report = f"{job.diagnostic_report} | AI relevance check: {reason}"
                session.commit()
            finally:
                session.close()
        st.success(f"Done: AI re-checked {checked_total} jobs — {rescued_total} rescued to Matched, {checked_total - rescued_total} stayed rejected.")
        st.rerun()

    # ---- Location Rescue: second chance for jobs rejected by the on-demand
    # country filter. The AI re-reads the full description and cross-references
    # the context's target country; HQ-vs-workplace confusions get restored with
    # their ORIGINAL score (the country filter no longer zeroes it). Jobs already
    # re-checked (" | Location re-check:" appended) are not eligible again.
    from core.evaluator import batch_location_trap, _LOCATION_RESCUE_RULES

    session = next(get_session())
    try:
        loc_rescuable_count = len(session.exec(
            select(JobListing)
            .where(JobListing.status == JobStatus.REJECTED.value)
            .where(JobListing.user_context_id.in_(ctx_ids))
            .where(JobListing.diagnostic_report.like("Rejected: Location mismatch%"))
            .where(JobListing.diagnostic_report.not_like("%| Location re-check:%"))
        ).all())
    finally:
        session.close()

    if st.button(f"🌍 Location Rescue: re-check {loc_rescuable_count} country-filter rejections", disabled=loc_rescuable_count == 0):
        with st.spinner(f"AI is cross-referencing {loc_rescuable_count} job descriptions with the target country..."):
            from core.ai import get_genai_client, MissingApiKeyError
            rescued_total, checked_total = 0, 0
            session = next(get_session())
            try:
                jobs = session.exec(
                    select(JobListing)
                    .where(JobListing.status == JobStatus.REJECTED.value)
                    .where(JobListing.user_context_id.in_(ctx_ids))
                    .where(JobListing.diagnostic_report.like("Rejected: Location mismatch%"))
                    .where(JobListing.diagnostic_report.not_like("%| Location re-check:%"))
                ).all()

                by_ctx = {}
                for j in jobs:
                    by_ctx.setdefault(j.user_context_id, []).append(j)

                for ctx_id, ctx_jobs in by_ctx.items():
                    ctx = session.get(UserContext, ctx_id)
                    if not ctx:
                        continue
                    try:
                        client, model_name = get_genai_client(session, ctx_id)
                    except MissingApiKeyError as e:
                        st.error(str(e))
                        continue
                    results = batch_location_trap(client, ctx_jobs, ctx.target_country, model=model_name, rules=_LOCATION_RESCUE_RULES)
                    for job in ctx_jobs:
                        mismatch, reason = results.get(job.id, (False, "Not analyzed by model; restored by default."))
                        checked_total += 1
                        if not mismatch:
                            job.status = JobStatus.MATCHED.value  # original score intact
                            job.diagnostic_report = f"Rescued by location re-check: {reason}"
                            rescued_total += 1
                        else:
                            job.diagnostic_report = f"{job.diagnostic_report} | Location re-check: {reason}"
                session.commit()
            finally:
                session.close()
        st.success(f"Done: {checked_total} location rejections re-checked — {rescued_total} restored to Matched, {checked_total - rescued_total} confirmed wrong country.")
        st.rerun()

    if st.button("Refresh (Rejected)"):
        session = next(get_session())
        ctx_ids = get_active_user_context_ids()
        statement = (
            select(JobListing)
            .where(JobListing.status == JobStatus.REJECTED.value)
            .where(JobListing.user_context_id.in_(ctx_ids))
        )
        rejected_jobs = session.exec(statement).all()
        
        if not rejected_jobs:
            st.info("No rejected jobs yet.")
        else:
            st.write(f"**Total rejected jobs:** {len(rejected_jobs)}")
            for job in rejected_jobs:
                with st.expander(f"{job.title}"):
                    st.markdown(f"**URL:** [Go to Job]({job.url})")
                    st.markdown(f"**Reason:** {job.diagnostic_report}")
                    if job.snippet:
                        st.markdown(f"**Snippet:** {job.snippet}")


