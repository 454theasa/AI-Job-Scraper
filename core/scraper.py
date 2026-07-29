import trafilatura
from curl_cffi import requests
from sqlmodel import Session, select
from .models import JobListing, JobStatus, UserContext
from .evaluator import evaluate_job, detect_language_hint, batch_language_trap, batch_relevance_rescue, batch_entry_level_trap, batch_location_trap, has_clear_experience_demand, NO_MUST_HAVE_REASON, AI_RESCUE_SCORE, SURVIVAL_LANGUAGE_RULES
from playwright.sync_api import sync_playwright
import httpx
import re
import html

def extract_via_ats_api(url: str) -> str:
    """Attempt to intercept ATS URLs and fetch clean JSON directly from their backend APIs."""
    
    # 1. Lever
    lever_match = re.search(r'jobs\.lever\.co/([^/]+)/([^/?]+)', url)
    if lever_match:
        slug = lever_match.group(1)
        job_id = lever_match.group(2)
        api_url = f"https://api.lever.co/v0/postings/{slug}/{job_id}"
        try:
            r = httpx.get(api_url, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                text = data.get("descriptionPlain", "")
                if "lists" in data:
                    for l in data["lists"]:
                        text += "\n\n" + l.get("text", "") + "\n"
                        # Handle Lever's nested list content
                        content = l.get("content", [])
                        if isinstance(content, list):
                            text += "\n".join([f"- {c}" for c in content if isinstance(c, str)])
                        elif isinstance(content, str):
                            text += f"\n- {content}"
                print(f"    -> [⚡] ATS Intercepted: Lever API")
                return text
        except Exception as e:
            print(f"    -> [~] Lever API fallback failed: {e}")

    # 2. Greenhouse
    greenhouse_match = re.search(r'(?:boards|job-boards(?:\.eu)?)\.greenhouse\.io/([^/]+)/jobs/([^/?]+)', url)
    if greenhouse_match:
        slug = greenhouse_match.group(1)
        job_id = greenhouse_match.group(2)
        api_url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs/{job_id}"
        try:
            r = httpx.get(api_url, timeout=10.0)
            if r.status_code == 200:
                data = r.json()
                html_content = data.get("content", "")
                if html_content:
                    unescaped = html.unescape(html_content)
                    text = trafilatura.extract(unescaped, include_links=False, include_tables=True)
                    print(f"    -> [⚡] ATS Intercepted: Greenhouse API")
                    return text if text else ""
        except Exception as e:
            print(f"    -> [~] Greenhouse API fallback failed: {e}")
            
    return ""

def extract_job_text_playwright(url: str, accept_language: str) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a realistic user agent along with the locale
        context = browser.new_context(
            locale=accept_language.split(',')[0],
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        try:
            # wait_until="domcontentloaded" is usually faster than networkidle but for JS-heavy sites networkidle is better. 
            # We'll use a 15-second timeout so it doesn't hang forever.
            page.goto(url, wait_until="networkidle", timeout=15000)
        except Exception as e:
            # Sometimes networkidle times out even when the page is mostly loaded
            print(f"    -> [~] Playwright warning: {e}")
            
        raw_html = page.content()
        browser.close()
        
        # Zero-Token HTML Pre-Filter (LinkedIn & Others)
        if "closed-job" in raw_html.lower():
            raise ValueError("EXPIRED_LINKEDIN")
        
        extracted_text = trafilatura.extract(
            raw_html, 
            include_links=False, 
            include_comments=False, 
            include_tables=True,
            no_fallback=False
        )
        return extracted_text if extracted_text else ""

from lxml import html as lxml_html
from .models import JobListing, JobStatus, UserContext, ScrapeQueue

def extract_job_text(url: str, accept_language: str = "en-US,en;q=0.9") -> str:
    # 1. Network Evasion Layer: Spoofing a standard human browser with TLS fingerprinting
    headers = {
        "Accept-Language": accept_language,
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1"
    }
    
    try:
        import os
        proxies = {}
        proxy_url = os.environ.get("PROXY_URL")
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
            
        response = requests.get(
            url, 
            headers=headers, 
            impersonate="chrome120", 
            proxies=proxies,
            timeout=15.0
        )
        
        if response.status_code in (403, 401) or response.status_code >= 500:
            raise ValueError(f"HTTP Error {response.status_code}")
            
        raw_html = response.text
        
        if "closed-job" in raw_html.lower():
            raise ValueError("EXPIRED_LINKEDIN")
            
    except Exception as exc:
        raise ValueError(f"Network failure: {str(exc)}")

    # 2. Semantic Extraction Layer using high-speed lxml
    try:
        tree = lxml_html.fromstring(raw_html)
        for element in tree.xpath('.//script | .//style | .//noscript | .//header | .//footer'):
            element.drop_tree()
        extracted_text = " ".join(tree.xpath('//body//text()'))
        extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
        return extracted_text
    except Exception as e:
        print(f"lxml extraction failed: {e}")
        return ""

def scrape_and_evaluate_jobs(session: Session, user_context_id: int, progress_callback=None):
    context = session.exec(select(UserContext).where(UserContext.id == user_context_id)).first()
    target_country = context.target_country.lower().replace('i̇', 'i').replace('ı', 'i') if context else ""
    
    lang_map = {
        "almanya": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "germany": "de-DE,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "isvicre": "de-CH,de;q=0.9,fr-CH;q=0.8,en-US;q=0.7",
        "isviçre": "de-CH,de;q=0.9,fr-CH;q=0.8,en-US;q=0.7",
        "switzerland": "de-CH,de;q=0.9,fr-CH;q=0.8,en-US;q=0.7",
        "avusturya": "de-AT,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "austria": "de-AT,de;q=0.9,en-US;q=0.8,en;q=0.7",
        "turkiye": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "turkey": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "hollanda": "nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7",
        "netherlands": "nl-NL,nl;q=0.9,en-US;q=0.8,en;q=0.7"
    }
    
    accept_language = lang_map.get(target_country, "en-US,en;q=0.9")
    statement = select(JobListing).where(JobListing.status == JobStatus.PENDING_SCRAPE.value)
    pending_jobs = session.exec(statement).all()

    print("\n" + "="*50)
    print(f"[DEBUG] NETWORK EVASION LAYER ACTIVE")
    print(f"[DEBUG] Spoofing Chrome 121 via curl_cffi")
    print(f"[DEBUG] Region Accept-Language: {accept_language}")
    print(f"[DEBUG] Targets to extract: {len(pending_jobs)} jobs")
    print("="*50 + "\n")

    total_jobs = len(pending_jobs)

    # =====================
    # PHASE 1: SCRAPE (no AI calls)
    # =====================
    jobs_with_text = []
    for idx, job in enumerate(pending_jobs, 1):
        if progress_callback:
            progress_callback(idx, total_jobs, f"Fetching data... ({job.title[:30]})")

        try:
            print(f"[{idx}/{total_jobs}] Extracting: {job.title[:45]}...")

            # If JobSpy already pulled the full description, skip network fetch
            text = job.full_text

            if not text:
                text = extract_via_ats_api(job.url)

            if not text:
                try:
                    text = extract_job_text(job.url, accept_language=accept_language)
                except ValueError as e:
                    if "EXPIRED_LINKEDIN" in str(e):
                        raise
                    else:
                        print(f"    -> [!] HTTP blocked ({str(e)}). Pushing to Tier 1 Queue...")
                        q_item = ScrapeQueue(job_id=job.id, url=job.url, accept_language=accept_language, user_context_id=user_context_id)
                        session.add(q_item)
                        continue # Let worker handle it

            if not text:
                print(f"    -> [!] Extracted empty text. Pushing to Tier 1 Queue...")
                q_item = ScrapeQueue(job_id=job.id, url=job.url, accept_language=accept_language, user_context_id=user_context_id)
                session.add(q_item)
                continue

            job.full_text = text
            jobs_with_text.append(job)
            print(f"    -> [+] Text extracted successfully.")

        except ValueError as e:
            if "EXPIRED_LINKEDIN" in str(e):
                print(f"    -> [⏳] Pre-Filter: LinkedIn Job is already closed.")
                job.status = JobStatus.EXPIRED.value
                job.score = 0.0
                job.diagnostic_report = "Rejected at HTML Pre-Filter: LinkedIn closed-job tag detected."
            else:
                print(f"    -> [!] Scraping error: {str(e)}")
                job.status = JobStatus.MANUAL_REVIEW.value
                job.diagnostic_report = f"Scraping engellendi: {str(e)}"
        except Exception as e:
            print(f"    -> [!] Unexpected error: {str(e)}")
            job.status = JobStatus.MANUAL_REVIEW.value
            job.diagnostic_report = f"Beklenmeyen hata: {str(e)}"

    session.commit()

    # =====================
    # PHASE 2: BATCH LANGUAGE TRAP (one Gemini call per ~20 ambiguous postings)
    # Survival mode uses its own rules (allows up to B1) and checks EVERY posting,
    # because the text language itself is irrelevant there — a German posting with
    # no language demand is fine, only the demanded LEVEL matters.
    # =====================
    language_results = {}
    survival = bool(context and getattr(context, 'survival_mode', False))
    if context and (getattr(context, 'english_only', False) or survival) and jobs_with_text:
        if survival:
            jobs_to_check = jobs_with_text
            rules = SURVIVAL_LANGUAGE_RULES
            print(f"\n[DEBUG] Survival mode: language-level check (B1 cap) on all {len(jobs_to_check)} postings.")
        else:
            jobs_to_check = jobs_with_text
            rules = None
            print(f"\n[DEBUG] English-only mode: AI language trap on all {len(jobs_to_check)} postings.")


        if jobs_to_check:
            from .ai import get_genai_client, MissingApiKeyError
            try:
                client, model_name = get_genai_client(session, user_context_id)
            except MissingApiKeyError as e:
                print(f"    -> [!] {e}")
                client, model_name = None, None
            if client:
                if progress_callback:
                    progress_callback(0, 1, f"AI language check on {len(jobs_to_check)} postings...")
                language_results = batch_language_trap(client, jobs_to_check, model=model_name, rules=rules)

    # =====================
    # PHASE 2B: ENTRY-LEVEL GATE (entry_level_only and/or reject_enrollment_required;
    # free regex first, AI batched)
    # =====================
    entry_results = {}
    check_enrollment = bool(context and getattr(context, 'reject_enrollment_required', False))
    if context and (getattr(context, 'entry_level_only', False) or check_enrollment) and jobs_with_text:
        clear_demand = [j for j in jobs_with_text if has_clear_experience_demand(j.full_text or "")]
        clear_ids = {j.id for j in clear_demand}
        ambiguous_exp = [j for j in jobs_with_text if j.id not in clear_ids]
        for j in clear_demand:
            entry_results[j.id] = (True, False, "Explicit 3+ years experience requirement (free pre-filter).")
        print(f"\n[DEBUG] Entry-level pre-filter: {len(clear_demand)} clear experience demands (free), {len(ambiguous_exp)} need AI check.")

        if ambiguous_exp:
            from .ai import get_genai_client, MissingApiKeyError
            try:
                client, model_name = get_genai_client(session, user_context_id)
            except MissingApiKeyError as e:
                print(f"    -> [!] {e}")
                client, model_name = None, None
            if client:
                if progress_callback:
                    progress_callback(0, 1, f"AI experience check on {len(ambiguous_exp)} postings...")
                entry_results.update(batch_entry_level_trap(client, ambiguous_exp, model=model_name, check_enrollment=check_enrollment,
                                                            candidate_status=getattr(context, 'enrollment_status', '') or 'graduate_master'))

    # =====================
    # PHASE 2C: LOCATION GATE (always on — kills "false corrects": remote and
    # neighboring-country jobs that leak through the boards' country hints.
    # Board-reported locations matching the target are free-passed without AI.)
    # =====================
    location_results = {}
    if context and context.target_country and jobs_with_text:
        from .ai import get_genai_client, MissingApiKeyError
        try:
            client, model_name = get_genai_client(session, user_context_id)
        except MissingApiKeyError as e:
            print(f"    -> [!] {e}")
            client, model_name = None, None
        if client:
            if progress_callback:
                progress_callback(0, 1, f"AI location check on {len(jobs_with_text)} postings...")
            location_results = batch_location_trap(client, jobs_with_text, context.target_country, model=model_name)
            free_n = sum(1 for v in location_results.values() if "free pre-pass" in v[1])
            print(f"\n[DEBUG] Location gate: {free_n} free-passed by stored location, {len(location_results) - free_n} AI-checked.")

    # =====================
    # PHASE 3: EVALUATE (local vector scoring)
    # =====================
    total_eval = len(jobs_with_text)
    for idx, job in enumerate(jobs_with_text, 1):
        if progress_callback:
            progress_callback(idx, total_eval, f"Analyzing AI Semantic Match... ({job.title[:30]})")
        try:
            evaluate_job(session, job, user_context_id, language_result=language_results.get(job.id), entry_result=entry_results.get(job.id), location_result=location_results.get(job.id))
        except Exception as e:
            print(f"    -> [!] Evaluation error: {str(e)}")
            job.status = JobStatus.MANUAL_REVIEW.value
            job.diagnostic_report = f"Değerlendirme hatası: {str(e)}"

    session.commit()

    # =====================
    # PHASE 3.5: AI RELEVANCE RESCUE (batched Gemini second opinion)
    # Only jobs rejected SOLELY because no must-have keyword matched the vector
    # filter are eligible. Hard-nos, language traps, gate-1 junk and aggregator
    # trash never enter this step. The AI prompt also rejects aggregator pages
    # with a good title but trash content, so those stay rejected.
    # =====================
    rescuable = [
        j for j in jobs_with_text
        if j.status == JobStatus.REJECTED.value and j.diagnostic_report == NO_MUST_HAVE_REASON
    ]
    if rescuable:
        print(f"\n[DEBUG] AI RELEVANCE RESCUE: {len(rescuable)} jobs rejected only for missing must-have keywords. Asking Gemini for a second opinion...")
        if progress_callback:
            progress_callback(0, 1, f"AI relevance re-check on {len(rescuable)} postings...")
        from .ai import get_genai_client, MissingApiKeyError
        try:
            client, model_name = get_genai_client(session, user_context_id)
        except MissingApiKeyError as e:
            print(f"    -> [!] {e} — skipping rescue, jobs stay rejected.")
            client, model_name = None, None
        if client:
            rescue_results = batch_relevance_rescue(client, session, user_context_id, rescuable, model=model_name)
            rescued = 0
            for job in rescuable:
                accept, reason = rescue_results.get(job.id, (False, "Not analyzed by model; stays rejected."))
                if accept:
                    job.status = JobStatus.MATCHED.value
                    job.score = AI_RESCUE_SCORE
                    job.diagnostic_report = f"Rescued by AI relevance check: {reason}"
                    rescued += 1
                else:
                    job.diagnostic_report = f"{job.diagnostic_report} | AI relevance check: {reason}"
            print(f"    -> [+] AI rescue: {rescued}/{len(rescuable)} jobs recovered to MATCHED.")
            session.commit()
