import re
import os
import json
import pandas as pd
from ddgs import DDGS
from sqlmodel import Session, select
from google import genai
from typing import List, Tuple
from .models import Keyword, KeywordCategory, UserContext, JobListing, JobStatus
from .database import is_job_exists, generate_url_hash
from jobspy import scrape_jobs
from .scraper import scrape_and_evaluate_jobs
from pydantic import BaseModel, Field

from .evaluator import check_title_domain_mismatch

# --- Tunable batching constants ---
PRESCREEN_BATCH_SIZE = 50   # candidates per Gemini pre-screen call
DDGS_MIN_SLEEP = 8.0        # polite delay between DuckDuckGo queries (seconds)
DDGS_MAX_SLEEP = 15.0
DDGS_MAX_CONSECUTIVE_FAILURES = 5  # circuit breaker: DDG rate-limits last ~30-60 min; stop early instead of hammering

# Survival mode: no-qualification job areas, used as the "skills" dimension of the
# search matrix when UserContext.survival_mode is on (CV keywords are ignored then).
def get_survival_job_areas(target_country: str) -> dict:
    country_lower = (target_country or "").lower().strip()
    if country_lower in ["germany", "switzerland", "austria", "almanya", "isviçre", "avusturya"]:
        return {
            "warehouse_production": ["lagerhelfer", "produktionshelfer", "kommissionierer", "hilfskraft", "warehouse worker", "production helper"],
            "gastro_cleaning": ["küchenhilfe", "service mitarbeiter", "reinigungskraft", "kitchen helper", "cleaning staff"],
            "retail_delivery": ["verkäufer", "kassierer", "zusteller", "lieferfahrer", "retail assistant", "delivery driver"],
            "office_data": ["datenerfassung", "bürohilfe", "call center", "data entry", "office assistant"]
        }
    elif country_lower in ["netherlands", "hollanda"]:
        return {
            "warehouse_production": ["magazijnmedewerker", "productiemedewerker", "orderpicker", "hulpkracht", "warehouse worker", "production helper"],
            "gastro_cleaning": ["keukenhulp", "schoonmaker", "schoonmaakmedewerker", "bediening", "kitchen helper", "cleaning staff"],
            "retail_delivery": ["verkoopmedewerker", "kassamedewerker", "bezorger", "koerier", "retail assistant", "delivery driver"],
            "office_data": ["data entry", "administratief medewerker", "callcenter medewerker", "office assistant"]
        }
    elif country_lower in ["turkey", "türkiye"]:
        return {
            "warehouse_production": ["depo elemanı", "üretim elemanı", "vasıfsız eleman", "yardımcı personel", "warehouse worker"],
            "gastro_cleaning": ["mutfak elemanı", "temizlik görevlisi", "garson", "komi", "kitchen helper", "cleaning staff"],
            "retail_delivery": ["satış danışmanı", "kasiyer", "kurye", "dağıtım elemanı", "retail assistant", "delivery driver"],
            "office_data": ["veri girişi", "çağrı merkezi", "büro elemanı", "data entry", "office assistant"]
        }
    else:
        return {
            "warehouse_production": ["warehouse worker", "production helper", "packer", "laborer"],
            "gastro_cleaning": ["kitchen helper", "cleaning staff", "waiter", "dishwasher", "cleaner"],
            "retail_delivery": ["retail assistant", "cashier", "delivery driver", "courier"],
            "office_data": ["data entry", "office assistant", "call center agent", "clerk"]
        }

def get_survival_area_labels(target_country: str) -> dict:
    country_lower = (target_country or "").lower().strip()
    if country_lower in ["germany", "switzerland", "austria", "almanya", "isviçre", "avusturya"]:
        return {
            "warehouse_production": "Warehouse & production (Lagerhelfer, Produktionshelfer, Hilfskraft)",
            "gastro_cleaning": "Gastro & cleaning (Küchenhilfe, Service, Reinigung)",
            "retail_delivery": "Retail & delivery (Verkäufer, Kassierer, Zusteller)",
            "office_data": "Office & data entry (Datenerfassung, Bürohilfe, Call Center)"
        }
    elif country_lower in ["netherlands", "hollanda"]:
        return {
            "warehouse_production": "Warehouse & production (Magazijnmedewerker, Productiemedewerker)",
            "gastro_cleaning": "Gastro & cleaning (Keukenhulp, Schoonmaker)",
            "retail_delivery": "Retail & delivery (Verkoopmedewerker, Bezorger)",
            "office_data": "Office & data entry (Data entry, Callcenter)"
        }
    elif country_lower in ["turkey", "türkiye"]:
        return {
            "warehouse_production": "Warehouse & production (Depo elemanı, Üretim elemanı, Vasıfsız)",
            "gastro_cleaning": "Gastro & cleaning (Mutfak elemanı, Temizlik görevlisi)",
            "retail_delivery": "Retail & delivery (Satış danışmanı, Kasiyer, Kurye)",
            "office_data": "Office & data entry (Veri girişi, Çağrı merkezi)"
        }
    else:
        return {
            "warehouse_production": "Warehouse & production (Warehouse worker, Packer, Laborer)",
            "gastro_cleaning": "Gastro & cleaning (Kitchen helper, Cleaner, Dishwasher)",
            "retail_delivery": "Retail & delivery (Retail assistant, Cashier, Courier)",
            "office_data": "Office & data entry (Data entry, Call center, Clerk)"
        }

def get_survival_caption(target_country: str) -> str:
    country_lower = (target_country or "").lower().strip()
    if country_lower in ["germany", "switzerland", "austria", "almanya", "isviçre", "avusturya"]:
        return "Language filter in this mode: no requirement or up to B1 accepted — 'fluent/C1/verhandlungssicher' rejected."
    elif country_lower in ["netherlands", "hollanda"]:
        return "Language filter in this mode: no requirement or up to B1 accepted — 'fluent/C1/vloeiend' rejected."
    elif country_lower in ["turkey", "türkiye"]:
        return "Language filter in this mode: no requirement or up to B1 accepted — 'fluent/C1/ileri düzey' rejected."
    else:
        return "Language filter in this mode: no requirement or up to B1 accepted — 'fluent/C1' rejected."
SURVIVAL_FIELD_DESC = "any no-qualification job (warehouse, production help, gastronomy, cleaning, retail, delivery, data entry, office support)"

class PreScreenItem(BaseModel):
    index: int = Field(description="The ID number of the search result as given in the input.")
    accept: bool = Field(description="True if this is a link to a real, single job vacancy worth scraping.")
    reason: str = Field(description="Short reason for the decision. Empty string when accepting.")

class PreScreenBatch(BaseModel):
    results: List[PreScreenItem]

PRESCREEN_PROMPT = """
You are filtering web search results to find REAL job vacancies.
The candidate's field: {field}.

You will receive numbered search results (Title + Snippet + URL). For each one decide: is this a link to a SINGLE, specific job vacancy?

REJECT (accept=false) ONLY when the result clearly is one of these:
1. An aggregator page: a search-results page, job directory, listicle ("10 best jobs..."), or a general portal/corporate hub.
2. Not a job posting at all: Wikipedia, dictionary, blog post, news article, rankings, forum thread.
3. An academic program: university degree pages, master/bachelor courses, student projects. NOTE: a paid PhD/Postdoc POSITION at a lab or company is a job - accept those.
4. A clear seniority mismatch: the TITLE itself says Senior/Lead/Principal/Manager/Director while the candidate targets intern/junior level.
5. A clear functional/domain mismatch: the TITLE belongs to a completely different discipline than {field} (e.g. candidate targets Industrial/Process/Supply Chain engineering, but title is IT Network Engineer, Software Developer, Sales Specialist, Accountant, Nurse, Legal Counsel).

ACCEPT (accept=true) everything else:
- A title naming a specific role at a specific company is almost always a real vacancy, even if the snippet is noisy or lists other jobs in a sidebar.
- When in doubt, ACCEPT. Later stages scrape and re-verify the page for free; a false rejection loses a real job forever.

Give a short reason for every rejection. Use an empty reason when accepting.
"""

# Word-boundary seniority detector for job titles. "sr" covered via \bsr\b;
# \bhead\b does not match "headquarters", \blead\b does not match "leader".
_SENIORITY_TITLE_RE = re.compile(r'\b(senior|sr|lead|principal|manager|director|head)\b', re.IGNORECASE)

def title_is_senior(title: str) -> bool:
    """Free title-based seniority check (intern/junior target). Used on JobSpy rows
    at ingest — the DDG path had its own hard-no filter, JobSpy had none, which let
    Senior/Manager roles straight into Matched (2026-07-24 review)."""
    return bool(_SENIORITY_TITLE_RE.search(title or ""))

def _ingest_jobspy_rows(session: Session, df, user_context_id: int) -> Tuple[List[JobListing], int]:
    """Insert new JobListing rows from a jobspy dataframe. Returns (added_jobs, raw_link_count)."""
    ctx = session.get(UserContext, user_context_id)
    survival_mode = bool(ctx and getattr(ctx, 'survival_mode', False))

    statement_mh = select(Keyword).where(Keyword.user_context_id == user_context_id).where(Keyword.category == KeywordCategory.MUST_HAVE.value)
    must_haves = [k.word for k in session.exec(statement_mh).all()]

    added = []
    raw = 0
    for index, row in df.iterrows():
        url = row.get('job_url', '')
        if not url or pd.isna(url):
            continue

        raw += 1
        if not is_job_exists(session, url):
            desc = row.get('description', '')
            if pd.isna(desc): desc = ""
            title = row.get('title', '')
            if pd.isna(title): title = "Job Listing"
            company = row.get('company', '')
            if pd.isna(company): company = ""

            # Career mode: senior-titled jobs never fit an intern/junior/graduate target
            if not survival_mode and title_is_senior(str(title)):
                session.add(JobListing(
                    url=url, url_hash=generate_url_hash(url), title=f"{title} at {company}",
                    location=str(row.get('location', '') or '') or None,
                    snippet=str(desc)[:200], full_text=str(desc),
                    status=JobStatus.REJECTED.value,
                    diagnostic_report="Early Python Rejection: seniority in title (JobSpy ingest filter)",
                    user_context_id=user_context_id
                ))
                continue

            # Career mode: title domain mismatch check (e.g. IT/Network Engineering for non-IT candidate)
            if not survival_mode and must_haves:
                is_mismatch, mismatch_reason = check_title_domain_mismatch(str(title), must_haves)
                if is_mismatch:
                    session.add(JobListing(
                        url=url, url_hash=generate_url_hash(url), title=f"{title} at {company}",
                        location=str(row.get('location', '') or '') or None,
                        snippet=str(desc)[:200], full_text=str(desc),
                        status=JobStatus.REJECTED.value,
                        diagnostic_report=f"Early Python Rejection: {mismatch_reason} (JobSpy ingest filter)",
                        user_context_id=user_context_id
                    ))
                    continue


            new_job = JobListing(
                url=url,
                url_hash=generate_url_hash(url),
                title=f"{title} at {company}",
                location=str(row.get('location', '') or '') or None,
                snippet=str(desc)[:200],
                full_text=str(desc),
                status=JobStatus.PENDING_SCRAPE.value,
                user_context_id=user_context_id
            )
            session.add(new_job)
            added.append(new_job)

    session.commit()
    return added, raw


def is_single_job_url(url: str) -> bool:
    """
    URL yapısına bakarak gerçek bir tekil ilan mı yoksa 
    çöplük bir arama dizini mi olduğunu söyler.
    """
    url_lower = url.lower()
    
    # 1. Agregatörlerin ARAMA/LİSTELEME sayfaları (Kesinlikle Reddet)
    blocking_patterns = [
        r"indeed\.com/q-", r"indeed\.com/jobs\?q=",
        r"linkedin\.com/jobs/search", r"linkedin\.com/jobs/collections",
        r"glassdoor\.com/job/.*-jobs-",
        r"jobs\.ch/.*/stellenangebote",
        r"careerjet\.ch/stellenangebote",
        r"/search/jobs", r"/jobs/search", r"jobs-in-", r"job_search",
        r"search\?", r"search-jobs", r"job-search",
        r"euroengineerjobs\.com/job_search",
        r"jobscout24\.ch/.*/jobs-in-"
    ]
    
    for pattern in blocking_patterns:
        if re.search(pattern, url_lower):
            return False # Arama sonucu sayfası, ele.

    # 2. Eğer bilinen bir kariyer sitesiyse, SADECE tekil ilan formatındaysa izin ver
    if "indeed.com" in url_lower and not ("viewjob" in url_lower or "rc/clk" in url_lower):
        return False
    if "linkedin.com" in url_lower and "/jobs/view/" not in url_lower:
        return False
    if "glassdoor.com" in url_lower and "/job-listing/" not in url_lower:
        return False

    return True

def get_default_job_terms(country: str) -> str:
    country_lower = country.replace('İ', 'i').replace('I', 'ı').strip().lower()
    region_map = {
        "almanya": "de-de", "germany": "de-de", "isviçre": "ch-de", "isvicre": "ch-de",
        "switzerland": "ch-de", "avusturya": "at-de", "austria": "at-de", "amerika": "us-en",
        "usa": "us-en", "united states": "us-en", "ingiltere": "uk-en", "uk": "uk-en",
        "united kingdom": "uk-en", "türkiye": "tr-tr", "turkey": "tr-tr", "hollanda": "nl-nl",
        "netherlands": "nl-nl"
    }
    job_terms_map = {
        "de-de": "job, jobs, stellen, karriere, stellenangebote",
        "ch-de": "job, jobs, stellen, karriere, emploi",
        "at-de": "job, jobs, stellen, karriere",
        "nl-nl": "vacature, vacatures, baan, werk, werken",
        "tr-tr": "iş, iş ilanı, kariyer, iş ilanları"
    }
    region_code = region_map.get(country_lower, "wt-wt")
    return job_terms_map.get(region_code, "job, hiring, vacancies, careers")

def discover_jobs(session: Session, user_context_id: int, max_results: int = 50, custom_job_terms: str = None, progress_callback=None) -> Tuple[List[JobListing], int, List[Tuple[str, str]]]:
    ctx = session.get(UserContext, user_context_id)
    if not ctx:
        return [], 0, []

    survival_mode = getattr(ctx, 'survival_mode', False)

    statement_sp = select(Keyword).where(Keyword.user_context_id == user_context_id).where(Keyword.category == KeywordCategory.SEARCH_PHRASE.value)
    search_phrase_keywords = session.exec(statement_sp).all()
    
    statement_mh = select(Keyword).where(Keyword.user_context_id == user_context_id).where(Keyword.category == KeywordCategory.MUST_HAVE.value)
    must_have_keywords = session.exec(statement_mh).all()

    statement_hn = select(Keyword).where(Keyword.user_context_id == user_context_id).where(Keyword.category == KeywordCategory.HARD_NO.value)
    hard_no_keywords = session.exec(statement_hn).all()

    if survival_mode:
        # Survival mode: job areas replace CV keywords as the search "skills".
        survival_areas_dict = get_survival_job_areas(ctx.target_country)
        area_keys = [a.strip() for a in (getattr(ctx, 'survival_areas', '') or '').split(',') if a.strip()]
        if not area_keys:
            area_keys = list(survival_areas_dict.keys())
        all_synonyms = []
        for key in area_keys:
            all_synonyms.extend(survival_areas_dict.get(key, []))
        if not all_synonyms:
            all_synonyms = [t for terms in survival_areas_dict.values() for t in terms]
        search_phrases = all_synonyms
    else:
        if not search_phrase_keywords:
            return [], 0, []

        search_phrases = [kw.word.strip() for kw in search_phrase_keywords if kw.word.strip()]
        
    country_lower = ctx.target_country.replace('İ', 'i').replace('I', 'ı').strip().lower()
    
    jobspy_country_map = {
        "almanya": "germany", "germany": "germany", 
        "isviçre": "switzerland", "isvicre": "switzerland", "switzerland": "switzerland", 
        "avusturya": "austria", "austria": "austria", 
        "amerika": "usa", "usa": "usa", "united states": "usa", 
        "ingiltere": "uk", "uk": "uk", "united kingdom": "uk", 
        "türkiye": "turkey", "turkey": "turkey", 
        "hollanda": "netherlands", "netherlands": "netherlands"
    }
    jobspy_country = jobspy_country_map.get(country_lower, "worldwide")
    
    region_map = {
        "almanya": "de-de", "germany": "de-de", "isviçre": "ch-de", "isvicre": "ch-de",
        "switzerland": "ch-de", "avusturya": "at-de", "austria": "at-de", "amerika": "us-en",
        "usa": "us-en", "united states": "us-en", "ingiltere": "uk-en", "uk": "uk-en",
        "united kingdom": "uk-en", "türkiye": "tr-tr", "turkey": "tr-tr", "hollanda": "nl-nl",
        "netherlands": "nl-nl"
    }
    region_code = region_map.get(country_lower, "wt-wt")
    
    added_jobs = []
    total_raw_links = 0
    failed_sites: List[Tuple[str, str]] = []
    
    # =================================================================
    # PHASE 1A: JOBSPY INTEGRATION (Primary Job Boards)
    # =================================================================
    print("\n" + "="*50)
    print(f"[DEBUG] JOBSPY INTEGRATION ACTIVE")
    
    country_terms_map = {
        "de-de": ["Germany", "Deutschland"],
        "ch-de": ["Switzerland", "Schweiz", "Suisse"],
        "at-de": ["Austria", "Österreich"],
        "us-en": ["USA", "United States"],
        "uk-en": ["UK", "United Kingdom", "England"],
        "nl-nl": ["Netherlands", "Nederland"],
        "tr-tr": ["Turkey", "Türkiye"]
    }
    
    # Strictly enforce place NOUNS only (filter out adjective forms like 'Swiss', 'German', 'French')
    BANNED_COUNTRY_ADJECTIVES = {"swiss", "german", "french", "dutch", "turkish", "american", "british", "austrian"}
    raw_country_terms = country_terms_map.get(region_code, [])
    country_terms_list = [c for c in raw_country_terms if c.lower() not in BANNED_COUNTRY_ADJECTIVES]
    
    # Validation is already done at extraction time, we just take the phrases
    valid_search_phrases = list(dict.fromkeys(search_phrases))
    
    # Limit JobSpy to the top 3 AI-composed search phrases to avoid rate limits
    jobspy_phrases = valid_search_phrases[:3] if valid_search_phrases else [""]

    import os
    proxy_url = os.environ.get("PROXY_URL")

    default_sites = "linkedin,indeed,google"
    JOBSPY_SITES = [s.strip() for s in os.environ.get("JOBSPY_SITES", default_sites).split(",") if s.strip()]

    # Calculate JobSpy's slice of the target quota based on the user's slider
    jobspy_ratio = getattr(ctx, 'jobspy_ratio', 80)
    jobspy_cap = int(max_results * (jobspy_ratio / 100.0))
    # Overfetch aggressively per site to survive the deduplication filter (max 75 to avoid hard blocks)
    total_per_site_wanted = min(75, max(30, jobspy_cap))
    per_query_wanted = max(10, total_per_site_wanted // max(1, len(jobspy_phrases)))

    GENTLE_SITES = {"google"}
    google_wanted = int(os.environ.get("JOBSPY_GOOGLE_WANTED", "10"))

    JOBSPY_SITES = [s for s in JOBSPY_SITES if s not in GENTLE_SITES] + [s for s in JOBSPY_SITES if s in GENTLE_SITES]

    print(f"[DEBUG] JobSpy Cap: {jobspy_cap}/{max_results} total jobs, {per_query_wanted} per phrase/site, {google_wanted} for gentle sites...")

    for phrase_idx, search_term in enumerate(jobspy_phrases):
        if len(added_jobs) >= jobspy_cap:
            break
            
        print(f"\n[DEBUG] --- JobSpy Phrase {phrase_idx+1}/{len(jobspy_phrases)}: '{search_term}' ---")
        for site_idx, site in enumerate(JOBSPY_SITES):
            if len(added_jobs) >= jobspy_cap:
                print(f"    [⏹] JobSpy cap reached ({len(added_jobs)}/{jobspy_cap}). Skipping remaining boards.")
                break
            try:
                wanted = per_query_wanted
                if site in GENTLE_SITES:
                    import time as _time, random as _random
                    wanted = google_wanted
                    delay = _random.uniform(5.0, 10.0)
                    print(f"    [DEBUG] Gentle mode for {site}: sleeping {delay:.1f}s, results_wanted={wanted} (1-2 requests max)")
                    _time.sleep(delay)

                jobspy_results = scrape_jobs(
                    site_name=[site],
                    search_term=search_term,
                    location=ctx.target_country,
                    results_wanted=wanted,
                    country_indeed=jobspy_country,
                    linkedin_fetch_description=True,
                    hours_old=720, # Limit to jobs posted in the last 30 days
                    proxies=[proxy_url] if proxy_url else None
                )

                print(f"    [+] JobSpy [{site}] returned {len(jobspy_results)} results.")

                site_added, site_raw = _ingest_jobspy_rows(session, jobspy_results, user_context_id)
                added_jobs.extend(site_added)
                total_raw_links += site_raw
            except Exception as e:
                err_str = str(e)
                print(f"    [DEBUG] JobSpy [{site}] failed: {e}")
                if site in GENTLE_SITES and ("sorry" in err_str.lower() or "429" in err_str):
                    failed_sites.append((site, f"CAPTCHA_BLOCKED:{err_str[:120]}"))
                else:
                    failed_sites.append((site, err_str))

    # =================================================================
    # PHASE 1B: DUCKDUCKGO MULTI-QUERY (Combinatorial Matrix)
    # =================================================================
    import itertools
    import time
    import random
    
    # Map UI Seniority Level to negative DDG operators
    seniority_exclusions_map = {
        "Internship": " -manager -senior -lead -head -director -principal",
        "Graduate": " -manager -senior -lead -head -director -principal",
        "Junior": " -manager -senior -lead -head -director -principal",
        "Mid-Level": " -senior -director -principal -head",
        "Senior": "",
        "Lead": ""
    }
    user_seniority = getattr(ctx, 'seniority_level', 'Junior')
    negative_seniority = seniority_exclusions_map.get(user_seniority, "") if not survival_mode else " -manager -senior -lead -head -director -principal"
    exclusion_operators = f"-site:linkedin.com -site:indeed.com -site:glassdoor.com -site:esco.ec.europa.eu{negative_seniority}"
    
    # Calculate Search Matrix (AI Search Phrases x Country Terms)
    queries = []
    if not country_terms_list:
        country_terms_list = [""]

    for phrase, country in itertools.product(valid_search_phrases, country_terms_list):
        clean_phrase = phrase.replace('"', '').strip()
        if country:
            queries.append((f'"{clean_phrase}" {country} {exclusion_operators}', country))
        else:
            queries.append((f'"{clean_phrase}" {exclusion_operators}', ""))
            
    # Randomize to prevent hitting the exact same pattern if we cap
    random.shuffle(queries)
    

    total_combinations = len(queries)
    
    # Cap at 30 to avoid DDGS IP blocks, but allow running all queries up to that limit
    MAX_DDGS_QUERIES = min(30, total_combinations)
    
    if len(queries) > MAX_DDGS_QUERIES:
        queries = queries[:MAX_DDGS_QUERIES]
    
    print("\n" + "="*50)
    print(f"[DEBUG] DUCKDUCKGO INTERLEAVED MATRIX ACTIVE")
    print(f"[DEBUG] Region: {region_code}")
    print(f"[DEBUG] Total combinations calculated: {len(valid_search_phrases)} SearchPhrases x {len(country_terms_list)} CountryTerms = {total_combinations}")
    print(f"[DEBUG] Executing {len(queries)} matrix queries (Limit: {MAX_DDGS_QUERIES})...")
    print("="*50 + "\n")
    
    seen_urls = set()
    remaining_target = max(10, max_results - len(added_jobs))
    # Massively increase the depth multiplier to counteract high duplicate rates (e.g. 80% duplicates)
    # We fetch up to 40 results per query so even if 35 are duplicates, 5 survive.
    per_query_limit = min(50, max(25, (remaining_target * 5) // max(1, len(queries))))

    from .ai import get_genai_client
    client, model_name = get_genai_client(session, user_context_id)
    
    def check_match_early(keyword_str: str, text: str) -> str:
        synonyms = [s.strip() for s in keyword_str.split(',') if s.strip()]
        if not synonyms:
            return ""
        pattern = r'\b(' + '|'.join(re.escape(s) for s in synonyms) + r')\b'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return match.group(0) if match else ""
        
    # Free Python pre-filter accumulates candidates; AI review happens AFTER all searching
    all_ai_candidates = []
    consecutive_failures = 0

    try:
        with DDGS() as ddgs:
            for idx, (q, q_country) in enumerate(queries):
                # Early stop: enough jobs collected (JobSpy + DDGS candidates), save tokens/time
                collected = len(added_jobs) + len(all_ai_candidates)
                if collected >= max_results:
                    print(f"[⏹] Target reached: {collected}/{max_results} jobs collected. Stopping search early.")
                    break

                print(f"[{idx+1}/{len(queries)}] Ping: {q}")
                count = 0
                query_results = []
                query_ok = False
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        for r in ddgs.text(q, region=region_code, max_results=per_query_limit):
                            url = r.get("href", "")
                            if not url:
                                continue

                            total_raw_links += 1
                            count += 1

                            if not is_single_job_url(url):
                                continue

                            if url not in seen_urls and not is_job_exists(session, url):
                                seen_urls.add(url)
                                query_results.append({
                                    "title": r.get("title", ""),
                                    "snippet": r.get("body", ""),
                                    "url": url,
                                    "location_hint": q_country or None
                                })
                        query_ok = True
                        break # Success! Exit the retry loop
                    except Exception as e:
                        if attempt < max_retries - 1:
                            backoff = 5.0 * (attempt + 1)
                            print(f"    [!] DDGS blocked ({e}). Sleeping {backoff:.0f}s and retrying (Attempt {attempt+2}/{max_retries})...")
                            import time
                            time.sleep(backoff)
                        else:
                            print(f"    [X] DDGS failed completely after {max_retries} attempts.")

                print(f"    -> Returned {count} results for this query.")

                # Circuit breaker: DDG rate-limits last a while; stop the search
                # phase early instead of burning retries on 25 more dead queries.
                if query_ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= DDGS_MAX_CONSECUTIVE_FAILURES:
                        print(f"    [⛔] {consecutive_failures} consecutive DDGS failures — DuckDuckGo is rate-limiting this IP.")
                        print(f"    [⛔] Stopping search phase early; moving on with the {len(all_ai_candidates)} candidates collected so far.")
                        try:
                            import streamlit as st
                            st.warning("⚠️ **DuckDuckGo Rate Limit Reached.** The search engine blocked our IP. We are stopping the search phase early and safely proceeding with the jobs collected so far. To get more results, please wait ~30 minutes for the block to clear, or connect to a VPN.")
                        except Exception:
                            pass
                        break

                if query_results:
                    for r in query_results:
                        combined_text = f"{r['title']} {r['snippet']}".lower()
                        title_text = r['title'].lower()
                        found_hard_no = ""
                        for hk in hard_no_keywords:
                            seniority_words = ["senior", "director", "manager", "lead", "head", "principal", "sr", "sr."]
                            is_seniority = any(sw in hk.word.lower() for sw in seniority_words)
                            target_text = title_text if is_seniority else combined_text

                            check_word = hk.word
                            if "senior" in check_word.lower() and "sr" not in check_word.lower():
                                check_word += ", sr, sr."

                            matched_word = check_match_early(check_word, target_text)
                            if matched_word:
                                found_hard_no = matched_word
                                break

                        if not found_hard_no:
                            academic_words = ["master in", "msc", "bsc", "bachelor in", "student project", "phd program", "online msc", "degree program", "master's program"]
                            for aw in academic_words:
                                if aw in title_text:
                                    found_hard_no = f"Academic Program ({aw})"
                                    break

                        if found_hard_no:
                            rejected_job = JobListing(
                                url=r["url"], url_hash=generate_url_hash(r["url"]), title=r["title"], location=(f"query:{r['location_hint']}" if r.get("location_hint") else None), snippet=r["snippet"],
                                status=JobStatus.REJECTED.value, diagnostic_report=f"Early Python Rejection: Found '{found_hard_no}' (Hard No)",
                                user_context_id=user_context_id
                            )
                            session.add(rejected_job)
                        else:
                            all_ai_candidates.append(r)
                    session.commit()

                # Polite sleep between pings to prevent rate limit
                time.sleep(random.uniform(DDGS_MIN_SLEEP, DDGS_MAX_SLEEP))
    except Exception as e:
        print(f"[DEBUG] DDGS Error: {e}")

    # =================================================================
    # PHASE 2: BATCH AI PRE-SCREEN (one Gemini call per PRESCREEN_BATCH_SIZE candidates)
    # =================================================================
    if all_ai_candidates:
        print("\n" + "="*50)
        print(f"[DEBUG] BATCH AI PRE-SCREEN: {len(all_ai_candidates)} candidates, batches of {PRESCREEN_BATCH_SIZE}")
        print("="*50 + "\n")
        from google.genai import types
        field_desc = SURVIVAL_FIELD_DESC if survival_mode else [k.word for k in must_have_keywords]
        prompt = PRESCREEN_PROMPT.format(field=field_desc)

        for start in range(0, len(all_ai_candidates), PRESCREEN_BATCH_SIZE):
            batch = all_ai_candidates[start:start + PRESCREEN_BATCH_SIZE]
            jobs_text = ""
            for j_idx, r in enumerate(batch):
                jobs_text += f"ID: {j_idx}\nTitle: {r['title']}\nSnippet: {r['snippet']}\nURL: {r['url']}\n---\n"

            print(f"[DEBUG] Pre-screen batch {start // PRESCREEN_BATCH_SIZE + 1}: {len(batch)} candidates...")
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt + "\n\n" + jobs_text,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=PreScreenBatch,
                        temperature=0.0
                    )
                )
                decisions = {item.index: item for item in response.parsed.results}
                for j_idx, r in enumerate(batch):
                    item = decisions.get(j_idx)
                    # Missing from model output => accept by default (when in doubt, accept)
                    if item is None or item.accept:
                        new_job = JobListing(
                            url=r["url"], url_hash=generate_url_hash(r["url"]), title=r["title"], location=(f"query:{r['location_hint']}" if r.get("location_hint") else None), snippet=r["snippet"],
                            status=JobStatus.PENDING_SCRAPE.value,
                            user_context_id=user_context_id
                        )
                        session.add(new_job)
                        added_jobs.append(new_job)
                    else:
                        rejected_job = JobListing(
                            url=r["url"], url_hash=generate_url_hash(r["url"]), title=r["title"], location=(f"query:{r['location_hint']}" if r.get("location_hint") else None), snippet=r["snippet"],
                            status=JobStatus.REJECTED.value, diagnostic_report=f"AI Rejection: {item.reason}",
                            user_context_id=user_context_id
                        )
                        session.add(rejected_job)
                session.commit()
            except Exception as e:
                print(f"[DEBUG] Gemini batch failed: {e}. Accepting batch by default so real jobs are not lost.")
                for r in batch:
                    new_job = JobListing(
                        url=r["url"], url_hash=generate_url_hash(r["url"]), title=r["title"], location=(f"query:{r['location_hint']}" if r.get("location_hint") else None), snippet=r["snippet"],
                        status=JobStatus.PENDING_SCRAPE.value,
                        user_context_id=user_context_id
                    )
                    session.add(new_job)
                    added_jobs.append(new_job)
                session.commit()

    # =================================================================
    # PHASE 3+4: SCRAPE & EVALUATE (once, after all searching is done)
    # =================================================================
    try:
        scrape_and_evaluate_jobs(session, user_context_id, progress_callback)
    except Exception as e:
        print(f"[DEBUG] Scraper evaluation failed: {e}")

    return added_jobs, total_raw_links, failed_sites


def search_google_only(session: Session, user_context_id: int, max_results: int = 10) -> Tuple[int, str]:
    """Gentle single-channel Google retry (1-2 HTTP requests max).
    Used by the UI after the user manually solves Google's /sorry/ captcha,
    which unflags the IP for subsequent requests.
    Returns (added_count, error_message). error_message is "" on success."""
    ctx = session.get(UserContext, user_context_id)
    if not ctx:
        return 0, "User context not found."

    statement = select(Keyword).where(Keyword.user_context_id == user_context_id).where(Keyword.category == KeywordCategory.MUST_HAVE.value)
    must_have_keywords = session.exec(statement).all()
    if not must_have_keywords:
        return 0, "No must-have keywords saved for this profile yet."

    primary_skill = must_have_keywords[0].word.split(',')[0].strip()
    proxy_url = os.environ.get("PROXY_URL")

    import time
    import random
    try:
        delay = random.uniform(5.0, 10.0)
        print(f"[DEBUG] Google retry: gentle mode, sleeping {delay:.1f}s before the call...")
        time.sleep(delay)

        jobspy_results = scrape_jobs(
            site_name=["google"],
            search_term=primary_skill,
            location=ctx.target_country,
            results_wanted=min(10, max_results),  # 10 jobs per page => 1-2 requests
            linkedin_fetch_description=True,
            proxies=[proxy_url] if proxy_url else None
        )

        added, raw = _ingest_jobspy_rows(session, jobspy_results, user_context_id)
        print(f"[DEBUG] Google retry succeeded: {len(added)} new jobs from {raw} raw links.")
        return len(added), ""
    except Exception as e:
        print(f"[DEBUG] Google retry failed: {e}")
        return 0, str(e)
