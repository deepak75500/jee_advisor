"""
scrapers/perplexity_scraper.py
==============================
Selenium-based scraper that queries Perplexity AI for detailed college
enquiry information. Results are cached in TiDB (scrape_cache table) with
a 24-hour TTL so we don't hammer the site.

The scraper is run in a thread pool (never in the asyncio event loop)
because Selenium is synchronous.

Usage:
    from scrapers.perplexity_scraper import scrape_college_info
    data = scrape_college_info("IIT Bombay", "Computer Science")
"""

import time
import json
import os
import sys
import hashlib
from typing import Dict, Any, Optional

# TiDB cache helpers (imported lazily to avoid circular imports at module load)
def _get_cache():
    from database import db_get_scrape, db_set_scrape
    return db_get_scrape, db_set_scrape


def _make_key(college: str, branch: str = "") -> str:
    raw = f"{college.strip().lower()}|{branch.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# Core scrape function
# ──────────────────────────────────────────────────────────────────────────────
def _do_scrape(query: str, timeout: int = 45) -> str:
    import logging
    log = logging.getLogger("scraper")
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError:
        log.error("selenium not installed")
        return ""

    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        log.info("Chrome driver launched OK")
        wait = WebDriverWait(driver, timeout)

        driver.get("https://www.perplexity.ai/")
        time.sleep(3)
        log.info(f"Page title after load: {driver.title!r}")
        # dump the page for inspection on failure
        driver.save_screenshot("/tmp/perplexity_debug.png")
        with open("/tmp/perplexity_debug.html", "w", encoding="utf-8") as f:
            f.write(driver.page_source)

        dismiss_xpaths = [
            "//button[normalize-space()='Allow all']",
            "//button[contains(text(),'Accept')]",
            "//button[@aria-label='Close']",
        ]
        for xpath in dismiss_xpaths:
            try:
                WebDriverWait(driver, 3).until(EC.element_to_be_clickable((By.XPATH, xpath))).click()
                time.sleep(0.5)
            except Exception:
                pass

        input_selectors = [
            (By.ID, "ask-input"),
            (By.CSS_SELECTOR, "textarea[placeholder*='Ask']"),
            (By.CSS_SELECTOR, "div[contenteditable='true']"),
            (By.XPATH, "//textarea"),
        ]
        input_box = None
        for by, sel in input_selectors:
            try:
                input_box = wait.until(EC.element_to_be_clickable((by, sel)))
                log.info(f"Input box found via {by}={sel}")
                break
            except Exception:
                continue

        if not input_box:
            log.error("No input box found — DOM likely changed. Check /tmp/perplexity_debug.html")
            return ""

        driver.execute_script("arguments[0].scrollIntoView(true);", input_box)
        input_box.click()
        time.sleep(0.5)
        input_box.send_keys(query)
        time.sleep(0.5)
        input_box.send_keys(Keys.ENTER)
        log.info(f"Submitted query: {query[:80]}")

        answer_selectors = [
            (By.XPATH, "//div[starts-with(@id,'markdown-content')]"),
            (By.CSS_SELECTOR, "div.prose"),
            (By.CSS_SELECTOR, "[data-testid='answer']"),
        ]
        answer_el = None
        for by, sel in answer_selectors:
            try:
                answer_el = WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((by, sel))
                )
                WebDriverWait(driver, 20).until(lambda d: len(answer_el.text.strip()) > 50)
                log.info(f"Answer found via {by}={sel}")
                break
            except Exception:
                continue

        if not answer_el:
            driver.save_screenshot("/tmp/perplexity_no_answer.png")
            log.error("No answer element found — check /tmp/perplexity_no_answer.png")
            return ""

        time.sleep(6)
        text = answer_el.text.strip()
        log.info(f"Got {len(text)} chars")
        return text

    except Exception as exc:
        log.exception(f"Scraper crashed: {exc}")
        return ""
    finally:
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def scrape_college_info(college_name: str, branch: str = "") -> Dict[str, Any]:
    """
    Return a dict with scraped college enquiry data.
    Uses TiDB cache; only calls Selenium on a cache miss.
    """
    db_get_scrape, db_set_scrape = _get_cache()
    key = _make_key(college_name, branch)

    cached = db_get_scrape(key)
    if cached:
        print(f"[SCRAPER] Cache hit: {college_name}")
        return cached

    branch_clause = f" for {branch}" if branch else ""
    query = (
        f"Provide detailed college enquiry information about {college_name}{branch_clause} "
        f"in India for JEE aspirants. Include: "
        f"(1) Admission process and JEE cutoff ranks, "
        f"(2) Average, median and highest placement packages with top recruiters, "
        f"(3) Best branches / programmes offered, "
        f"(4) Campus infrastructure, hostels and facilities, "
        f"(5) Research opportunities and faculty quality, "
        f"(6) Student life, clubs and extracurricular activities, "
        f"(7) Notable alumni and industry connections, "
        f"(8) Fee structure and scholarships available, "
        f"(9) Pros and cons for a JEE student choosing this college. "
        f"Give specific numbers and facts wherever possible."
    )

    raw_text = _do_scrape(query)

    result: Dict[str, Any] = {
        "college":    college_name,
        "branch":     branch,
        "query":      query,
        "raw_text":   raw_text,
        "source":     "perplexity_ai",
        "scraped":    True,
        "error":      "" if raw_text else "Scrape failed or timed out",
    }

    if raw_text:
        # parse into sections for structured display
        result["sections"] = _parse_sections(raw_text)

    db_set_scrape(key, result)
    return result


def _parse_sections(text: str) -> Dict[str, str]:
    """
    Best-effort section splitter — looks for numbered headings like
    "1." / "**1." or bold lines and splits the text.
    """
    import re
    sections: Dict[str, str] = {}
    section_names = [
        "admission", "placements", "branches", "campus",
        "research", "student life", "alumni", "fees", "pros and cons",
    ]

    # Try to split on numbered bullets / bold headers
    pattern = re.compile(
        r"(?:^|\n)\s*(?:\*{0,2})(\d+)[.)]\s*\*{0,2}([^\n]{3,60})\*{0,2}",
        re.MULTILINE,
    )
    parts  = pattern.split(text)
    chunks = []
    i = 0
    while i < len(parts):
        chunks.append(parts[i])
        i += 1
        if i + 1 < len(parts):
            header  = parts[i].strip()
            content = parts[i + 1].strip()
            chunks.append((header, content))
            i += 2

    idx = 0
    for chunk in chunks:
        if isinstance(chunk, tuple):
            header, content = chunk
            key = section_names[idx] if idx < len(section_names) else f"section_{idx}"
            sections[key] = f"**{header}**\n{content}"
            idx += 1

    if not sections:
        sections["full"] = text

    return sections
