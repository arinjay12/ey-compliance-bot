"""
level2.py
---------
Level 2 Feature: Automated daily comparison between SEBI RSS feed
and stored documents in ChromaDB.

What it does:
  1. Fetches the SEBI RSS feed every day
  2. Finds new circulars published in the last 24 hours
  3. Checks if any new circular relates to topics in your stored documents
  4. If it does, asks Llama 3 to identify gaps or conflicts
  5. Sends an email alert automatically if anything is found

Usage:
  Run once manually (for testing):
      python level2.py

  Run on daily schedule (keep this running in background):
      python level2.py --schedule
"""

import os
import json
import time
import smtplib
import argparse
import schedule
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import requests
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load credentials from a local .env file if present (never hard-coded).
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── CONFIG — edit these before running ────────────────────────────────────────

SEBI_RSS_URL = "https://www.sebi.gov.in/sebirss.xml"
RBI_URL      = "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx"
CHROMA_DIR   = "./chroma_db"
EMBED_MODEL  = "all-MiniLM-L6-v2"
LLM_MODEL    = "llama3"

# Email settings — read from environment / .env, never hard-coded. Copy
# .env.example to .env and fill in these three keys (see README for App Password):
#   EY_BOT_SENDER_EMAIL, EY_BOT_SENDER_PASSWORD, EY_BOT_RECEIVER_EMAIL
SENDER_EMAIL    = os.getenv("EY_BOT_SENDER_EMAIL", "")
SENDER_PASSWORD = os.getenv("EY_BOT_SENDER_PASSWORD", "")
RECEIVER_EMAIL  = os.getenv("EY_BOT_RECEIVER_EMAIL", "")

# How far back to look for new circulars (in hours)
# Set to 720 (30 days) for testing. Change back to 24 for production.
LOOKBACK_HOURS = 720

# File to track circulars we've already processed (avoids duplicate alerts)
SEEN_FILE   = "./seen_circulars.json"
STATUS_FILE = "./level2_status.json"   # read by app.py for the UI panel

# Minimum similarity score to consider a circular "related" to stored docs
# ChromaDB uses L2 distance: 0 = identical, higher = less similar
# 0.6 = strict (only very close matches), 1.5 = permissive (broader matches)
SIMILARITY_THRESHOLD = 1.5

# ── HEADERS for web requests ──────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Fetch and parse the SEBI RSS feed
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rss() -> list:
    """
    Fetches the SEBI RSS feed and returns a list of recent circulars
    as dicts: {title, link, date, description}
    """
    circulars = []
    cutoff    = datetime.now(timezone.utc) - timedelta(hours=LOOKBACK_HOURS)

    # ── Source 1: SEBI RSS feed ────────────────────────────────────────────────
    print(f"[{now()}] Fetching SEBI RSS feed...")
    try:
        resp = requests.get(SEBI_RSS_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "xml")
        items = soup.find_all("item")
        print(f"  SEBI RSS: {len(items)} total items")

        for item in items:
            title     = item.find("title")
            link      = item.find("link")
            desc      = item.find("description")
            pub       = item.find("pubDate")
            title_text = title.text.strip() if title else ""
            link_text  = link.text.strip()  if link  else ""
            desc_text  = desc.text.strip()  if desc  else ""
            date_text  = pub.text.strip()   if pub   else ""
            pub_date   = parse_date(date_text)
            if pub_date and pub_date < cutoff:
                continue
            circulars.append({
                "title": title_text, "link": link_text,
                "description": desc_text, "date": date_text,
                "pub_date": pub_date, "source": "SEBI",
            })
    except Exception as e:
        print(f"  SEBI RSS ERROR: {e}")

    # ── Source 2: RBI Circulars page (confirmed accessible) ───────────────────
    print(f"[{now()}] Fetching RBI circulars page...")
    try:
        resp = requests.get(RBI_URL, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        rows  = soup.select("table tr")
        count = 0
        for row in rows[1:]:   # skip header row
            cols = row.find_all("td")
            if len(cols) < 4:
                continue
            # RBI table: col0=reference+link, col1=date, col2=dept, col3=subject/title
            ref_cell   = cols[0]
            date_text  = cols[1].get_text(strip=True)
            title_text = cols[3].get_text(strip=True)
            a_tag      = ref_cell.find("a")
            href       = a_tag["href"] if a_tag and a_tag.get("href") else ""
            link_text  = ("https://www.rbi.org.in" + href) if href.startswith("/") else href
            pub_date   = parse_date(date_text)
            if pub_date and pub_date < cutoff:
                continue
            circulars.append({
                "title": title_text, "link": link_text,
                "description": "", "date": date_text,
                "pub_date": pub_date, "source": "RBI",
            })
            count += 1
        print(f"  RBI: {count} recent circular(s) found")
    except Exception as e:
        print(f"  RBI ERROR: {e}")

    print(f"  Total new items to check: {len(circulars)} (last {LOOKBACK_HOURS}h)")
    return circulars


def parse_date(date_str: str):
    """Parse RSS pubDate string into a timezone-aware datetime."""
    if not date_str:
        return None
    formats = [
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Try to fetch the full text of a circular's page
# ─────────────────────────────────────────────────────────────────────────────

def fetch_circular_text(url: str) -> str:
    """
    Try to get the full text of a circular from its SEBI page.
    Returns empty string if blocked (403) — we fall back to title + description.
    """
    if not url:
        return ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [l.strip() for l in soup.get_text("\n").splitlines() if l.strip()]
        return "\n".join(lines)[:3000]
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Compare new circular against stored documents
# ─────────────────────────────────────────────────────────────────────────────

def find_conflicts(circular: dict, embeddings, llm) -> str | None:
    """
    1. Searches ChromaDB for stored chunks related to this circular
    2. If related content found, asks Llama 3 to identify conflicts/gaps
    3. Returns conflict description string, or None if no issues found
    """
    # Build the text we'll compare — title + description + page text (if available)
    circular_text = circular["title"]
    if circular["description"] and circular["description"] != circular["title"]:
        circular_text += "\n" + circular["description"]

    page_text = fetch_circular_text(circular["link"])
    if page_text:
        circular_text += "\n" + page_text
        print(f"  Fetched full page text ({len(page_text)} chars)")
    else:
        print(f"  Using RSS title/description only (page blocked)")

    # Search ChromaDB for stored chunks related to this circular
    vs      = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    results = vs.similarity_search_with_score(circular_text, k=4)

    # Filter to only chunks that are actually related (low distance = high similarity)
    related = [(doc, score) for doc, score in results if score < SIMILARITY_THRESHOLD]

    if not related:
        print(f"  No related stored content found — skipping")
        return None

    print(f"  Found {len(related)} related chunk(s) in stored documents")

    # Build context from the related stored chunks
    stored_context = "\n\n---\n\n".join(
        f"[From: {os.path.basename(doc.metadata.get('source', 'Unknown'))}, "
        f"Page {doc.metadata.get('page', 'N/A')}]\n{doc.page_content}"
        for doc, _ in related
    )

    # Ask Llama 3 to compare and find conflicts
    prompt = f"""You are a SEBI compliance analyst at EY.

A new SEBI circular has been published:
Title: {circular['title']}
Date:  {circular['date']}
Content: {circular_text[:1000]}

This relates to the following content already stored in our documents:
{stored_context}

Your task:
1. Identify any FACTUAL CONFLICTS between the new circular and the stored documents
   (e.g. different dates, percentages, deadlines, regulation numbers, requirements)
2. Identify any GAPS — things the new circular introduces that our stored docs don't mention
3. If there are NO conflicts or gaps, respond with exactly: NO CONFLICTS FOUND

Be specific. Mention exact figures, dates, or regulation references where possible.
Keep your response under 200 words."""

    response = llm.invoke(prompt)
    response = response.strip()

    if "NO CONFLICTS FOUND" in response.upper():
        print(f"  Llama 3: No conflicts found")
        return None

    print(f"  Llama 3: CONFLICT DETECTED")
    return response


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Send email alert
# ─────────────────────────────────────────────────────────────────────────────

def send_email(subject: str, body: str):
    """Send an alert email via Gmail SMTP."""
    if not (SENDER_EMAIL and SENDER_PASSWORD and RECEIVER_EMAIL):
        print("\n  [EMAIL SKIPPED] — set EY_BOT_SENDER_EMAIL / EY_BOT_SENDER_PASSWORD"
              " / EY_BOT_RECEIVER_EMAIL in your .env")
        print(f"  Subject: {subject}")
        print(f"  Body preview: {body[:300]}")
        return

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL
        msg.attach(MIMEText(body, "plain"))

        # Try port 587 (STARTTLS) first — works on more networks than 465 (SSL)
        try:
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.ehlo()
                server.starttls()
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        except Exception:
            # Fallback to port 465 (SSL)
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"  Email sent to {RECEIVER_EMAIL}")
    except Exception as e:
        print(f"  ERROR sending email: {e}")


def build_email_body(circular: dict, conflict_text: str) -> tuple:
    """Build the email subject and body."""
    subject = f"[EY Compliance Alert] New SEBI Circular — Potential Conflict Detected"
    body = f"""EY SEBI Compliance Bot — Automated Alert
Generated: {now()}

NEW CIRCULAR DETECTED:
Title : {circular['title']}
Date  : {circular['date']}
Link  : {circular['link']}

CONFLICT / GAP ANALYSIS:
{conflict_text}

---
This alert was generated automatically by the EY Compliance Bot.
Review the circular and stored documents to confirm.
"""
    return subject, body


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Seen circulars tracker (avoid duplicate alerts)
# ─────────────────────────────────────────────────────────────────────────────

def save_status(circulars_checked: int, alerts_found: int, last_alert_title: str = ""):
    """Write run results to a JSON file so the Streamlit UI can display them."""
    with open(STATUS_FILE, "w") as f:
        json.dump({
            "last_run":          now(),
            "circulars_checked": circulars_checked,
            "alerts_found":      alerts_found,
            "last_alert_title":  last_alert_title,
        }, f)


def load_seen() -> set:
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_seen(seen: set):
    with open(SEEN_FILE, "w") as f:
        json.dump(list(seen), f)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Orchestrates all steps
# ─────────────────────────────────────────────────────────────────────────────

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_check():
    """Full Level 2 pipeline: fetch → compare → email."""
    print(f"\n{'='*60}")
    print(f"Level 2 Check starting: {now()}")
    print(f"{'='*60}")

    if not os.path.exists(CHROMA_DIR):
        print("No ChromaDB found. Run ingest.py first.")
        return

    # Load models
    print("Loading models...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Lazy import Ollama to avoid startup delay when not needed
    from langchain_community.llms import Ollama
    llm = Ollama(model=LLM_MODEL, temperature=0.1)

    # Load seen circulars
    seen = load_seen()

    # Fetch RSS
    circulars = fetch_rss()
    if not circulars:
        print("No new circulars found. All clear.")
        return

    alerts_sent      = 0
    last_alert_title = ""

    for circular in circulars:
        uid = circular["link"] or circular["title"]
        if uid in seen:
            print(f"\nSkipping (already seen): {circular['title'][:60]}")
            continue

        print(f"\nChecking: {circular['title'][:70]}")
        print(f"  Date: {circular['date']}")

        conflict = find_conflicts(circular, embeddings, llm)

        if conflict:
            subject, body = build_email_body(circular, conflict)
            send_email(subject, body)
            alerts_sent     += 1
            last_alert_title = circular["title"]

        seen.add(uid)

    save_seen(seen)
    save_status(
        circulars_checked=len(circulars),
        alerts_found=alerts_sent,
        last_alert_title=last_alert_title,
    )

    print(f"\n{'='*60}")
    print(f"Check complete. {alerts_sent} alert(s) sent.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="EY Compliance Bot — Level 2")
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run on a daily schedule (every 24 hours) instead of once"
    )
    args = parser.parse_args()

    if args.schedule:
        print("Scheduler started. Running daily at 09:00 AM.")
        print("Press Ctrl+C to stop.\n")
        schedule.every().day.at("09:00").do(run_check)
        run_check()   # Also run immediately on start
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        run_check()


if __name__ == "__main__":
    main()
