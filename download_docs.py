"""
download_docs.py
----------------
Expands the knowledge base with PUBLIC regulatory PDFs so the RAG pipeline can
be evaluated on a realistic corpus (not just 5 hand-picked documents).

We use RBI as the bulk source: RBI publishes its Master Directions and large
circulars as open PDFs on rbidocs.rbi.org.in and — unlike SEBI's main site — it
does not block automated access. Same compliance domain; Level 2 already
monitors RBI.

Usage:
    python download_docs.py --list          # probe sources, list candidate PDFs
    python download_docs.py --download 25    # download up to 25 new PDFs to ./docs
"""
import argparse
import os
import re
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DOCS_DIR = "./docs"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}

# RBI index pages that list documents with links to PDFs.
RBI_SOURCES = [
    "https://www.rbi.org.in/Scripts/BS_ViewMasterDirections.aspx",
    "https://www.rbi.org.in/Scripts/BS_ViewMasterCirculars.aspx",
]


def find_pdf_links(session, index_url):
    """Return list of (pdf_url, title) found on an RBI index page."""
    try:
        r = session.get(index_url, headers=HEADERS, timeout=30, verify=True)
        r.raise_for_status()
    except Exception as e:
        print(f"  ! could not fetch {index_url}: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    out = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if ".pdf" in href.lower() or "/PDFs/" in href:
            url = urljoin(index_url, href)
            title = (a.get_text(strip=True) or "").strip()
            out.append((url, title))
    # de-dup by url
    seen, uniq = set(), []
    for url, title in out:
        if url not in seen:
            seen.add(url)
            uniq.append((url, title))
    return uniq


def safe_name(url, title):
    base = re.sub(r"[^A-Za-z0-9]+", "_", (title or "")).strip("_")[:60]
    tail = os.path.basename(url.split("?")[0])
    if not base:
        base = "RBI_doc"
    return f"RBI_{base}_{tail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--download", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    session = requests.Session()

    all_links = []
    for src in RBI_SOURCES:
        print(f"Scanning: {src}")
        links = find_pdf_links(session, src)
        print(f"  found {len(links)} PDF link(s)")
        all_links.extend(links)

    # de-dup across sources
    seen, links = set(), []
    for url, title in all_links:
        if url not in seen:
            seen.add(url)
            links.append((url, title))
    print(f"\nTotal unique PDF candidates: {len(links)}\n")

    for url, title in links[:40]:
        print(f"  - {title[:70]:<70} {url}")

    if args.download:
        print(f"\nDownloading up to {args.download} PDFs into {DOCS_DIR} ...")
        got = 0
        for url, title in links:
            if got >= args.download:
                break
            name = safe_name(url, title)
            dest = os.path.join(DOCS_DIR, name)
            if os.path.exists(dest):
                continue
            try:
                r = session.get(url, headers=HEADERS, timeout=60)
                r.raise_for_status()
                if not r.content[:4] == b"%PDF":
                    print(f"  skip (not a PDF): {url}")
                    continue
                with open(dest, "wb") as f:
                    f.write(r.content)
                kb = len(r.content) // 1024
                print(f"  saved {name}  ({kb} KB)")
                got += 1
            except Exception as e:
                print(f"  ! failed {url}: {e}")
        print(f"\nDone. Downloaded {got} new PDF(s).")


if __name__ == "__main__":
    main()
