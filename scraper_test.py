"""
scraper_test.py
---------------
Tests multiple approaches to get SEBI/RBI circular data.
Tries SEBI RSS feed, RBI website, and a session-based approach.

Run: python scraper_test.py
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.google.com/",
}

def try_url(label, url, is_xml=False):
    print(f"\nTrying [{label}]: {url}")
    try:
        session = requests.Session()
        session.headers.update(HEADERS)
        resp = session.get(url, timeout=15)
        resp.raise_for_status()

        if is_xml:
            soup = BeautifulSoup(resp.text, "xml")
            items = soup.find_all("item")[:5]
            if items:
                print(f"  SUCCESS — Found {len(soup.find_all('item'))} items. First 5:")
                for item in items:
                    title = item.find("title")
                    date  = item.find("pubDate") or item.find("dc:date")
                    print(f"    - {title.text.strip() if title else 'N/A'} | {date.text.strip() if date else 'N/A'}")
                return resp.text, "xml"
            else:
                print("  XML parsed but no <item> tags found")
                return None, None
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            text = "\n".join(
                line.strip()
                for line in soup.get_text(separator="\n").splitlines()
                if line.strip()
            )
            print(f"  SUCCESS — {len(text)} characters scraped. Preview:")
            print(f"  {text[:400]}")
            return text, "html"

    except Exception as e:
        print(f"  FAILED — {e}")
        return None, None


if __name__ == "__main__":
    print("=" * 60)
    print("Testing available data sources for Level 2")
    print("=" * 60)

    # 1. SEBI RSS feed (XML — bypasses JS rendering issues)
    text, kind = try_url(
        "SEBI RSS Feed",
        "https://www.sebi.gov.in/sebirss.xml",
        is_xml=True
    )
    if text:
        with open("scraped_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n✅ SEBI RSS works! Saved to scraped_output.txt")
        exit()

    # 2. RBI circulars page (often more accessible than SEBI)
    text, kind = try_url(
        "RBI Circulars Page",
        "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx"
    )
    if text and len(text) > 500:
        with open("scraped_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n✅ RBI page works! Saved to scraped_output.txt")
        exit()

    # 3. RBI RSS feed
    text, kind = try_url(
        "RBI RSS Feed",
        "https://www.rbi.org.in/Scripts/RSS.aspx",
        is_xml=True
    )
    if text:
        with open("scraped_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n✅ RBI RSS works! Saved to scraped_output.txt")
        exit()

    # 4. SEBI circulars page with session (last attempt)
    text, kind = try_url(
        "SEBI Circulars (session)",
        "https://www.sebi.gov.in/legal/circulars/"
    )
    if text and len(text) > 500:
        with open("scraped_output.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("\n✅ SEBI page works! Saved to scraped_output.txt")
        exit()

    print("\n" + "=" * 60)
    print("All sources blocked. See options below.")
    print("=" * 60)
