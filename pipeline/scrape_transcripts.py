"""
Blacklist Transcript Scraper
=============================
Scrapes all episode transcripts from subslikescript.com.
Saves each episode as a JSON file with metadata.

Usage:
    python scrape_transcripts.py

Output:
    transcripts/S01E01.json, S01E02.json, ...
    transcripts/_index.json  (overview of all episodes)

Note: Be respectful — the script includes delays between requests.
"""

import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://subslikescript.com"
SERIES_URL = f"{BASE_URL}/series/The_Blacklist-2741602"
OUTPUT_DIR = Path("transcripts")
DELAY_SECONDS = 2  # Be polite to the server

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


# ---------------------------------------------------------------------------
# Step 1: Get all episode URLs from the series page
# ---------------------------------------------------------------------------

def get_episode_urls() -> list[dict]:
    """
    Fetch the series overview page and extract all episode links.
    Returns list of dicts with: season, episode_num, episode_id, title, url
    """
    print(f"Fetching series page: {SERIES_URL}")
    resp = requests.get(SERIES_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    episodes = []
    current_season = 0

    # The page has <h4> tags like "Season 1", "Season 2", etc.
    # followed by <ul> lists with episode links
    for heading in soup.find_all("h4"):
        season_match = re.search(r"Season\s+(\d+)", heading.get_text())
        if season_match:
            current_season = int(season_match.group(1))

        # Find the next <ul> sibling
        ul = heading.find_next_sibling("ul")
        if not ul:
            continue

        for li in ul.find_all("li"):
            a_tag = li.find("a")
            if not a_tag:
                continue

            href = a_tag.get("href", "")
            title = a_tag.get_text(strip=True)

            # Extract episode number from the list item number or title
            # The <li> items are numbered 1, 2, 3... per season
            ep_match = re.search(r"episode-(\d+)", href)
            if ep_match:
                ep_num = int(ep_match.group(1))
            else:
                # Fallback: count position
                ep_num = len([e for e in episodes if e["season"] == current_season]) + 1

            episode_id = f"S{current_season:02d}E{ep_num:02d}"
            full_url = urljoin(BASE_URL, href)

            episodes.append({
                "season": current_season,
                "episode_num": ep_num,
                "episode_id": episode_id,
                "title": title,
                "url": full_url,
            })

    print(f"Found {len(episodes)} episodes across {current_season} seasons")
    return episodes


# ---------------------------------------------------------------------------
# Step 2: Fetch individual episode transcript
# ---------------------------------------------------------------------------

def fetch_transcript(url: str) -> str | None:
    """
    Fetch a single episode page and extract the transcript text.
    Returns cleaned transcript text or None on failure.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ✗ Request failed: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # The transcript text is inside a <div class="full-script">
    script_div = soup.find("div", class_="full-script")

    if not script_div:
        # Fallback: try the main content area
        script_div = soup.find("article") or soup.find("div", class_="main-article")

    if not script_div:
        print("  ✗ Could not find transcript div")
        return None

    # Get text, preserving line breaks
    text = script_div.get_text(separator="\n", strip=False)

    # Clean up
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 3: Main scraping loop
# ---------------------------------------------------------------------------

def scrape_all():
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Get episode list
    episodes = get_episode_urls()

    if not episodes:
        print("No episodes found! Check if the site structure changed.")
        return

    # Save index
    index_path = OUTPUT_DIR / "_index.json"
    with open(index_path, "w") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    print(f"Saved episode index to {index_path}")

    # Step 2: Fetch each transcript
    success = 0
    skipped = 0

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        out_path = OUTPUT_DIR / f"{ep_id}.json"

        # Skip if already downloaded (for resume capability)
        if out_path.exists():
            print(f"  [{i+1}/{len(episodes)}] {ep_id} — already exists, skipping")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(episodes)}] {ep_id} - {ep['title']}...", end=" ")

        transcript = fetch_transcript(ep["url"])

        if transcript:
            # Save as JSON with metadata
            data = {
                "episode_id": ep_id,
                "season": ep["season"],
                "episode_num": ep["episode_num"],
                "title": ep["title"],
                "source_url": ep["url"],
                "transcript": transcript,
                "char_count": len(transcript),
                "word_count": len(transcript.split()),
            }
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✓ ({data['word_count']} words)")
            success += 1
        else:
            print("✗ failed")

        # Be polite
        time.sleep(DELAY_SECONDS)

    print(f"\n{'=' * 50}")
    print(f"Done! {success} downloaded, {skipped} skipped (already existed)")
    print(f"Transcripts saved in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    scrape_all()
