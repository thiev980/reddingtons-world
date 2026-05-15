"""
Blacklist Anecdotes Extraction Pipeline (v2)
=============================================
Extracts Raymond Reddington's anecdotes from episode transcripts.
Works with plaintext transcripts (from scrape_transcripts.py) or SRT files.

Usage:
    # With scraped transcripts (recommended):
    python extract_anecdotes.py --transcript-dir ./transcripts/

    # With SRT files (legacy):
    python extract_anecdotes.py --srt-dir ./srt_files/

Requirements:
    pip install anthropic geopy pandas pysrt  (pysrt only needed for SRT mode)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import pandas as pd

# ---------------------------------------------------------------------------
# 1. TRANSCRIPT LOADING
# ---------------------------------------------------------------------------

@dataclass
class EpisodeTranscript:
    """A full episode transcript with metadata."""
    episode_id: str     # e.g. "S01E03"
    season: int
    episode_num: int
    title: str
    transcript: str     # full dialog text
    source: str         # "subslikescript" or "srt"


def load_transcripts_from_json(transcript_dir: Path) -> list[EpisodeTranscript]:
    """Load transcripts scraped by scrape_transcripts.py."""
    transcripts = []
    json_files = sorted(transcript_dir.glob("S*.json"))
    print(f"Found {len(json_files)} transcript files in {transcript_dir}")

    for filepath in json_files:
        try:
            with open(filepath) as f:
                data = json.load(f)
            transcripts.append(EpisodeTranscript(
                episode_id=data["episode_id"],
                season=data["season"],
                episode_num=data["episode_num"],
                title=data.get("title", ""),
                transcript=data["transcript"],
                source="subslikescript",
            ))
            print(f"  ✓ {data['episode_id']} - {data.get('title', '')} ({data.get('word_count', '?')} words)")
        except Exception as e:
            print(f"  ✗ {filepath.name}: {e}")

    return transcripts


def load_transcripts_from_srt(srt_dir: Path) -> list[EpisodeTranscript]:
    """Load transcripts from SRT subtitle files (legacy support)."""
    import pysrt

    transcripts = []
    srt_files = sorted(srt_dir.glob("*.srt"))
    print(f"Found {len(srt_files)} SRT files in {srt_dir}")

    for filepath in srt_files:
        try:
            match = re.search(r'[Ss](\d{1,2})[Ee](\d{1,2})', filepath.name)
            if not match:
                print(f"  ✗ {filepath.name}: can't parse season/episode")
                continue

            season = int(match.group(1))
            ep_num = int(match.group(2))
            episode_id = f"S{season:02d}E{ep_num:02d}"

            subs = pysrt.open(str(filepath), encoding='utf-8')
            text_lines = []
            for sub in subs:
                clean = re.sub(r'<[^>]+>', '', sub.text)
                clean = re.sub(r'\{[^}]+\}', '', clean)
                clean = re.sub(r'♪.*?♪', '', clean)
                clean = clean.strip()
                if clean:
                    text_lines.append(clean)

            transcript = "\n".join(text_lines)
            transcripts.append(EpisodeTranscript(
                episode_id=episode_id,
                season=season,
                episode_num=ep_num,
                title=filepath.stem,
                transcript=transcript,
                source="srt",
            ))
            print(f"  ✓ {episode_id} ({len(text_lines)} lines)")
        except Exception as e:
            print(f"  ✗ {filepath.name}: {e}")

    return transcripts


# ---------------------------------------------------------------------------
# 2. CHUNKING — split long transcripts for LLM processing
# ---------------------------------------------------------------------------

@dataclass
class DialogChunk:
    """A chunk of dialog ready for LLM extraction."""
    episode_id: str
    season: int
    episode_num: int
    title: str
    chunk_index: int
    total_chunks: int
    text: str
    word_count: int


def chunk_transcript(
    ep: EpisodeTranscript,
    max_words: int = 1500,
    overlap_words: int = 200,
) -> list[DialogChunk]:
    """
    Split a transcript into overlapping word-based chunks.

    ~1500 words ≈ ~2 minutes of dialog.
    Overlap ensures anecdotes at chunk boundaries aren't lost.
    """
    words = ep.transcript.split()
    total_words = len(words)

    if total_words == 0:
        return []

    step = max_words - overlap_words
    chunks = []
    chunk_idx = 0
    start = 0

    while start < total_words:
        end = min(start + max_words, total_words)
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append(DialogChunk(
            episode_id=ep.episode_id,
            season=ep.season,
            episode_num=ep.episode_num,
            title=ep.title,
            chunk_index=chunk_idx,
            total_chunks=0,  # filled in below
            text=chunk_text,
            word_count=len(chunk_words),
        ))
        chunk_idx += 1
        start += step

    for c in chunks:
        c.total_chunks = len(chunks)

    return chunks


def chunk_all_transcripts(
    transcripts: list[EpisodeTranscript],
    max_words: int = 1500,
    overlap_words: int = 200,
) -> list[DialogChunk]:
    """Chunk all transcripts."""
    all_chunks = []
    for ep in transcripts:
        chunks = chunk_transcript(ep, max_words, overlap_words)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks from {len(transcripts)} episodes")
    return all_chunks


# ---------------------------------------------------------------------------
# 3. LLM EXTRACTION via Claude API
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are analyzing dialog from the TV show "The Blacklist" (Season {season}, \
Episode {episode_num}: "{title}").

Your task: identify **personal anecdotes** told by Raymond "Red" Reddington.

An "anecdote" is a short story or reminiscence from Red's past that he shares \
in conversation. Typical characteristics:
- References a specific past event or personal experience
- Often mentions a place (city, country, or specific location)
- Often mentions a person he knew, met, or dealt with
- Usually in first person ("I once...", "Years ago...", "There was a time...")
- Can start subtly — Red flowing into a memory mid-conversation
- Ranges from a single sentence to a short paragraph

NOT anecdotes:
- Plot exposition or instructions to other characters
- General philosophical musings without a specific story
- Descriptions of current events in the episode
- References to well-known historical events without personal involvement

For each anecdote found, extract:
1. **verbatim_text**: The exact quote — Red's full anecdote as spoken. \
   Include only Red's words, not other characters' lines in between.
2. **locations**: List of specific places mentioned (cities, countries, \
   landmarks, venues). Use the most specific name. Empty list if none.
3. **persons**: Names or descriptions of people mentioned. Empty list if none.
4. **time_hint**: Any temporal reference ("the 80s", "before the Wall fell", \
   "twenty years ago"). Null if none.
5. **context**: One sentence — what situation is Red in when he tells this? \
   (e.g., "Threatening a suspect", "Having dinner with Liz", "Negotiating a deal")
6. **mood**: One of: nostalgic, menacing, humorous, melancholic, cautionary, \
   admiring, wistful, sardonic

Respond ONLY with a JSON object:
{{
  "anecdotes": [
    {{
      "verbatim_text": "...",
      "locations": ["Damascus", "Syria"],
      "persons": ["a carpet merchant named Farhad"],
      "time_hint": "the early 90s",
      "context": "Advising Liz on how to handle a suspect",
      "mood": "nostalgic"
    }}
  ]
}}

If there are NO anecdotes in this chunk, respond: {{"anecdotes": []}}

Be selective — only genuine personal anecdotes where Red is recounting a \
past experience. Quality over quantity.

--- DIALOG (chunk {chunk_index}/{total_chunks}) ---
{dialog_text}
--- END ---
"""


def extract_anecdotes_from_chunk(
    chunk: DialogChunk,
    client,  # anthropic.Anthropic
    model: str = "claude-sonnet-4-20250514",
) -> list[dict]:
    """Send a single chunk to Claude API and parse extracted anecdotes."""
    prompt = EXTRACTION_PROMPT.format(
        season=chunk.season,
        episode_num=chunk.episode_num,
        title=chunk.title,
        chunk_index=chunk.chunk_index + 1,
        total_chunks=chunk.total_chunks,
        dialog_text=chunk.text,
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        data = json.loads(raw)
        anecdotes = data.get("anecdotes", [])

        for a in anecdotes:
            a["episode_id"] = chunk.episode_id
            a["season"] = chunk.season
            a["episode_num"] = chunk.episode_num
            a["episode_title"] = chunk.title
            a["chunk_index"] = chunk.chunk_index

        return anecdotes

    except json.JSONDecodeError as e:
        print(f"  ⚠ JSON error {chunk.episode_id} chunk {chunk.chunk_index}: {e}")
        return []
    except Exception as e:
        print(f"  ⚠ API error {chunk.episode_id} chunk {chunk.chunk_index}: {e}")
        return []


def run_extraction(
    chunks: list[DialogChunk],
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    rate_limit_delay: float = 0.5,
    output_path: Path = Path("anecdotes_raw.json"),
    resume: bool = True,
) -> list[dict]:
    """Run extraction with progress tracking, incremental saves, and resume."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    all_anecdotes = []
    processed_keys = set()

    if resume and output_path.exists():
        with open(output_path) as f:
            all_anecdotes = json.load(f)
        processed_keys = {
            f"{a['episode_id']}_{a.get('chunk_index', -1)}"
            for a in all_anecdotes
        }
        print(f"  Resuming: {len(all_anecdotes)} anecdotes from previous run")

    total = len(chunks)
    new_count = 0

    for i, chunk in enumerate(chunks):
        key = f"{chunk.episode_id}_{chunk.chunk_index}"
        if key in processed_keys:
            continue

        print(f"  [{i+1}/{total}] {chunk.episode_id} chunk {chunk.chunk_index+1}/{chunk.total_chunks}...", end=" ")
        anecdotes = extract_anecdotes_from_chunk(chunk, client, model)
        all_anecdotes.extend(anecdotes)
        new_count += len(anecdotes)
        print(f"found {len(anecdotes)}")

        if (i + 1) % 25 == 0:
            with open(output_path, 'w') as f:
                json.dump(all_anecdotes, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved ({len(all_anecdotes)} total)")

        time.sleep(rate_limit_delay)

    with open(output_path, 'w') as f:
        json.dump(all_anecdotes, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done: {new_count} new, {len(all_anecdotes)} total")
    return all_anecdotes


# ---------------------------------------------------------------------------
# 4. DEDUPLICATION
# ---------------------------------------------------------------------------

def deduplicate_anecdotes(
    anecdotes: list[dict],
    similarity_threshold: float = 0.6,
) -> list[dict]:
    """Remove duplicates from overlapping chunks via Jaccard similarity."""
    if not anecdotes:
        return []

    def jaccard(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    sorted_anecdotes = sorted(
        anecdotes,
        key=lambda a: (a.get("episode_id", ""), a.get("chunk_index", 0))
    )

    unique = []
    for anecdote in sorted_anecdotes:
        is_dup = False
        for existing in unique:
            if existing.get("episode_id") == anecdote.get("episode_id"):
                sim = jaccard(
                    existing.get("verbatim_text", ""),
                    anecdote.get("verbatim_text", ""),
                )
                if sim > similarity_threshold:
                    if len(anecdote.get("verbatim_text", "")) > len(existing.get("verbatim_text", "")):
                        unique.remove(existing)
                        unique.append(anecdote)
                    is_dup = True
                    break
        if not is_dup:
            unique.append(anecdote)

    print(f"Deduplication: {len(anecdotes)} → {len(unique)}")
    return unique


# ---------------------------------------------------------------------------
# 5. GEOCODING
# ---------------------------------------------------------------------------

def geocode_anecdotes(anecdotes: list[dict], delay: float = 1.0) -> list[dict]:
    """Add lat/lon using geopy/Nominatim."""
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut
    except ImportError:
        print("⚠ Install geopy: pip install geopy")
        return anecdotes

    geolocator = Nominatim(user_agent="blacklist_anecdotes_v2")
    cache: dict[str, dict | None] = {}

    for anecdote in anecdotes:
        geo_results = []
        for loc_name in anecdote.get("locations", []):
            if loc_name in cache:
                if cache[loc_name]:
                    geo_results.append(cache[loc_name])
                continue

            try:
                location = geolocator.geocode(loc_name, timeout=10)
                if location:
                    geo = {
                        "name": loc_name,
                        "lat": location.latitude,
                        "lon": location.longitude,
                        "display_name": location.address,
                    }
                    cache[loc_name] = geo
                    geo_results.append(geo)
                else:
                    cache[loc_name] = None
                    print(f"  ⚠ Not found: {loc_name}")

                time.sleep(delay)
            except (GeocoderTimedOut, Exception) as e:
                cache[loc_name] = None
                print(f"  ⚠ Error: {loc_name}: {e}")

        anecdote["geo"] = geo_results

    geocoded = sum(1 for a in anecdotes if a.get("geo"))
    print(f"Geocoded {geocoded}/{len(anecdotes)} anecdotes")
    return anecdotes


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract Reddington anecdotes from Blacklist transcripts"
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript-dir", type=Path,
                        help="Directory with JSON transcripts from scraper")
    source.add_argument("--srt-dir", type=Path,
                        help="Directory with SRT files (legacy)")

    parser.add_argument("--output", type=Path, default=Path("anecdotes.json"))
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--max-words", type=int, default=1500)
    parser.add_argument("--overlap-words", type=int, default=200)
    parser.add_argument("--skip-extraction", action="store_true")
    parser.add_argument("--skip-geocoding", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seasons", type=int, nargs="+",
                        help="Only these seasons (e.g. --seasons 1 2)")
    args = parser.parse_args()

    # Load
    print("\n📖 Loading transcripts...")
    if args.transcript_dir:
        transcripts = load_transcripts_from_json(args.transcript_dir)
    else:
        transcripts = load_transcripts_from_srt(args.srt_dir)

    if not transcripts:
        print("❌ No transcripts found!")
        return

    if args.seasons:
        transcripts = [t for t in transcripts if t.season in args.seasons]
        print(f"Filtered to seasons {args.seasons}: {len(transcripts)} episodes")

    # Chunk
    print(f"\n🔪 Chunking ({args.max_words} words, {args.overlap_words} overlap)...")
    chunks = chunk_all_transcripts(transcripts, args.max_words, args.overlap_words)

    if not args.skip_extraction:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("\n❌ export ANTHROPIC_API_KEY='your-key-here'")
            return

        raw_path = args.output.parent / "anecdotes_raw.json"
        print(f"\n🤖 Extracting via {args.model} ({len(chunks)} chunks)...")
        anecdotes = run_extraction(
            chunks, api_key, args.model,
            output_path=raw_path, resume=not args.no_resume,
        )
    else:
        raw_path = args.output.parent / "anecdotes_raw.json"
        print(f"\n⏭ Loading from {raw_path}")
        with open(raw_path) as f:
            anecdotes = json.load(f)

    # Deduplicate
    print("\n🧹 Deduplicating...")
    anecdotes = deduplicate_anecdotes(anecdotes)

    # Geocode
    if not args.skip_geocoding:
        print("\n🌍 Geocoding...")
        anecdotes = geocode_anecdotes(anecdotes)

    # Save
    with open(args.output, 'w') as f:
        json.dump(anecdotes, f, indent=2, ensure_ascii=False)
    print(f"\n💾 {args.output} ({len(anecdotes)} anecdotes)")

    csv_path = args.output.with_suffix('.csv')
    df = pd.json_normalize(anecdotes)
    df.to_csv(csv_path, index=False)
    print(f"📊 {csv_path}")

    # Stats
    print(f"\n📈 Stats:")
    print(f"   Anecdotes: {len(anecdotes)}")
    locations = set()
    for a in anecdotes:
        for loc in a.get("locations", []):
            locations.add(loc)
    print(f"   Unique locations: {len(locations)}")
    moods = {}
    for a in anecdotes:
        m = a.get("mood", "unknown")
        moods[m] = moods.get(m, 0) + 1
    for mood, count in sorted(moods.items(), key=lambda x: -x[1]):
        print(f"   {mood}: {count}")


if __name__ == "__main__":
    main()
