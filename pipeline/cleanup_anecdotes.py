"""
Blacklist Anecdotes — Post-Processing Cleanup
===============================================
Cleans, enriches, and filters the raw extraction results.

Usage:
    python cleanup_anecdotes.py anecdotes.csv

Output:
    anecdotes_clean.csv   — filtered & enriched, ready for the map
    anecdotes_dropped.csv — removed entries (for review)

Run this AFTER all seasons are extracted. Re-runnable and idempotent.
"""

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False
    print("⚠ geopy not installed — skipping re-geocoding. pip install geopy")


# ============================================================================
# 1. RELEVANCE FILTER — remove non-anecdotes
# ============================================================================

# Patterns that indicate something is NOT a real anecdote
NOT_ANECDOTE_PATTERNS = [
    # Too short / generic statements
    r"^I never sleep",
    r"^I raised my family",
    r"^I rarely enter",
    r"^I keep meaning to attend",
    r"^That's why we're all here",
    r"^I've been moving comfortably",
    r"^this is a Colt",
    # One-liners without a story
    r"^I once had a bad experience in a deep hole",
    r"^I had that done once",
]

# Minimum word count — but short entries WITH a location are kept
MIN_WORDS_NO_LOCATION = 15
MIN_WORDS_WITH_LOCATION = 8


def filter_relevance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove entries that aren't genuine personal anecdotes.
    Short entries are kept if they mention a specific location.
    Returns (kept, dropped) DataFrames.
    """
    drop_mask = pd.Series(False, index=df.index)

    # Pattern-based removal
    for pattern in NOT_ANECDOTE_PATTERNS:
        matches = df["verbatim_text"].str.contains(pattern, regex=True, na=False)
        drop_mask |= matches

    # Too short — but be lenient if there's a location
    word_counts = df["verbatim_text"].str.split().str.len()
    has_location = df["locations"].apply(lambda x: str(x) not in ("[]", "nan", ""))
    too_short = (word_counts < MIN_WORDS_WITH_LOCATION) | (
        (word_counts < MIN_WORDS_NO_LOCATION) & ~has_location
    )
    drop_mask |= too_short

    dropped = df[drop_mask].copy()
    dropped["drop_reason"] = "relevance_filter"
    kept = df[~drop_mask].copy()

    print(f"  Relevance filter: {len(df)} → {len(kept)} (dropped {len(dropped)})")
    return kept, dropped


# ============================================================================
# 2. DEDUPLICATION — overlapping chunks catch the same anecdote
# ============================================================================

def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove duplicate anecdotes (from overlapping chunks).
    Keeps the longer version.
    """
    df = df.sort_values(["episode_id", "chunk_index"]).reset_index(drop=True)

    def word_jaccard(a: str, b: str) -> float:
        wa = set(str(a).lower().split())
        wb = set(str(b).lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def is_substring(short: str, long: str) -> bool:
        """Check if the shorter text is substantially contained in the longer one."""
        short_words = set(short.lower().split())
        long_words = set(long.lower().split())
        if not short_words:
            return False
        overlap = len(short_words & long_words) / len(short_words)
        return overlap > 0.8  # 80% of short text's words appear in long text

    drop_indices = set()

    for i in range(len(df)):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(df)):
            if j in drop_indices:
                continue
            if df.iloc[i]["episode_id"] != df.iloc[j]["episode_id"]:
                continue

            text_i = str(df.iloc[i]["verbatim_text"])
            text_j = str(df.iloc[j]["verbatim_text"])

            # Check Jaccard similarity OR substring containment
            sim = word_jaccard(text_i, text_j)
            shorter_in_longer = is_substring(
                min(text_i, text_j, key=len),
                max(text_i, text_j, key=len),
            )

            if sim > 0.5 or shorter_in_longer:
                # Keep the longer one
                if len(str(df.iloc[i]["verbatim_text"])) >= len(str(df.iloc[j]["verbatim_text"])):
                    drop_indices.add(j)
                else:
                    drop_indices.add(i)

    dropped = df.loc[list(drop_indices)].copy()
    dropped["drop_reason"] = "duplicate"
    kept = df.drop(index=list(drop_indices)).reset_index(drop=True)

    print(f"  Deduplication: {len(df)} → {len(kept)} (dropped {len(dropped)})")
    return kept, dropped


# ============================================================================
# 3. LOCATION ENRICHMENT — infer locations from context/persons
# ============================================================================

# Map of keywords (in verbatim_text, persons, context, locations) → inferred location
LOCATION_INFERENCES = {
    # From persons/text
    "somali pirate":    {"name": "Somalia", "lat": 5.1521, "lon": 46.1996},
    "somali":           {"name": "Somalia", "lat": 5.1521, "lon": 46.1996},
    "oaxaca":           {"name": "Oaxaca, Mexico", "lat": 17.0732, "lon": -96.7266},
    "yawalapiti":       {"name": "Kuluene River, Brazil", "lat": -12.1, "lon": -53.3},

    # Feel free to add more as you discover them in later seasons
    # "chechnya":       {"name": "Chechnya, Russia", "lat": 43.3, "lon": 45.7},
    # "mossad":         {"name": "Israel", "lat": 31.77, "lon": 35.21},
}

# Geocoding corrections — when the geocoder picks the wrong place
GEO_CORRECTIONS = {
    # --- Corrected to right location ---
    "Dingle":              {"name": "Dingle, Ireland", "lat": 52.1409, "lon": -10.2671,
                            "display_name": "Dingle, County Kerry, Ireland"},
    "barrier reef":        {"name": "Belize Barrier Reef", "lat": 17.4, "lon": -87.8,
                            "display_name": "Belize Barrier Reef, Belize"},
    "Amazon":              {"name": "Kuluene River, Brazil", "lat": -12.1, "lon": -53.3,
                            "display_name": "Kuluene River, Xingu, Brazil"},
    "Kuluene River":       {"name": "Kuluene River, Brazil", "lat": -12.1, "lon": -53.3,
                            "display_name": "Kuluene River, Xingu, Brazil"},
    "the Andes":           {"name": "The Andes, Colombia", "lat": -13.5, "lon": -72.0,
                            "display_name": "The Andes, South America"},
    "Zambezi Valley":      {"name": "Zambezi Valley, Zambia", "lat": -15.5, "lon": 28.3,
                            "display_name": "Zambezi Valley, Zambia"},
    "Everest":             {"name": "Mount Everest", "lat": 27.9881, "lon": 86.925,
                            "display_name": "Mount Everest, Nepal/Tibet"},
    "SoHo":                {"name": "SoHo, New York", "lat": 40.7233, "lon": -74.0030,
                            "display_name": "SoHo, Manhattan, New York"},
    "Patagonia":           {"name": "Patagonia, Argentina", "lat": -46.0, "lon": -69.0,
                            "display_name": "Patagonia, Argentina"},
    "Cartagena":           {"name": "Cartagena, Colombia", "lat": 10.3910, "lon": -75.5364,
                            "display_name": "Cartagena, Colombia"},
    "west Africa":         {"name": "West Africa (Gabon)", "lat": -0.8, "lon": 11.6,
                            "display_name": "Gabon, West Africa"},
    "Marseilles":          {"name": "Marseille, France", "lat": 43.2965, "lon": 5.3698,
                            "display_name": "Marseille, France"},
    "West Village":        {"name": "West Village, New York", "lat": 40.7336, "lon": -74.0027,
                            "display_name": "West Village, Manhattan, New York"},
    "Dupont":              {"name": "Dupont Circle, Washington D.C.", "lat": 38.9096, "lon": -77.0434,
                            "display_name": "Dupont Circle, Washington D.C."},
    "Belmont":             {"name": "Belmont Stakes, New York", "lat": 40.7210, "lon": -73.7186,
                            "display_name": "Belmont Park, Elmont, New York"},
    "Augustine":           {"name": "Augustine, Manhattan", "lat": 40.7128, "lon": -74.0060,
                            "display_name": "Augustine Restaurant, Manhattan, New York"},
    "Bard":                {"name": "Bard College, New York", "lat": 42.0230, "lon": -73.9100,
                            "display_name": "Bard College, Annandale-on-Hudson, New York"},
    "La Bernadin":         {"name": "Le Bernardin, New York", "lat": 40.7616, "lon": -73.9817,
                            "display_name": "Le Bernardin, Manhattan, New York"},
    "Soviet union":        {"name": "Moscow, Soviet Union", "lat": 55.7558, "lon": 37.6173,
                            "display_name": "Moscow, Russia (former Soviet Union)"},
    "Seventh Ward":        {"name": "Seventh Ward, New Orleans", "lat": 29.9694, "lon": -90.0488,
                            "display_name": "Seventh Ward, New Orleans, Louisiana"},
    "Altiplano":           {"name": "Altiplano, Bolivia", "lat": -17.0, "lon": -66.0,
                            "display_name": "Altiplano, Bolivia"},
    "Constitution Avenue": {"name": "Constitution Ave, Washington D.C.", "lat": 38.8913, "lon": -77.0226,
                            "display_name": "Constitution Avenue, Washington D.C."},
    "DMV":                 {"name": "DMV, Washington D.C. area", "lat": 38.9072, "lon": -77.0369,
                            "display_name": "D.C./Maryland/Virginia metro area"},
    "Royal Academy":       {"name": "Royal Academy, London", "lat": 51.5094, "lon": -0.1392,
                            "display_name": "Royal Academy of Arts, London"},
    "Mardale":             {"name": "Mardale, England", "lat": 54.5, "lon": -2.8,
                            "display_name": "Mardale, Lake District, England"},
    "place St. Pierre":    {"name": "Place St. Pierre, Paris", "lat": 48.8865, "lon": 2.3431,
                            "display_name": "Place Saint-Pierre, Montmartre, Paris"},
    "Safeway":             None,  # Not a real location for the map
    "Eastern seaboard":    None,  # Too vague

    # --- Nonsense geocodes, remove entirely ---
    "that nightclub":      None,
    "that diner":          None,
    "the park":            None,
    "the woods":           None,
    "beach":               None,
    "coastal beaches":     None,
    "desert":              None,
    "mesquite":            None,
    "middle school":       None,
    "our village":         None,
    "Crimean War":         None,
    "Immigration Services": None,
    "America":             None,  # Too broad
}


def safe_parse_list(val) -> list:
    """Parse a string representation of a list."""
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s in ("[]", "nan", "", "None"):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def safe_parse_geo(val) -> list:
    """Parse the geo column (list of dicts)."""
    if isinstance(val, list):
        return val
    s = str(val).strip()
    if s in ("[]", "nan", "", "None"):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return []


def enrich_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer locations from context, persons, and text when locations list is empty.
    Also fix known geocoding errors.
    """
    df = df.copy()

    for idx, row in df.iterrows():
        locations = safe_parse_list(row["locations"])
        geo = safe_parse_geo(row["geo"])
        persons = safe_parse_list(row.get("persons", []))

        text_lower = str(row["verbatim_text"]).lower()
        persons_lower = " ".join(str(p) for p in persons).lower()
        context_lower = str(row.get("context", "")).lower()
        all_text = f"{text_lower} {persons_lower} {context_lower}"

        # --- Infer missing locations from text/persons/context ---
        if not geo:  # Only if no geo data yet
            for keyword, inferred in LOCATION_INFERENCES.items():
                if keyword in all_text:
                    # Add to geo
                    geo.append({
                        "name": inferred["name"],
                        "lat": inferred["lat"],
                        "lon": inferred["lon"],
                        "display_name": inferred["name"],
                        "inferred": True,
                    })
                    if inferred["name"] not in locations:
                        locations.append(inferred["name"])
                    break  # One inference per anecdote is enough

        # --- Fix known geocoding errors ---
        corrected_geo = []
        for g in geo:
            name = g.get("name", "")
            if name in GEO_CORRECTIONS:
                correction = GEO_CORRECTIONS[name]
                if correction is None:
                    continue  # Drop this geo entry
                corrected_geo.append(correction)
            else:
                corrected_geo.append(g)
        geo = corrected_geo

        df.at[idx, "locations"] = str(locations)
        df.at[idx, "geo"] = str(geo)

    enriched = df["geo"].apply(lambda x: str(x) not in ("[]", "nan", "")).sum()
    print(f"  After enrichment: {enriched}/{len(df)} anecdotes have geo data")
    return df


# ============================================================================
# 4. PREFER SPECIFIC LOCATIONS
# ============================================================================

# When both a broad and specific location are present, keep only the specific one
SPECIFICITY_RULES = [
    # (broad, specific) — if both present, drop broad from geo
    ("Amazon", "Kuluene River"),
    ("Andaman Sea", "Ko Ri"),
    ("Belize", "barrier reef"),  # after correction → Belize Barrier Reef
]


def prefer_specific_location(df: pd.DataFrame) -> pd.DataFrame:
    """When a broad and specific location co-occur, keep only the specific one for the map pin."""
    df = df.copy()

    for idx, row in df.iterrows():
        geo = safe_parse_geo(row["geo"])
        if len(geo) <= 1:
            continue

        geo_names = {g.get("name", "") for g in geo}

        for broad, specific in SPECIFICITY_RULES:
            # Check if both are present (also check partial matches)
            has_broad = any(broad.lower() in n.lower() for n in geo_names)
            has_specific = any(specific.lower() in n.lower() for n in geo_names)

            if has_broad and has_specific:
                geo = [g for g in geo if broad.lower() not in g.get("name", "").lower()]

        df.at[idx, "geo"] = str(geo)

    return df


# ============================================================================
# 5. FINAL ENRICHMENT — re-geocode missing locations
# ============================================================================

def regeocode_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Try to geocode locations that have location names but no geo data."""
    if not HAS_GEOPY:
        return df

    import time
    geolocator = Nominatim(user_agent="blacklist_cleanup_v1")
    cache = {}

    df = df.copy()
    fixed = 0

    for idx, row in df.iterrows():
        locations = safe_parse_list(row["locations"])
        geo = safe_parse_geo(row["geo"])

        if geo or not locations:
            continue

        # Try to geocode the first location
        for loc_name in locations:
            clean_name = loc_name.replace("-", " ").strip()

            if clean_name in cache:
                if cache[clean_name]:
                    geo.append(cache[clean_name])
                continue

            try:
                result = geolocator.geocode(clean_name, timeout=10)
                if result:
                    g = {
                        "name": loc_name,
                        "lat": result.latitude,
                        "lon": result.longitude,
                        "display_name": result.address,
                        "re_geocoded": True,
                    }
                    cache[clean_name] = g
                    geo.append(g)
                    fixed += 1
                else:
                    cache[clean_name] = None
                time.sleep(1)
            except Exception as e:
                cache[clean_name] = None
                print(f"  ⚠ Geocode error for '{clean_name}': {e}")

        df.at[idx, "geo"] = str(geo)

    print(f"  Re-geocoded {fixed} additional locations")
    return df


# ============================================================================
# 6. ADD MAP-READY COLUMNS
# ============================================================================

def add_map_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add flattened columns for easy map rendering:
    - map_lat, map_lon: coordinates of the primary (most specific) location
    - map_label: short location name for the pin
    - has_location: boolean flag
    """
    df = df.copy()
    map_lats = []
    map_lons = []
    map_labels = []

    for _, row in df.iterrows():
        geo = safe_parse_geo(row["geo"])

        if geo:
            # Pick the first geo entry (most specific after our filtering)
            primary = geo[0]
            map_lats.append(primary.get("lat"))
            map_lons.append(primary.get("lon"))
            map_labels.append(primary.get("name", "Unknown"))
        else:
            map_lats.append(None)
            map_lons.append(None)
            map_labels.append(None)

    df["map_lat"] = map_lats
    df["map_lon"] = map_lons
    df["map_label"] = map_labels
    df["has_location"] = df["map_lat"].notna()

    with_loc = df["has_location"].sum()
    print(f"  Map-ready: {with_loc}/{len(df)} anecdotes have coordinates")
    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python cleanup_anecdotes.py anecdotes.csv")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} anecdotes from {input_path}\n")

    all_dropped = []

    # Step 1: Relevance filter
    print("1️⃣  Filtering irrelevant entries...")
    df, dropped = filter_relevance(df)
    all_dropped.append(dropped)

    # Step 2: Deduplicate
    print("\n2️⃣  Removing duplicates...")
    df, dropped = deduplicate(df)
    all_dropped.append(dropped)

    # Step 3: Enrich locations
    print("\n3️⃣  Enriching locations...")
    df = enrich_locations(df)

    # Step 4: Prefer specific locations
    print("\n4️⃣  Selecting most specific locations...")
    df = prefer_specific_location(df)

    # Step 5: Re-geocode missing
    print("\n5️⃣  Re-geocoding missing locations...")
    df = regeocode_missing(df)

    # Step 6: Add map columns
    print("\n6️⃣  Adding map-ready columns...")
    df = add_map_columns(df)

    # Save results
    clean_path = input_path.parent / "anecdotes_clean.csv"
    df.to_csv(clean_path, index=False)
    print(f"\n💾 Clean data: {clean_path} ({len(df)} anecdotes)")

    dropped_df = pd.concat(all_dropped, ignore_index=True)
    if not dropped_df.empty:
        dropped_path = input_path.parent / "anecdotes_dropped.csv"
        dropped_df.to_csv(dropped_path, index=False)
        print(f"🗑  Dropped: {dropped_path} ({len(dropped_df)} entries)")

    # Summary
    print(f"\n📈 Summary:")
    print(f"   Input:          {len(pd.read_csv(input_path))}")
    print(f"   After cleanup:  {len(df)}")
    print(f"   With location:  {df['has_location'].sum()}")
    print(f"   Without:        {(~df['has_location']).sum()}")
    print(f"\n   Locations on map:")
    for _, row in df[df["has_location"]].iterrows():
        print(f"     📍 {row['map_label']:30s} ({row['episode_id']})")


if __name__ == "__main__":
    main()
