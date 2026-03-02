import os
import re
import time
import difflib
import logging

import lyricsgenius
from mutagen import File as MutagenFile

from app.transcription import Segment, WordTiming

logger = logging.getLogger(__name__)


def get_metadata_from_file(audio_path: str) -> tuple[str | None, str | None]:
    """Extract artist and title from audio file metadata."""
    try:
        audio = MutagenFile(audio_path, easy=True)
        if audio is None:
            return None, None
        artist = audio.get("artist", [None])[0]
        title = audio.get("title", [None])[0]
        return artist, title
    except Exception as e:
        logger.warning(f"Could not read metadata from {audio_path}: {e}")
        return None, None


def _sanitize_query(text: str) -> str:
    """Clean up artist/title strings for better Genius search results.

    Removes '&', 'feat.', 'ft.', 'featuring', parenthetical suffixes like
    '(Remix)', '(Remastered)', etc. that confuse the Genius search and
    normalises whitespace.
    """
    # Remove parenthetical/bracketed suffixes with known keywords
    text = re.sub(
        r"\s*[\(\[]"
        r"(remix|remaster(ed)?|live|bonus\s*track|deluxe|explicit|clean"
        r"|acoustic|version|edit|radio\s*edit|original\s*mix"
        r"|feat\.?[^)\]]*|ft\.?[^)\]]*)"
        r"[\)\]]",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Replace common separators / feature markers with a space
    text = re.sub(r"\s*&\s*", " ", text)
    text = re.sub(r"\s*\b(feat\.?|ft\.?|featuring)\s*", " ", text, flags=re.IGNORECASE)
    # Collapse multiple spaces
    return re.sub(r"\s{2,}", " ", text).strip()


def _primary_artist(artist: str) -> str:
    """Extract the primary (first) artist, dropping features and collabs.

    Genius lists songs under the primary artist only, so including featured
    artists in the search query often returns zero results.
    """
    # Split on common separators: &, feat., ft., featuring, ","
    primary = re.split(r"\s*(?:&|,|\b(?:feat\.?|ft\.?|featuring)\b)\s*", artist, flags=re.IGNORECASE)[0]
    return _sanitize_query(primary)


def _similarity(a: str, b: str) -> float:
    """Return a 0-1 similarity ratio between two strings (case-insensitive)."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _pick_best_hit(
    hits: list[dict], query_artist: str, query_title: str, threshold: float = 0.5
) -> tuple[dict, float, float] | None:
    """Score all Genius search hits and return the best match.

    Returns ``(song_info, artist_sim, title_sim)`` for the hit with the
    highest combined (artist + title) similarity, or ``None`` when the best
    candidate still falls below *threshold* on **both** dimensions.
    """
    san_artist = _sanitize_query(query_artist)
    san_title = _sanitize_query(query_title)

    best: tuple[dict, float, float, float] | None = None  # (info, a, t, combined)

    for hit in hits:
        result = hit["result"]
        a_sim = _similarity(
            san_artist,
            _sanitize_query(result.get("primary_artist", {}).get("name", "")),
        )
        t_sim = _similarity(san_title, _sanitize_query(result.get("title", "")))
        combined = a_sim + t_sim
        if best is None or combined > best[3]:
            best = (result, a_sim, t_sim, combined)

    if best is None:
        return None

    song_info, artist_sim, title_sim, _ = best

    result_artist = song_info.get("primary_artist", {}).get("name", "")
    result_title = song_info.get("title", "")
    logger.info(
        f"Genius result: '{result_artist} - {result_title}' | "
        f"similarity  artist={artist_sim:.2f}  title={title_sim:.2f}"
    )

    if artist_sim < threshold and title_sim < threshold:
        logger.warning(
            f"Genius result rejected (artist_sim={artist_sim:.2f}, "
            f"title_sim={title_sim:.2f}, threshold={threshold}): "
            f"'{result_artist} - {result_title}' does not match "
            f"'{query_artist} - {query_title}'"
        )
        return None

    return song_info, artist_sim, title_sim


def _replace_homoglyphs(text: str) -> str:
    """Replace Unicode homoglyphs with their standard Latin equivalents.

    Genius deliberately embeds Cyrillic and other lookalike characters as a
    copy-protection measure. These are visually identical to Latin letters but
    have different code points, which breaks word-level matching and produces
    garbage in output files.
    """
    # Mapping of known homoglyphs → correct Latin character
    # Covers the most common Cyrillic substitutions Genius uses
    _HOMOGLYPH_MAP: dict[str, str] = {
        # Cyrillic lowercase
        "\u0430": "a",  # а → a
        "\u0435": "e",  # е → e
        "\u0456": "i",  # і → i
        "\u043E": "o",  # о → o
        "\u0440": "r",  # р → r
        "\u0441": "c",  # с → c
        "\u0445": "x",  # х → x
        "\u0443": "y",  # у → y
        # Cyrillic uppercase
        "\u0410": "A",  # А → A
        "\u0415": "E",  # Е → E
        "\u0406": "I",  # І → I
        "\u041E": "O",  # О → O
        "\u0420": "R",  # Р → R
        "\u0421": "C",  # С → C
        "\u0425": "X",  # Х → X
        "\u0412": "B",  # В → B
        "\u041C": "M",  # М → M
        "\u0422": "T",  # Т → T
        # Greek lookalikes
        "\u03BF": "o",  # ο (omicron) → o
        "\u03B1": "a",  # α → a
        # Other common Unicode substitutions
        "\u2019": "'",  # ' (right single quotation) → '
        "\u2018": "'",  # ' (left single quotation) → '
        "\u201C": '"',  # " (left double quotation) → "
        "\u201D": '"',  # " (right double quotation) → "
        "\u2013": "-",  # – (en dash) → -
        "\u2014": "-",  # — (em dash) → -
    }
    return text.translate(str.maketrans(_HOMOGLYPH_MAP))


def _clean_genius_lyrics(lyrics: str, title: str) -> str:
    """Clean up raw Genius lyrics text.

    Removes residual section headers, embed markers, and the title line
    that Genius prepends. Also normalises Unicode homoglyphs.
    """
    # Replace Cyrillic/Greek homoglyphs and smart quotes injected by Genius
    lyrics = _replace_homoglyphs(lyrics)
    # Remove trailing "...Embed" and contributor count
    lyrics = re.sub(r"\d*Embed$", "", lyrics).strip()
    # Remove residual section headers like [Chorus], [Verse 1], [Intro], etc.
    lyrics = re.sub(r"^\[.*?\]\s*$", "", lyrics, flags=re.MULTILINE)
    # Remove song title header that Genius adds (first line)
    lines = lyrics.split("\n")
    if lines and title.lower() in lines[0].lower():
        lines = lines[1:]
    # Strip empty lines at start/end, preserve internal structure
    return "\n".join(lines).strip()


def fetch_genius_lyrics(
    artist: str, title: str, max_retries: int = 3
) -> str | None:
    """Fetch lyrics from Genius API with retry and validation.

    Searches for multiple candidates and picks the one whose artist + title
    best matches the query, avoiding the "most popular hit wins" problem.
    """
    token = os.getenv("GENIUS_ACCESS_TOKEN")
    if not token:
        logger.error("GENIUS_ACCESS_TOKEN not set")
        return None

    genius = lyricsgenius.Genius(token, verbose=False, timeout=15)

    search_artist = _primary_artist(artist)
    search_title = _sanitize_query(title)
    logger.info(f"Genius search: artist='{search_artist}', title='{search_title}'")

    search_term = f"{search_title} {search_artist}"

    response = None
    for attempt in range(1, max_retries + 1):
        try:
            response = genius.search(search_term, per_page=10)
            break
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                logger.warning(
                    f"Genius lookup attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
            else:
                logger.warning(
                    f"Genius lookup failed after {max_retries} attempts "
                    f"for '{artist} - {title}': {e}"
                )
                return None

    if not response:
        return None

    hits = response.get("hits", [])
    if not hits:
        return None

    match = _pick_best_hit(hits, artist, title)
    if match is None:
        return None

    song_info = match[0]

    # Fetch the actual lyrics from the song page
    lyrics = genius.lyrics(song_url=song_info["url"])
    if not lyrics:
        return None

    return _clean_genius_lyrics(lyrics, title)


def _clean_word(w: str) -> str:
    """Lowercase and strip punctuation for comparison."""
    return re.sub(r"[^\w]", "", w.lower())


def _estimate_insert_timing(
    opcodes: list[tuple[str, int, int, int, int]],
    op_idx: int,
    whisper_words: list[WordTiming],
    count: int,
) -> list[tuple[float, float]]:
    """Estimate timing for Genius-only words (insert ops) that Whisper missed.

    Scans neighboring opcodes for the nearest Whisper timestamps, then
    distributes `count` word slots evenly across that gap.
    """
    if count == 0:
        return []

    # Find prev_end: end time of last Whisper word before this op
    prev_end: float | None = None
    for i in range(op_idx - 1, -1, -1):
        _, pw_start, pw_end, _, _ = opcodes[i]
        if pw_end > pw_start:
            prev_end = whisper_words[pw_end - 1].end
            break
    if prev_end is None:
        prev_end = whisper_words[0].start

    # Find next_start: start time of first Whisper word after this op
    next_start: float | None = None
    for i in range(op_idx + 1, len(opcodes)):
        _, nw_start, nw_end, _, _ = opcodes[i]
        if nw_end > nw_start:
            next_start = whisper_words[nw_start].start
            break
    if next_start is None:
        next_start = whisper_words[-1].end

    # Degenerate gap: assign all words the same point timestamp
    if prev_end >= next_start:
        return [(prev_end, prev_end)] * count

    duration = next_start - prev_end
    return [
        (prev_end + duration * i / count, prev_end + duration * (i + 1) / count)
        for i in range(count)
    ]


def _interpolate_timing(
    whisper_words: list[WordTiming], w_start: int, w_end: int, count: int
) -> list[tuple[float, float]]:
    """
    Distribute timing from a range of whisper words across `count` output words.
    Returns list of (start, end) tuples.
    """
    if w_start >= w_end or count == 0:
        return []

    total_start = whisper_words[w_start].start
    total_end = whisper_words[w_end - 1].end
    total_duration = total_end - total_start

    timings = []
    for i in range(count):
        t_start = total_start + (total_duration * i / count)
        t_end = total_start + (total_duration * (i + 1) / count)
        timings.append((t_start, t_end))
    return timings


def correct_transcription(
    segments: list[Segment], genius_lyrics: str
) -> list[Segment]:
    """
    Correct transcription using Genius lyrics as reference.

    Uses difflib.SequenceMatcher for word-level alignment.
    - equal: adopt Genius spelling, keep Whisper timing
    - replace: use Genius words, interpolate timing from Whisper range
    - delete (Whisper-only words): drop them (hallucinations/repetitions)
    - insert (Genius-only words): add with timing interpolated from neighboring ops

    Then re-segment based on Genius line break positions.
    """
    # Flatten all whisper words with their timing
    whisper_words: list[WordTiming] = []
    for seg in segments:
        for w in seg.words:
            whisper_words.append(w)

    if not whisper_words:
        logger.warning("No word-level timestamps available, skipping correction")
        return segments

    whisper_text_list = [_clean_word(w.word) for w in whisper_words]

    # Parse Genius lyrics into lines and track line break positions
    genius_lines = [line.strip() for line in genius_lyrics.split("\n") if line.strip()]
    genius_words_flat: list[str] = []
    # genius_word_line[i] = which line number genius word i belongs to
    genius_word_line: list[int] = []

    for line_idx, line in enumerate(genius_lines):
        words_in_line = line.split()
        for w in words_in_line:
            genius_words_flat.append(w)
            genius_word_line.append(line_idx)

    genius_text_list = [_clean_word(w) for w in genius_words_flat]

    # Align whisper and genius words
    matcher = difflib.SequenceMatcher(None, whisper_text_list, genius_text_list)

    # Build output: list of (WordTiming, genius_word_index)
    output_words: list[tuple[WordTiming, int]] = []

    # Quality metric counters
    n_equal = 0
    n_replace = 0
    n_delete = 0
    n_insert = 0

    opcodes = matcher.get_opcodes()
    for op_idx, (op, w_start, w_end, g_start, g_end) in enumerate(opcodes):
        if op == "equal":
            n_equal += w_end - w_start
            # Use Genius spelling with Whisper timing
            for wi, gi in zip(range(w_start, w_end), range(g_start, g_end)):
                output_words.append((
                    WordTiming(
                        start=whisper_words[wi].start,
                        end=whisper_words[wi].end,
                        word=genius_words_flat[gi],
                    ),
                    gi,
                ))

        elif op == "replace":
            n_replace += max(w_end - w_start, g_end - g_start)
            # Use all Genius words, distribute Whisper timing range across them
            g_count = g_end - g_start
            timings = _interpolate_timing(whisper_words, w_start, w_end, g_count)
            for idx, gi in enumerate(range(g_start, g_end)):
                t_start, t_end = timings[idx]
                output_words.append((
                    WordTiming(
                        start=t_start,
                        end=t_end,
                        word=genius_words_flat[gi],
                    ),
                    gi,
                ))

        elif op == "delete":
            n_delete += w_end - w_start
            # Whisper has words that Genius doesn't - drop them (hallucinations)
            dropped = " ".join(whisper_words[i].word.strip() for i in range(w_start, w_end))
            logger.debug(f"Dropping Whisper-only words: '{dropped}'")

        elif op == "insert":
            n_insert += g_end - g_start
            # Genius has words that Whisper missed - insert with estimated timing
            g_count = g_end - g_start
            timings = _estimate_insert_timing(opcodes, op_idx, whisper_words, g_count)
            inserted = " ".join(genius_words_flat[i] for i in range(g_start, g_end))
            logger.debug(f"Inserting Genius-only words with estimated timing: '{inserted}'")
            for idx, gi in enumerate(range(g_start, g_end)):
                t_start, t_end = timings[idx]
                output_words.append((
                    WordTiming(start=t_start, end=t_end, word=genius_words_flat[gi]),
                    gi,
                ))

    # Log correction quality metric
    total_aligned = n_equal + n_replace
    match_pct = (n_equal / total_aligned * 100) if total_aligned > 0 else 0.0
    logger.info(
        f"Correction quality: {match_pct:.1f}% exact match "
        f"(equal={n_equal}, replace={n_replace}, delete={n_delete}, insert={n_insert})"
    )
    if match_pct < 30:
        logger.warning(
            f"Very low match rate ({match_pct:.1f}%) — Genius lyrics may not match this song"
        )

    if not output_words:
        logger.warning("Correction produced no output, returning original segments")
        return segments

    # Use Genius capitalization as-is. Genius is far more reliable than Whisper
    # for capitalization, especially for German (where nouns are capitalized).
    # No "fix" needed - Genius spelling/casing is the ground truth.

    # Re-segment based on Genius line structure
    new_segments: list[Segment] = []
    current_words: list[WordTiming] = []
    current_line = -1

    for wt, gi in output_words:
        word_line = genius_word_line[gi]

        # Start new segment when Genius line changes
        if word_line != current_line and current_words:
            text = " ".join(w.word.strip() for w in current_words)
            new_segments.append(
                Segment(
                    start=current_words[0].start,
                    end=current_words[-1].end,
                    text=text,
                    words=list(current_words),
                )
            )
            current_words = []

        current_words.append(wt)
        current_line = word_line

    # Flush remaining words
    if current_words:
        text = " ".join(w.word.strip() for w in current_words)
        new_segments.append(
            Segment(
                start=current_words[0].start,
                end=current_words[-1].end,
                text=text,
                words=list(current_words),
            )
        )

    logger.info(
        f"Correction complete: {len(segments)} original segments -> "
        f"{len(new_segments)} corrected lines ({match_pct:.1f}% match)"
    )
    return new_segments
