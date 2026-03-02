import os
import logging
from typing import Callable

from app.separation import separate_vocals, unload_model as unload_separation_model
from app.transcription import transcribe, write_output_files
from app.correction import get_metadata_from_file, fetch_genius_lyrics, correct_transcription

logger = logging.getLogger(__name__)


def run_full_pipeline(
    input_path: str,
    output_dir: str,
    format: str = "lrc",
    no_separation: bool = False,
    whisper_model: str = "large-v3-turbo",
    language: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    no_correction: bool = False,
    progress_callback: Callable[[str], None] | None = None,
    warning_callback: Callable[[str], None] | None = None,
) -> list[str]:
    """Run the full transcription pipeline. Returns list of output file paths."""
    stem = os.path.splitext(os.path.basename(input_path))[0]

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)

    # Step 1: Vocal separation
    if no_separation:
        vocals_path = input_path
        report("Skipping vocal separation")
    else:
        report("Separating vocals...")
        vocals_path = separate_vocals(input_path, output_dir)

    # Free GPU memory from separation before loading Whisper
    if not no_separation:
        unload_separation_model()

    # Step 2: Transcription
    report("Transcribing audio")
    # Resolve artist/title for both transcription prompt and Genius lookup
    file_artist, file_title = get_metadata_from_file(input_path)
    lookup_artist = artist or file_artist
    lookup_title = title or file_title

    segments, detected_lang = transcribe(
        vocals_path,
        model_size=whisper_model,
        language=language,
        artist=lookup_artist,
        title=lookup_title,
        language_callback=lambda lang: report(f"Detected language: {lang}"),
    )

    # Check if any vocals were detected
    has_content = any(seg.text.strip() for seg in segments)
    if not has_content:
        logger.warning("No vocals detected in audio, skipping output file creation")
        if warning_callback:
            warning_callback("No vocals detected in audio")
        report("Complete - no vocals detected")
        # Clean up temporary vocals file
        if not no_separation and os.path.exists(vocals_path) and vocals_path != input_path:
            os.remove(vocals_path)
            logger.info(f"Cleaned up temporary vocals file: {vocals_path}")
        return []

    # Step 3: Lyrics correction
    if not no_correction:
        report("Correcting lyrics via Genius")

        if lookup_artist and lookup_title:
            genius_lyrics = fetch_genius_lyrics(lookup_artist, lookup_title)
            if genius_lyrics:
                segments = correct_transcription(segments, genius_lyrics)
            else:
                logger.warning("No Genius lyrics found, skipping correction")
        else:
            logger.warning("No artist/title available for Genius lookup, skipping correction")

    # Step 4: Write output files
    report("Writing output files...")
    output_files = write_output_files(segments, output_dir, stem, format)

    # Clean up temporary vocals file
    if not no_separation and os.path.exists(vocals_path) and vocals_path != input_path:
        os.remove(vocals_path)
        logger.info(f"Cleaned up temporary vocals file: {vocals_path}")

    report("Complete")
    return output_files
