import os
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from app.transcription import Segment, WordTiming

logger = logging.getLogger(__name__)

IDLE_UNLOAD_SECONDS = float(os.getenv("WHISPER_IDLE_UNLOAD_SECONDS", "15"))

# VAD's default min_silence_duration_ms (2000ms) treats any gap shorter than
# that as still "one continuous speech region", so short instrumental pauses
# between lyric lines never get trimmed out -- Whisper then decodes straight
# across the raw silence and its own timestamp placement there is imprecise,
# often landing on the next line well before it's actually sung. Lowering
# the threshold makes VAD split those pauses out too, so the word timestamp
# anchors to the real, VAD-detected speech onset instead.
VAD_MIN_SILENCE_MS = 500


class TranscriptionEngine(ABC):
    """Base class for transcription engines.

    Wraps `_transcribe` with an idle-unload timer: after each call the model
    stays resident for `IDLE_UNLOAD_SECONDS` in case another job follows
    right away (e.g. a batch of songs), then frees GPU memory automatically.
    """

    def __init__(self):
        self._idle_timer: threading.Timer | None = None
        self._timer_lock = threading.Lock()
        # Guards actual model access so the idle-unload timer can never fire
        # concurrently with an in-progress _transcribe() call.
        self._model_lock = threading.Lock()

    def transcribe(
        self,
        audio_path: str,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        language_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[Segment], str]:
        """Transcribe audio. Returns (segments, detected_language)."""
        self._cancel_idle_timer()
        try:
            with self._model_lock:
                return self._transcribe(
                    audio_path,
                    model_size=model_size,
                    language=language,
                    artist=artist,
                    title=title,
                    language_callback=language_callback,
                )
        finally:
            self._schedule_idle_unload()

    def _cancel_idle_timer(self) -> None:
        with self._timer_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None

    def _schedule_idle_unload(self) -> None:
        with self._timer_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
            timer = threading.Timer(IDLE_UNLOAD_SECONDS, self._idle_unload)
            timer.daemon = True
            self._idle_timer = timer
            timer.start()

    def _idle_unload(self) -> None:
        with self._timer_lock:
            self._idle_timer = None
        logger.info(
            f"No transcription for {IDLE_UNLOAD_SECONDS:.0f}s, unloading model"
        )
        with self._model_lock:
            self.unload_model()

    @abstractmethod
    def _transcribe(
        self,
        audio_path: str,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        language_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[Segment], str]:
        """Engine-specific transcription. Returns (segments, detected_language)."""
        ...

    @abstractmethod
    def load_model(self, model_size: str = "large-v3-turbo") -> None:
        """Pre-load a model into memory."""
        ...

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model(s) from memory."""
        ...


class FasterWhisperEngine(TranscriptionEngine):
    """Transcription engine using faster-whisper (CTranslate2). For CUDA and CPU."""

    def __init__(self):
        super().__init__()
        from app.gpu_backend import get_backend
        self._model: Any = None
        self._model_size: str | None = None
        self._backend = get_backend()

    def _get_model(self, model_size: str):
        if self._model is None or self._model_size != model_size:
            from faster_whisper import WhisperModel

            device = "cuda" if self._backend == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(
                f"[faster-whisper] Loading '{model_size}' on {device} ({compute_type})"
            )
            self._model = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
            self._model_size = model_size
        return self._model

    def _transcribe(
        self,
        audio_path: str,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        language_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[Segment], str]:
        model = self._get_model(model_size)

        kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "vad_filter": True,
            "vad_parameters": {"min_silence_duration_ms": VAD_MIN_SILENCE_MS},
            "condition_on_previous_text": False,
        }
        if language:
            kwargs["language"] = language
        if artist and title:
            kwargs["initial_prompt"] = f"{artist} - {title}"
            logger.info(f"Using initial_prompt: '{kwargs['initial_prompt']}'")
        elif artist:
            kwargs["initial_prompt"] = artist
        elif title:
            kwargs["initial_prompt"] = title

        segments_iter, info = model.transcribe(audio_path, **kwargs)
        detected_language = info.language
        logger.info(
            f"Detected language: {detected_language} "
            f"(prob: {info.language_probability:.2f})"
        )

        if language_callback:
            language_callback(detected_language)

        results = []
        for seg in segments_iter:
            words = []
            if seg.words:
                words = [
                    WordTiming(start=w.start, end=w.end, word=w.word)
                    for w in seg.words
                ]
            results.append(
                Segment(
                    start=seg.start, end=seg.end, text=seg.text.strip(), words=words
                )
            )

        logger.info(f"Transcription complete: {len(results)} segments")
        return results, detected_language

    def load_model(self, model_size: str = "large-v3-turbo") -> None:
        self._get_model(model_size)

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_size = None
            if self._backend == "cuda":
                from app.gpu_backend import empty_cache
                empty_cache()
            logger.info("[faster-whisper] Model unloaded")


class OpenAIWhisperEngine(TranscriptionEngine):
    """Transcription engine using OpenAI Whisper (PyTorch). For XPU and ROCm."""

    def __init__(self):
        super().__init__()
        from app.gpu_backend import get_device
        self._device = get_device()
        self._model = None
        self._model_size: str | None = None

    def _get_model(self, model_size: str):
        if self._model is None or self._model_size != model_size:
            import whisper

            logger.info(
                f"[openai-whisper] Loading '{model_size}' on {self._device}"
            )
            self._model = whisper.load_model(model_size, device=self._device)
            self._model_size = model_size
        return self._model

    def _transcribe(
        self,
        audio_path: str,
        model_size: str = "large-v3-turbo",
        language: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        language_callback: Callable[[str], None] | None = None,
    ) -> tuple[list[Segment], str]:
        model = self._get_model(model_size)
        import whisper
        from faster_whisper.vad import (
            SpeechTimestampsMap,
            VadOptions,
            collect_chunks,
            get_speech_timestamps,
        )

        # openai-whisper has no built-in VAD (unlike faster-whisper's
        # vad_filter=True), so without this it hallucinates lyrics over
        # instrumental/silent stretches. faster-whisper is a dependency on
        # every backend, so we reuse its bundled Silero VAD here and remap
        # the resulting timestamps back to the original audio afterwards,
        # the same way faster-whisper does internally.
        sampling_rate = whisper.audio.SAMPLE_RATE
        audio = whisper.load_audio(audio_path)
        vad_options = VadOptions(min_silence_duration_ms=VAD_MIN_SILENCE_MS)
        speech_chunks = get_speech_timestamps(
            audio, vad_options, sampling_rate=sampling_rate
        )
        ts_map = None
        if speech_chunks:
            audio_chunks, _ = collect_chunks(
                audio, speech_chunks, sampling_rate=sampling_rate
            )
            audio = (
                audio_chunks[0] if len(audio_chunks) == 1 else np.concatenate(audio_chunks)
            )
            ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)
        else:
            logger.warning("VAD detected no speech; transcribing full audio")

        kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "condition_on_previous_text": False,
            # faster-whisper's default; openai-whisper's own default (None)
            # falls back to greedy decoding instead of beam search.
            "beam_size": 5,
        }
        if language:
            kwargs["language"] = language

        initial_prompt = None
        if artist and title:
            initial_prompt = f"{artist} - {title}"
        elif artist:
            initial_prompt = artist
        elif title:
            initial_prompt = title
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
            logger.info(f"Using initial_prompt: '{initial_prompt}'")

        result = whisper.transcribe(model, audio, **kwargs)
        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")

        if language_callback:
            language_callback(detected_language)

        segments = []
        for seg in result.get("segments", []):
            words = [
                WordTiming(start=w["start"], end=w["end"], word=w["word"])
                for w in seg.get("words", [])
            ]

            if ts_map is not None and words:
                for word in words:
                    chunk_index = ts_map.get_chunk_index((word.start + word.end) / 2)
                    word.start = ts_map.get_original_time(word.start, chunk_index)
                    word.end = ts_map.get_original_time(word.end, chunk_index)
                seg_start, seg_end = words[0].start, words[-1].end
            elif ts_map is not None:
                seg_start = ts_map.get_original_time(seg["start"])
                seg_end = ts_map.get_original_time(seg["end"], is_end=True)
            else:
                seg_start, seg_end = seg["start"], seg["end"]

            segments.append(
                Segment(
                    start=seg_start, end=seg_end, text=seg["text"].strip(), words=words
                )
            )

        logger.info(f"Transcription complete: {len(segments)} segments")
        return segments, detected_language

    def load_model(self, model_size: str = "large-v3-turbo") -> None:
        self._get_model(model_size)

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_size = None
            from app.gpu_backend import empty_cache
            empty_cache()
            logger.info("[openai-whisper] Model unloaded")


# ── Singleton engine instance ──────────────────────────────────────────────

_engine: TranscriptionEngine | None = None


def get_engine() -> TranscriptionEngine:
    """Return the singleton transcription engine for the active backend."""
    global _engine
    if _engine is None:
        from app.gpu_backend import use_faster_whisper
        if use_faster_whisper():
            _engine = FasterWhisperEngine()
            logger.info("Using faster-whisper transcription engine")
        else:
            _engine = OpenAIWhisperEngine()
            logger.info("Using OpenAI Whisper transcription engine")
    return _engine
