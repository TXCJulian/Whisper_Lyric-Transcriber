import os
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.transcription import Segment, WordTiming

logger = logging.getLogger(__name__)

IDLE_UNLOAD_SECONDS = float(os.getenv("WHISPER_IDLE_UNLOAD_SECONDS", "15"))


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

    # faster-whisper model names that don't exist in OpenAI Whisper
    _MODEL_MAP: dict[str, str] = {
        "large-v3-turbo": "large-v3",
        "turbo": "large-v3",
    }

    def __init__(self):
        super().__init__()
        from app.gpu_backend import get_device
        self._device = get_device()
        self._model = None
        self._model_size: str | None = None

    def _resolve_model_name(self, model_size: str) -> str:
        """Map faster-whisper model names to OpenAI Whisper equivalents."""
        resolved = self._MODEL_MAP.get(model_size, model_size)
        if resolved != model_size:
            logger.info(
                f"[openai-whisper] Mapped model '{model_size}' -> '{resolved}'"
            )
        return resolved

    def _get_model(self, model_size: str):
        resolved = self._resolve_model_name(model_size)
        if self._model is None or self._model_size != resolved:
            import whisper

            logger.info(
                f"[openai-whisper] Loading '{resolved}' on {self._device}"
            )
            self._model = whisper.load_model(resolved, device=self._device)
            self._model_size = resolved
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

        kwargs: dict[str, Any] = {
            "word_timestamps": True,
            "condition_on_previous_text": False,
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

        result = whisper.transcribe(model, audio_path, **kwargs)
        detected_language = result.get("language", "unknown")
        logger.info(f"Detected language: {detected_language}")

        if language_callback:
            language_callback(detected_language)

        segments = []
        for seg in result.get("segments", []):
            words = []
            for w in seg.get("words", []):
                words.append(
                    WordTiming(start=w["start"], end=w["end"], word=w["word"])
                )
            segments.append(
                Segment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                    words=words,
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
