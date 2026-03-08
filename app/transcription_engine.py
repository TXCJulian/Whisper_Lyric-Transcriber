import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.transcription import Segment, WordTiming

logger = logging.getLogger(__name__)


class TranscriptionEngine(ABC):
    """Base class for transcription engines."""

    @abstractmethod
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
        from app.gpu_backend import get_backend
        self._models: dict[str, Any] = {}
        self._backend = get_backend()

    def _get_model(self, model_size: str):
        if model_size not in self._models:
            from faster_whisper import WhisperModel

            device = "cuda" if self._backend == "cuda" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(
                f"[faster-whisper] Loading '{model_size}' on {device} ({compute_type})"
            )
            self._models[model_size] = WhisperModel(
                model_size, device=device, compute_type=compute_type
            )
        return self._models[model_size]

    def transcribe(
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
        self._models.clear()
        logger.info("[faster-whisper] Models unloaded")


class OpenAIWhisperEngine(TranscriptionEngine):
    """Transcription engine using OpenAI Whisper (PyTorch). For XPU and ROCm."""

    def __init__(self):
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

    def transcribe(
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
