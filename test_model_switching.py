"""Model lifecycle regression tests; run with python3 -m unittest discover.

The real engine code runs with fake external model loaders so these tests need
neither GPU hardware nor downloaded models. NumPy is unused in this code path.
"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import weakref


class Model:
    pass


class ModelSwitchingTests(unittest.TestCase):
    def setUp(self):
        # Import an isolated copy without leaving a fake NumPy in other tests.
        spec = importlib.util.spec_from_file_location(
            "engine_under_test",
            Path(__file__).parent / "app" / "transcription_engine.py",
        )
        self.engines = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"numpy": ModuleType("numpy")}):
            spec.loader.exec_module(self.engines)

    def make_engine(self, engine_name):
        self.cache_cleared = False

        def empty_cache():
            self.cache_cleared = True

        backend = SimpleNamespace(
            get_device=lambda: SimpleNamespace(type="xpu"),
            get_backend=lambda: "cuda",
            empty_cache=empty_cache,
        )
        modules = patch.dict(sys.modules, {
            "app.gpu_backend": backend,
            "whisper": SimpleNamespace(load_model=self.load_model),
            "faster_whisper": SimpleNamespace(WhisperModel=self.load_model),
        })
        modules.start()
        self.addCleanup(modules.stop)
        return getattr(self.engines, engine_name)()

    def load_model(self, name, **kwargs):
        return Model()

    def test_switch_releases_previous_model_and_cache_before_loading(self):
        for name in ("OpenAIWhisperEngine", "FasterWhisperEngine"):
            with self.subTest(engine=name):
                engine = self.make_engine(name)
                engine.load_model("large-v3-turbo")
                previous = weakref.ref(engine._model)

                def replacement(model_size, **kwargs):
                    self.assertIsNone(previous(), "old model still alive during replacement load")
                    self.assertTrue(self.cache_cleared, "GPU cache not released before loading")
                    return Model()

                loader = sys.modules["whisper"] if name == "OpenAIWhisperEngine" else sys.modules["faster_whisper"]
                attribute = "load_model" if name == "OpenAIWhisperEngine" else "WhisperModel"
                with patch.object(loader, attribute, replacement):
                    engine.load_model("large-v3")
                self.assertIsNotNone(engine._model)
                self.assertEqual(engine._model_size, "large-v3")

    def test_same_model_is_reused_without_clearing_cache(self):
        for name in ("OpenAIWhisperEngine", "FasterWhisperEngine"):
            with self.subTest(engine=name):
                engine = self.make_engine(name)
                engine.load_model("small")
                original = engine._model
                engine.load_model("small")
                self.assertIs(engine._model, original)
                self.assertFalse(self.cache_cleared)

    def test_failed_switch_leaves_empty_state_and_allows_retry(self):
        for name in ("OpenAIWhisperEngine", "FasterWhisperEngine"):
            with self.subTest(engine=name):
                engine = self.make_engine(name)
                engine.load_model("small")
                previous = weakref.ref(engine._model)
                loader = sys.modules["whisper"] if name == "OpenAIWhisperEngine" else sys.modules["faster_whisper"]
                attribute = "load_model" if name == "OpenAIWhisperEngine" else "WhisperModel"
                with patch.object(loader, attribute, side_effect=RuntimeError("load failed")):
                    with self.assertRaisesRegex(RuntimeError, "load failed"):
                        engine.load_model("large-v3")
                self.assertIsNone(previous())
                self.assertIsNone(engine._model)
                self.assertIsNone(engine._model_size)
                engine.load_model("large-v3")
                self.assertIsNotNone(engine._model)
                self.assertEqual(engine._model_size, "large-v3")


if __name__ == "__main__":
    unittest.main()
