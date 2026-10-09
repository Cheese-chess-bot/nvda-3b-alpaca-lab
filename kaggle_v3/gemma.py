"""Gemma 3 4B adapter. This version supplies text only, not chart images."""
from model_runtime import FrozenTextAnalyzer


class GemmaAnalyzer(FrozenTextAnalyzer):
    @staticmethod
    def model_class():
        # The official 4B checkpoint has the multimodal architecture even for text-only input.
        from transformers import Gemma3ForConditionalGeneration
        return Gemma3ForConditionalGeneration
