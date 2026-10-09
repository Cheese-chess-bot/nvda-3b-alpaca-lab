"""Qwen2.5-3B adapter, kept separate from Gemma and PPO."""
from model_runtime import FrozenTextAnalyzer


class QwenAnalyzer(FrozenTextAnalyzer):
    @staticmethod
    def model_class():
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM
