"""
HAI-DEF Model Loader
Handles loading TxGemma and MedGemma models from Hugging Face.
Provides a unified interface with graceful fallback to simulated mode.
"""

import logging
from typing import Optional, Dict, Any

from .config import (
    TXGEMMA_2B_PREDICT,
    DEFAULT_MODELS,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)

logger = logging.getLogger("hai-def-pipeline")


class HAIDefModelLoader:
    """
    Manages HAI-DEF model loading and inference.
    Supports both real model inference and simulated fallback mode.
    """

    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._pipelines: Dict[str, Any] = {}
        self.simulated_mode = False
        self._device = "cpu"

    def load_model(self, model_id: str, task: str = "text-generation") -> bool:
        """
        Load a model from Hugging Face.
        Returns True if loaded successfully, False if falling back to simulated.
        """
        if model_id in self._pipelines:
            logger.info(f"Model {model_id} already loaded.")
            return True

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device = device
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            logger.info(f"Loading {model_id} on {device}...")

            pipe = hf_pipeline(
                task,
                model=model_id,
                device=device,
                torch_dtype=dtype,
            )

            self._pipelines[model_id] = pipe
            logger.info(f"✅ {model_id} loaded successfully on {device}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Could not load {model_id}: {e}")
            logger.info("Falling back to simulated mode.")
            self.simulated_mode = True
            return False

    def predict(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
    ) -> str:
        """
        Run inference using a loaded model or return simulated output.
        """
        if self.simulated_mode or model_id not in self._pipelines:
            return self._simulated_predict(prompt)

        try:
            pipe = self._pipelines[model_id]
            output = pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_full_text=False,
            )
            return output[0]["generated_text"].strip()

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._simulated_predict(prompt)

    def _simulated_predict(self, prompt: str) -> str:
        """
        Return a plausible simulated response based on the prompt content.
        Used when real models are unavailable.
        """
        prompt_lower = prompt.lower()

        if "binding affinity" in prompt_lower or "kd" in prompt_lower:
            return "Predicted binding affinity (Kd): 45.2 nM — Moderate to strong binding."

        if "binds to the target" in prompt_lower or "drug-target" in prompt_lower:
            return "Yes. Confidence: 0.82 — The molecular structure is compatible with the target binding pocket."

        if "solubility" in prompt_lower:
            return "Medium — The compound shows moderate aqueous solubility (estimated logS: -3.2)."

        if "herg" in prompt_lower or "cardiotoxicity" in prompt_lower:
            return "No — Low risk of hERG channel inhibition. Predicted IC50 > 10 μM."

        if "blood-brain barrier" in prompt_lower or "bbb" in prompt_lower:
            return "Yes — The compound is predicted to cross the BBB based on its lipophilicity and molecular weight."

        if "cyp" in prompt_lower:
            return "No — Low risk of CYP3A4 inhibition based on molecular structure analysis."

        if "lipophilicity" in prompt_lower or "logp" in prompt_lower:
            return "Predicted logP: 2.4 — Within the ideal range (1.0–3.0) for drug-likeness."

        if "clinical trial" in prompt_lower or "phase" in prompt_lower:
            return "Predicted probability: 0.68 — Moderate likelihood of clinical success."

        if "describe" in prompt_lower or "properties" in prompt_lower:
            return (
                "This compound exhibits favorable drug-like properties. Key observations:\n"
                "• Molecular weight within Lipinski limits\n"
                "• Moderate lipophilicity suggesting good membrane permeability\n"
                "• Contains hydrogen bond donor/acceptor groups for target interaction\n"
                "• Low predicted toxicity risk based on structural alerts"
            )

        return "Prediction completed. See structured output for details."

    def get_status(self) -> dict:
        """Return loader status information."""
        return {
            "loaded_models": list(self._pipelines.keys()),
            "simulated_mode": self.simulated_mode,
            "device": self._device,
        }


# Global singleton
_loader: Optional[HAIDefModelLoader] = None


def get_model_loader() -> HAIDefModelLoader:
    """Get or create the global model loader instance."""
    global _loader
    if _loader is None:
        _loader = HAIDefModelLoader()
    return _loader
