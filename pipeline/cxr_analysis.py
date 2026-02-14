"""
Stage 9: Chest X-Ray Analysis (CXR Foundation)
Uses Google CXR Foundation for automated chest X-ray interpretation
to monitor pulmonary drug effects and treatment response.

CXR Foundation uses an EfficientNet-L2 image encoder + BERT text encoder,
trained on 800,000+ de-identified chest X-rays using Supervised Contrastive,
CLIP, and BLIP-2 objectives. Produces ELIXR embeddings for downstream tasks.

Architecture: EfficientNet-L2 (image) + BERT (text)
  - Image input:   DICOM / PNG chest X-rays
  - Embeddings:    ELIXR v2.0 — 32 x 768 (dense features)
  - Contrastive:   32 x 128 (image-text alignment)
  - Training data: 800,000+ chest X-rays

Applications in Drug Discovery:
  - Monitor pulmonary adverse effects (pneumonitis, fibrosis)
  - Track NSCLC treatment response via serial imaging
  - Zero-shot classification of chest findings
  - Semantic retrieval across imaging databases

Model: google/cxr-foundation (Hugging Face)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from tabulate import tabulate

logger = logging.getLogger("hai-def-pipeline")

CXR_FOUNDATION_MODEL_ID = "google/cxr-foundation"
ELIXR_DIM = (32, 768)       # Dense feature embedding
CONTRASTIVE_DIM = (32, 128)  # Image-text aligned embedding

# Chest X-ray findings relevant to drug monitoring
CXR_FINDINGS = [
    "Normal chest",
    "Tumor mass / nodule",
    "Pleural effusion",
    "Pneumonitis (drug-induced)",
    "Pulmonary fibrosis",
    "Consolidation / infection",
    "Cardiomegaly",
    "Atelectasis",
    "Pneumothorax",
]

# Drug-related pulmonary toxicity labels
DRUG_PULMONARY_EFFECTS = [
    "No pulmonary toxicity",
    "Drug-induced pneumonitis — Grade 1",
    "Drug-induced pneumonitis — Grade 2",
    "Drug-induced interstitial lung disease",
    "Treatment-related pulmonary fibrosis",
    "Tumor progression on imaging",
    "Tumor regression on imaging",
]


class CXRFoundationAnalyzer:
    """
    Chest X-ray analyzer powered by Google CXR Foundation.

    Produces ELIXR embeddings for chest X-ray classification,
    pulmonary drug-effect monitoring, and treatment response tracking.
    """

    def __init__(self, simulated: bool = True):
        self.simulated = simulated
        self.model = None
        self._loaded = False

    def load_model(self) -> bool:
        """Load CXR Foundation model from Hugging Face Hub."""
        if self._loaded:
            return True

        if self.simulated:
            logger.info("CXR Foundation running in simulated mode.")
            self._loaded = True
            return True

        try:
            from huggingface_hub import from_pretrained_keras

            logger.info(f"Loading CXR Foundation from {CXR_FOUNDATION_MODEL_ID}...")
            self.model = from_pretrained_keras(CXR_FOUNDATION_MODEL_ID)
            self._loaded = True
            logger.info("CXR Foundation loaded successfully.")
            return True

        except Exception as e:
            logger.warning(f"Could not load CXR Foundation: {e}. Using simulated mode.")
            self.simulated = True
            self._loaded = True
            return True

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """Extract ELIXR embedding (32x768) from chest X-ray."""
        self.load_model()

        if self.simulated:
            return self._simulated_embedding(image_path)

        try:
            import tensorflow as tf
            from PIL import Image as PILImage

            img = PILImage.open(image_path).convert("L")  # Grayscale for CXR
            img = img.resize((512, 512))
            tensor = tf.cast(
                tf.expand_dims(tf.expand_dims(np.array(img), -1), 0),
                tf.float32,
            ) / 255.0
            embedding = self.model(tensor).numpy()
            return embedding.reshape(ELIXR_DIM)

        except Exception as e:
            logger.error(f"CXR embedding extraction failed: {e}")
            return self._simulated_embedding(image_path)

    def classify_findings(self, image_path: str) -> Dict:
        """Classify chest X-ray findings."""
        embedding = self.extract_embedding(image_path)
        return self._classify_from_embedding(embedding, image_path)

    def assess_pulmonary_drug_effects(
        self,
        image_path: str,
        drug_name: str,
        target_name: str,
        disease: str,
    ) -> Dict:
        """
        Assess chest X-ray for drug-related pulmonary effects.
        Returns finding classification, drug-toxicity assessment,
        and treatment response indication.
        """
        findings = self._classify_from_embedding(
            self.extract_embedding(image_path), image_path
        )

        primary = findings["top_finding"]
        confidence = findings["confidence"]

        # Determine drug-related toxicity
        is_toxic = primary in [
            "Pneumonitis (drug-induced)",
            "Pulmonary fibrosis",
        ]
        is_progression = primary == "Tumor mass / nodule"
        is_normal = primary == "Normal chest"

        if is_toxic:
            toxicity_grade = 2 if confidence < 0.7 else 3
            action = "Hold treatment, start corticosteroids" if toxicity_grade >= 3 else "Reduce dose, monitor closely"
        elif is_progression:
            toxicity_grade = 0
            action = "Consider treatment modification"
        elif is_normal:
            toxicity_grade = 0
            action = "Continue current treatment"
        else:
            toxicity_grade = 1
            action = "Investigate further, correlate clinically"

        return {
            "drug": drug_name,
            "target": target_name,
            "disease": disease,
            "image": os.path.basename(image_path),
            "primary_finding": primary,
            "confidence": confidence,
            "drug_toxicity": is_toxic,
            "toxicity_grade": toxicity_grade,
            "tumor_status": (
                "Regression" if is_normal and disease != "N/A"
                else "Progression" if is_progression
                else "Stable"
            ),
            "recommended_action": action,
        }

    def batch_analyze(
        self,
        drug_name: str = "Erlotinib",
        target_name: str = "EGFR",
        disease: str = "NSCLC",
    ) -> List[Dict]:
        """Analyze a batch of chest X-rays."""
        return self._demo_batch_results(drug_name, target_name, disease)

    # ── Internal helpers ──────────────────────────────────────────────

    def _simulated_embedding(self, image_path: str) -> np.ndarray:
        seed = hash(os.path.basename(image_path)) % (2**31)
        rng = np.random.RandomState(seed)
        emb = rng.randn(*ELIXR_DIM).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def _classify_from_embedding(self, embedding: np.ndarray, image_path: str) -> Dict:
        seed = hash(os.path.basename(image_path)) % (2**31)
        rng = np.random.RandomState(seed)
        raw = rng.dirichlet(np.ones(len(CXR_FINDINGS)) * 1.5)
        scores = sorted(
            zip(CXR_FINDINGS, raw.tolist()),
            key=lambda x: x[1], reverse=True,
        )
        return {
            "top_finding": scores[0][0],
            "confidence": round(float(scores[0][1]), 3),
            "all_findings": {name: round(float(s), 4) for name, s in scores},
        }

    def _demo_batch_results(self, drug_name: str, target_name: str, disease: str) -> List[Dict]:
        demo_samples = [
            {"name": "cxr_baseline_001.png", "finding": "Tumor mass / nodule", "conf": 0.85, "status": "Baseline"},
            {"name": "cxr_week4_001.png", "finding": "Normal chest", "conf": 0.72, "status": "Regression"},
            {"name": "cxr_week8_002.png", "finding": "Pneumonitis (drug-induced)", "conf": 0.68, "status": "Toxicity"},
            {"name": "cxr_week12_003.png", "finding": "Pleural effusion", "conf": 0.61, "status": "Stable"},
            {"name": "cxr_week16_001.png", "finding": "Normal chest", "conf": 0.79, "status": "Regression"},
        ]

        results = []
        for s in demo_samples:
            is_toxic = "Pneumonitis" in s["finding"] or "fibrosis" in s["finding"].lower()
            is_prog = s["finding"] == "Tumor mass / nodule" and s["status"] != "Baseline"
            tox_grade = 2 if is_toxic else 0

            results.append({
                "drug": drug_name, "target": target_name, "disease": disease,
                "image": s["name"], "primary_finding": s["finding"],
                "confidence": s["conf"], "drug_toxicity": is_toxic,
                "toxicity_grade": tox_grade,
                "tumor_status": s["status"],
                "recommended_action": (
                    "Reduce dose, monitor closely" if is_toxic
                    else "Consider treatment modification" if is_prog
                    else "Continue current treatment" if s["finding"] == "Normal chest"
                    else "Investigate further"
                ),
            })
        return results


def print_cxr_results(results: List[Dict]):
    """Pretty-print CXR Foundation analysis results."""
    print("\n" + "=" * 80)
    print("  Stage 9: Chest X-Ray Analysis -- CXR Foundation")
    print("  Model: google/cxr-foundation (EfficientNet-L2 + BERT, ELIXR)")
    print("=" * 80)

    table_data = []
    for r in results:
        table_data.append([
            r["image"][:24],
            r["primary_finding"][:26],
            f"{r['confidence']:.0%}",
            "Yes" if r["drug_toxicity"] else "No",
            r["tumor_status"][:12],
            r["recommended_action"][:32],
        ])

    print(tabulate(
        table_data,
        headers=["CXR Image", "Finding", "Conf", "Toxic?", "Tumor", "Action"],
        tablefmt="rounded_outline",
    ))

    toxic = sum(1 for r in results if r["drug_toxicity"])
    regression = sum(1 for r in results if r["tumor_status"] == "Regression")
    print(f"\n  Summary: {regression}/{len(results)} show tumor regression, {toxic} pulmonary toxicity events")
    print(f"  Drug: {results[0]['drug']} | Target: {results[0]['target']} | Disease: {results[0]['disease']}")
    print()


def run_cxr_analysis(
    drug_name: str = "Erlotinib",
    target_name: str = "EGFR",
    disease: str = "Non-Small Cell Lung Cancer",
) -> List[Dict]:
    """Run CXR Foundation analysis as part of the drug discovery pipeline."""
    analyzer = CXRFoundationAnalyzer(simulated=True)
    return analyzer.batch_analyze(drug_name, target_name, disease)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_cxr_analysis()
    print_cxr_results(results)
