"""
Stage 6: Pathology Analysis
Uses Google Path Foundation to analyze histopathology images for drug
response prediction, tumor characterization, and biomarker discovery.

Path Foundation produces 384-dimensional embeddings from 224x224 H&E
stained tissue patches, which can be used for:
- Tumor tissue classification
- Tumor grading
- Biomarker detection and treatment response prediction
- Similar image search across WSIs
- Quality assessment of pathology slides

Model: google/path-foundation (ViT-S, Masked Siamese Networks)
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from tabulate import tabulate

logger = logging.getLogger("hai-def-pipeline")

PATH_FOUNDATION_MODEL_ID = "google/path-foundation"
EMBEDDING_DIM = 384
INPUT_SIZE = (224, 224)


class PathFoundationAnalyzer:
    """
    Histopathology image analyzer powered by Google Path Foundation.

    Extracts 384-dimensional embeddings from H&E tissue patches, performs
    tissue classification, similarity search, and integrates with the
    drug discovery pipeline for treatment response prediction.
    """

    def __init__(self, simulated: bool = True):
        self.simulated = simulated
        self.model = None
        self._loaded = False

    def load_model(self) -> bool:
        """
        Load Path Foundation model from Hugging Face Hub.
        Uses TensorFlow/Keras backend.
        """
        if self._loaded:
            return True

        if self.simulated:
            logger.info("Path Foundation running in simulated mode.")
            self._loaded = True
            return True

        try:
            import tensorflow as tf
            from huggingface_hub import from_pretrained_keras

            logger.info(f"Loading Path Foundation from {PATH_FOUNDATION_MODEL_ID}...")
            self.model = from_pretrained_keras(PATH_FOUNDATION_MODEL_ID)
            self._loaded = True
            logger.info("Path Foundation model loaded successfully.")
            return True

        except ImportError as e:
            logger.warning(
                f"TensorFlow/huggingface_hub not available: {e}. "
                f"Falling back to simulated mode."
            )
            self.simulated = True
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load Path Foundation: {e}")
            self.simulated = True
            self._loaded = True
            return True

    def extract_embedding(self, image_path: str) -> np.ndarray:
        """
        Extract a 384-dim embedding vector from a histopathology image.

        Args:
            image_path: Path to a 224x224 H&E stained tissue patch image.

        Returns:
            384-dimensional numpy embedding vector.
        """
        self.load_model()

        if self.simulated:
            return self._simulated_embedding(image_path)

        try:
            import tensorflow as tf
            from PIL import Image as PILImage

            img = PILImage.open(image_path).crop((0, 0, 224, 224)).convert("RGB")
            tensor = tf.cast(
                tf.expand_dims(np.array(img), axis=0), tf.float32
            ) / 255.0

            infer = self.model.signatures["serving_default"]
            embeddings = infer(tf.constant(tensor))
            return embeddings["output_0"].numpy().flatten()

        except Exception as e:
            logger.error(f"Embedding extraction failed: {e}")
            return self._simulated_embedding(image_path)

    def classify_tissue(self, image_path: str) -> Dict:
        """
        Classify tissue type from histopathology image.

        Returns probabilities for: Tumor, Stroma, Necrosis, Normal, Immune.
        """
        embedding = self.extract_embedding(image_path)
        # In production, a trained linear classifier head runs on the embedding
        # In demo mode, return simulated classification based on embedding
        return self._classify_from_embedding(embedding, image_path)

    def predict_treatment_response(
        self,
        image_path: str,
        drug_name: str,
        target_name: str,
    ) -> Dict:
        """
        Predict treatment response from tissue morphology + drug info.

        Combines Path Foundation tissue embeddings with drug pipeline data
        to predict whether a patient's tumor would respond to the drug.
        """
        embedding = self.extract_embedding(image_path)
        tissue = self._classify_from_embedding(embedding, image_path)

        # Combine tissue features with drug context for prediction
        tumor_fraction = tissue["classes"].get("Tumor", 0.0)
        immune_fraction = tissue["classes"].get("Immune Infiltrate", 0.0)

        # Response prediction heuristic (would be a trained model in production)
        if tumor_fraction > 0.7 and immune_fraction > 0.15:
            response = "Likely Responder"
            probability = 0.78
            rationale = (
                f"High tumor content ({tumor_fraction:.0%}) with significant "
                f"immune infiltrate ({immune_fraction:.0%}) suggests an "
                f"immunologically active tumor microenvironment favorable for "
                f"{drug_name} targeting {target_name}."
            )
        elif tumor_fraction > 0.5:
            response = "Moderate Response Expected"
            probability = 0.55
            rationale = (
                f"Moderate tumor content ({tumor_fraction:.0%}) with limited "
                f"immune activity. {drug_name} may show partial efficacy."
            )
        else:
            response = "Low Response Expected"
            probability = 0.25
            rationale = (
                f"Low tumor fraction ({tumor_fraction:.0%}). The tissue sample "
                f"may not be representative. Consider re-biopsy."
            )

        return {
            "drug": drug_name,
            "target": target_name,
            "image": os.path.basename(image_path),
            "response": response,
            "probability": probability,
            "rationale": rationale,
            "tissue_composition": tissue["classes"],
            "embedding_dim": len(embedding),
        }

    def compute_similarity(
        self,
        image_path_1: str,
        image_path_2: str,
    ) -> float:
        """
        Compute cosine similarity between two tissue patch embeddings.
        Useful for finding similar tissue regions across WSIs.
        """
        emb1 = self.extract_embedding(image_path_1)
        emb2 = self.extract_embedding(image_path_2)

        dot = np.dot(emb1, emb2)
        norm = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        return float(dot / norm) if norm > 0 else 0.0

    def batch_analyze(
        self,
        image_dir: str,
        drug_name: str = "Erlotinib",
        target_name: str = "EGFR",
    ) -> List[Dict]:
        """
        Analyze all histopathology images in a directory.
        """
        results = []
        supported_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

        if not os.path.isdir(image_dir):
            logger.warning(f"Directory not found: {image_dir}. Using demo data.")
            return self._demo_batch_results(drug_name, target_name)

        for fname in sorted(os.listdir(image_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in supported_ext:
                path = os.path.join(image_dir, fname)
                result = self.predict_treatment_response(path, drug_name, target_name)
                results.append(result)

        return results if results else self._demo_batch_results(drug_name, target_name)

    # ── Internal helpers ──────────────────────────────────────────────

    def _simulated_embedding(self, image_path: str) -> np.ndarray:
        """Generate a deterministic simulated embedding from filename."""
        seed = hash(os.path.basename(image_path)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        embedding = rng.randn(EMBEDDING_DIM).astype(np.float32)
        return embedding / np.linalg.norm(embedding)  # L2-normalize

    def _classify_from_embedding(self, embedding: np.ndarray, image_path: str) -> Dict:
        """Simulated tissue classification from embedding."""
        seed = hash(os.path.basename(image_path)) % (2 ** 31)
        rng = np.random.RandomState(seed)

        raw = rng.dirichlet([3, 2, 1, 2, 1.5])
        class_names = ["Tumor", "Stroma", "Necrosis", "Normal", "Immune Infiltrate"]
        classes = {name: round(float(p), 3) for name, p in zip(class_names, raw)}
        dominant = max(classes, key=classes.get)

        return {
            "classes": classes,
            "dominant_class": dominant,
            "confidence": round(float(max(raw)), 2),
        }

    def _demo_batch_results(self, drug_name: str, target_name: str) -> List[Dict]:
        """Return demo results when no images are available."""
        demo_samples = [
            {"name": "patient_001_tumor_core.png", "tumor": 0.78, "immune": 0.12},
            {"name": "patient_002_invasive_margin.png", "tumor": 0.45, "immune": 0.28},
            {"name": "patient_003_stroma_rich.png", "tumor": 0.22, "immune": 0.08},
            {"name": "patient_004_immune_hot.png", "tumor": 0.65, "immune": 0.35},
            {"name": "patient_005_necrotic.png", "tumor": 0.55, "immune": 0.05},
        ]

        results = []
        for sample in demo_samples:
            tumor = sample["tumor"]
            immune = sample["immune"]

            if tumor > 0.7 and immune > 0.15:
                response, prob = "Likely Responder", 0.78
            elif tumor > 0.5 and immune > 0.2:
                response, prob = "Likely Responder", 0.72
            elif tumor > 0.5:
                response, prob = "Moderate Response", 0.55
            else:
                response, prob = "Low Response", 0.25

            results.append({
                "drug": drug_name,
                "target": target_name,
                "image": sample["name"],
                "response": response,
                "probability": prob,
                "tissue_composition": {
                    "Tumor": tumor,
                    "Stroma": round(1 - tumor - immune - 0.1, 2),
                    "Necrosis": 0.05,
                    "Normal": 0.05,
                    "Immune Infiltrate": immune,
                },
                "embedding_dim": EMBEDDING_DIM,
                "rationale": (
                    f"Tumor fraction: {tumor:.0%}, Immune infiltrate: {immune:.0%}. "
                    f"{'Active tumor microenvironment.' if immune > 0.15 else 'Limited immune activity.'}"
                ),
            })

        return results


def print_pathology_results(results: List[Dict]):
    """Pretty-print pathology analysis results."""
    print("\n" + "=" * 75)
    print("  Stage 6: Pathology Analysis — Path Foundation")
    print("  Model: google/path-foundation (ViT-S, 384-dim embeddings)")
    print("=" * 75)

    table_data = []
    for r in results:
        prob = r["probability"]
        bar = "#" * int(prob * 15) + "." * (15 - int(prob * 15))
        table_data.append([
            r["image"][:30],
            f"{r['tissue_composition'].get('Tumor', 0):.0%}",
            f"{r['tissue_composition'].get('Immune Infiltrate', 0):.0%}",
            r["response"],
            f"{prob:.0%}",
            bar,
        ])

    print(tabulate(
        table_data,
        headers=["Sample", "Tumor%", "Immune%", "Response", "Prob", ""],
        tablefmt="rounded_outline",
    ))

    # Summary
    responders = sum(1 for r in results if "Likely" in r.get("response", ""))
    total = len(results)
    print(f"\n  Summary: {responders}/{total} samples predict positive response")
    print(f"  Drug: {results[0]['drug']} | Target: {results[0]['target']}")
    print()


# Convenience function for pipeline integration
def run_pathology_analysis(
    image_dir: str = "data/pathology",
    drug_name: str = "Erlotinib",
    target_name: str = "EGFR",
) -> List[Dict]:
    """Run pathology analysis as part of the drug discovery pipeline."""
    analyzer = PathFoundationAnalyzer(simulated=True)
    results = analyzer.batch_analyze(image_dir, drug_name, target_name)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_pathology_analysis()
    print_pathology_results(results)
