"""
Stage 3: Binding Affinity Prediction
Predicts drug-target interaction strength using TxGemma-Predict.
"""

import logging
from typing import Dict, List, Optional
from tabulate import tabulate

from .config import (
    DEFAULT_MODELS,
    PROMPT_TEMPLATES,
    SAMPLE_TARGETS,
)
from .model_loader import get_model_loader

logger = logging.getLogger("hai-def-pipeline")


def predict_binding(
    smiles: str,
    target_name: str,
    protein_seq: Optional[str] = None,
) -> Dict:
    """
    Predict binding affinity between a compound and a target.

    Args:
        smiles: SMILES string of the compound.
        target_name: Name/gene symbol of the protein target.
        protein_seq: Optional amino acid sequence fragment.

    Returns:
        Dict with binding prediction, affinity, and confidence.
    """
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["binding_affinity"]
    loader.load_model(model_id)

    # Get protein sequence if not provided
    if protein_seq is None:
        target_info = SAMPLE_TARGETS.get(target_name, {})
        protein_seq = target_info.get("sequence_fragment", "UNKNOWN")

    # --- Binary binding prediction ---
    dti_prompt = PROMPT_TEMPLATES["dti_binding"].format(
        smiles=smiles, protein_seq=protein_seq
    )
    dti_response = loader.predict(model_id, dti_prompt)

    # --- Binding affinity (Kd) ---
    affinity_prompt = PROMPT_TEMPLATES["binding_affinity"].format(
        smiles=smiles, target_name=target_name
    )
    affinity_response = loader.predict(model_id, affinity_prompt)

    # Parse results
    binds = _parse_yes_no(dti_response)
    kd = _parse_numeric(affinity_response)
    confidence = _parse_confidence(dti_response)

    return {
        "smiles": smiles,
        "target": target_name,
        "binds": binds,
        "kd_nM": kd,
        "confidence": confidence,
        "affinity_class": _classify_affinity(kd),
        "raw_dti": dti_response,
        "raw_affinity": affinity_response,
    }


def batch_binding_prediction(
    compounds: List[Dict],
    target_name: str,
) -> List[Dict]:
    """
    Run binding predictions for a batch of compounds against one target.

    Args:
        compounds: List of dicts with 'name' and 'smiles' keys.
        target_name: Target to test against.

    Returns:
        List of binding prediction results.
    """
    results = []
    for compound in compounds:
        pred = predict_binding(
            smiles=compound["smiles"],
            target_name=target_name,
        )
        pred["compound_name"] = compound.get("name", "Unknown")
        results.append(pred)

    # Sort by Kd (lower = better binding)
    results.sort(key=lambda x: x.get("kd_nM", 9999))
    return results


def _parse_yes_no(text: str) -> bool:
    """Parse a yes/no response."""
    text_lower = text.lower()
    if "yes" in text_lower:
        return True
    return False


def _parse_numeric(text: str) -> float:
    """Extract first number from text."""
    import re
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        return float(numbers[0])
    return 100.0  # Default moderate affinity


def _parse_confidence(text: str) -> float:
    """Extract confidence score from text."""
    import re
    # Look for patterns like "0.82" or "82%"
    patterns = [
        r"confidence[:\s]*(\d+\.?\d*)",
        r"(\d+\.?\d*)(?:\s*%)",
        r"(\d+\.\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            val = float(match.group(1))
            return val if val <= 1.0 else val / 100.0
    return 0.75


def _classify_affinity(kd_nM: float) -> str:
    """Classify binding affinity strength."""
    if kd_nM < 10:
        return "Very Strong"
    elif kd_nM < 100:
        return "Strong"
    elif kd_nM < 1000:
        return "Moderate"
    elif kd_nM < 10000:
        return "Weak"
    else:
        return "Very Weak"


def print_binding_results(results: List[Dict]):
    """Pretty-print binding prediction results."""
    print("\n" + "=" * 75)
    print("  Stage 3: Binding Affinity Prediction Results")
    print("=" * 75)

    table_data = []
    for r in results:
        table_data.append([
            r.get("compound_name", "N/A"),
            r["target"],
            "✅ Yes" if r["binds"] else "❌ No",
            f"{r['kd_nM']:.1f}",
            r["affinity_class"],
            f"{r['confidence']:.2f}",
        ])

    print(tabulate(
        table_data,
        headers=["Compound", "Target", "Binds?", "Kd (nM)", "Strength", "Confidence"],
        tablefmt="rounded_outline",
    ))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = predict_binding(
        smiles="COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        target_name="EGFR",
    )
    print_binding_results([{**result, "compound_name": "Erlotinib"}])
