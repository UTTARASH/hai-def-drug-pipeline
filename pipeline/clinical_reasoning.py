"""
Stage 5: Clinical Reasoning
Uses TxGemma-Chat (27B) for interactive clinical reasoning about drug candidates.
Evaluates clinical trial viability and generates scientific rationale.
"""

import logging
from typing import Dict, List
from tabulate import tabulate

from .config import DEFAULT_MODELS, PROMPT_TEMPLATES
from .model_loader import get_model_loader

logger = logging.getLogger("hai-def-pipeline")


def assess_clinical_viability(
    smiles: str,
    compound_name: str,
    target_name: str,
    disease: str,
    admet_profile: Dict = None,
    binding_result: Dict = None,
) -> Dict:
    """
    Comprehensive clinical viability assessment using TxGemma-Chat.

    Combines binding, ADMET, and molecular data to evaluate a candidate's
    probability of clinical success across Phases I–III.

    Args:
        smiles: SMILES string of the compound.
        compound_name: Human-readable compound name.
        target_name: Drug target gene symbol.
        disease: Disease indication.
        admet_profile: Optional ADMET profiling results from Stage 4.
        binding_result: Optional binding affinity results from Stage 3.

    Returns:
        Dict with phase predictions and overall recommendation.
    """
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["clinical_reasoning"]
    loader.load_model(model_id)

    # Phase I — Safety
    phase1 = _predict_phase(loader, model_id, smiles, target_name, disease, phase="I")

    # Phase II — Efficacy
    phase2 = _predict_phase(loader, model_id, smiles, target_name, disease, phase="II")

    # Phase III — Confirmatory
    phase3 = _predict_phase(loader, model_id, smiles, target_name, disease, phase="III")

    # Generate scientific rationale
    rationale = _generate_rationale(
        loader, model_id, compound_name, smiles, target_name,
        disease, admet_profile, binding_result
    )

    # Overall assessment
    avg_prob = (phase1["probability"] + phase2["probability"] + phase3["probability"]) / 3

    if avg_prob >= 0.7:
        recommendation = "🟢 Strong Candidate — Proceed to development"
    elif avg_prob >= 0.5:
        recommendation = "🟡 Moderate — Optimize before advancing"
    elif avg_prob >= 0.3:
        recommendation = "🟠 Weak — Consider structural modifications"
    else:
        recommendation = "🔴 Not Recommended — Significant risks identified"

    return {
        "compound_name": compound_name,
        "smiles": smiles,
        "target": target_name,
        "disease": disease,
        "phases": {
            "I":   phase1,
            "II":  phase2,
            "III": phase3,
        },
        "avg_probability": round(avg_prob, 2),
        "recommendation": recommendation,
        "rationale": rationale,
    }


def _predict_phase(loader, model_id, smiles, target_name, disease, phase) -> Dict:
    """Predict probability of passing a specific clinical trial phase."""
    prompt = PROMPT_TEMPLATES["clinical_approval"].format(
        smiles=smiles,
        target_name=target_name,
        disease=disease,
        phase=phase,
    )
    response = loader.predict(model_id, prompt)

    # Parse probability
    import re
    match = re.search(r"(\d+\.?\d*)", response)
    probability = float(match.group(1)) if match else 0.5
    if probability > 1.0:
        probability = probability / 100.0

    return {
        "phase": phase,
        "probability": round(min(probability, 1.0), 2),
        "raw_response": response,
    }


def _generate_rationale(
    loader, model_id, compound_name, smiles, target_name,
    disease, admet_profile, binding_result
) -> str:
    """Generate a comprehensive scientific rationale."""

    context_parts = [
        f"Compound: {compound_name} (SMILES: {smiles})",
        f"Target: {target_name}",
        f"Disease: {disease}",
    ]

    if binding_result:
        kd = binding_result.get("kd_nM", "Unknown")
        context_parts.append(f"Binding affinity Kd: {kd} nM")

    if admet_profile:
        overall = admet_profile.get("overall", "Unknown")
        flags = admet_profile.get("flags", [])
        context_parts.append(f"ADMET: {overall}, Flags: {', '.join(flags) or 'None'}")

    context = "\n".join(context_parts)

    prompt = (
        f"As a clinical pharmacologist, provide a brief scientific rationale "
        f"for the following drug candidate:\n\n{context}\n\n"
        f"Address: mechanism of action, competitive landscape, risk factors, "
        f"and recommendation."
    )

    return loader.predict(model_id, prompt)


def print_clinical_assessment(assessment: Dict):
    """Pretty-print clinical viability assessment."""
    print("\n" + "=" * 70)
    print("  Stage 5: Clinical Viability Assessment")
    print("=" * 70)
    print(f"  Compound: {assessment['compound_name']}")
    print(f"  Target:   {assessment['target']} → {assessment['disease']}")
    print(f"  Overall:  {assessment['recommendation']}")
    print(f"{'─' * 70}")

    # Phase table
    table_data = []
    for phase_key in ["I", "II", "III"]:
        phase = assessment["phases"][phase_key]
        prob = phase["probability"]
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        table_data.append([
            f"Phase {phase_key}",
            f"{prob:.0%}",
            bar,
        ])

    print(tabulate(
        table_data,
        headers=["Phase", "Probability", "Confidence"],
        tablefmt="rounded_outline",
    ))

    print(f"\n  📋 Rationale:")
    for line in assessment["rationale"].split("\n"):
        print(f"     {line}")
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = assess_clinical_viability(
        smiles="COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        compound_name="Erlotinib",
        target_name="EGFR",
        disease="Non-Small Cell Lung Cancer",
    )
    print_clinical_assessment(result)
