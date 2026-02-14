"""
Stage 1: Target Identification
Identifies and characterizes protein targets for a given disease indication
using TxGemma-Chat for scientific reasoning.
"""

import logging
from typing import Dict, List
from tabulate import tabulate

from .config import DEFAULT_MODELS, SAMPLE_TARGETS
from .model_loader import get_model_loader

logger = logging.getLogger("hai-def-pipeline")


def identify_targets(disease: str) -> List[Dict]:
    """
    Identify potential drug targets for a given disease using TxGemma.

    Args:
        disease: Disease indication (e.g., "Non-Small Cell Lung Cancer")

    Returns:
        List of target dictionaries with name, rationale, druggability score.
    """
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["target_identification"]
    loader.load_model(model_id)

    prompt = (
        f"As a drug discovery expert, identify the top 3 protein targets "
        f"for treating {disease}. For each target, provide:\n"
        f"1. Target name and gene symbol\n"
        f"2. Biological rationale for targeting\n"
        f"3. Druggability assessment (High/Medium/Low)\n"
        f"4. Known approved drugs (if any)\n"
    )

    response = loader.predict(model_id, prompt)

    # Parse or use pre-defined targets if in simulated mode
    if loader.simulated_mode:
        return _get_simulated_targets(disease)

    return [{"raw_response": response}]


def characterize_target(target_name: str) -> Dict:
    """
    Generate a detailed characterization of a protein target.
    """
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["target_identification"]
    loader.load_model(model_id)

    target_info = SAMPLE_TARGETS.get(target_name, {})

    prompt = (
        f"Provide a detailed characterization of {target_name} "
        f"({target_info.get('name', '')}) as a drug target:\n"
        f"- Protein function and signaling pathway\n"
        f"- Role in disease pathology\n"
        f"- Binding site characteristics\n"
        f"- Resistance mechanisms\n"
        f"- Current therapeutic strategies\n"
    )

    response = loader.predict(model_id, prompt)
    return {
        "target": target_name,
        "full_name": target_info.get("name", target_name),
        "disease": target_info.get("disease", "Unknown"),
        "characterization": response,
    }


def _get_simulated_targets(disease: str) -> List[Dict]:
    """Return pre-curated targets for demonstration."""
    disease_lower = disease.lower()

    if "lung cancer" in disease_lower or "nsclc" in disease_lower:
        return [
            {
                "target": "EGFR",
                "full_name": "Epidermal Growth Factor Receptor",
                "druggability": "High",
                "rationale": "Overexpressed/mutated in 10-35% of NSCLC. Drives proliferation via RAS/MAPK pathway.",
                "known_drugs": ["Erlotinib", "Gefitinib", "Osimertinib"],
                "score": 0.95,
            },
            {
                "target": "ALK",
                "full_name": "Anaplastic Lymphoma Kinase",
                "druggability": "High",
                "rationale": "ALK rearrangements in ~5% of NSCLC. Constitutive kinase activation drives tumorigenesis.",
                "known_drugs": ["Crizotinib", "Alectinib", "Lorlatinib"],
                "score": 0.90,
            },
            {
                "target": "KRAS G12C",
                "full_name": "KRAS Proto-Oncogene (G12C Mutant)",
                "druggability": "Medium",
                "rationale": "KRAS G12C mutation in ~13% of NSCLC. Previously 'undruggable' — now targetable.",
                "known_drugs": ["Sotorasib", "Adagrasib"],
                "score": 0.82,
            },
        ]

    elif "leukemia" in disease_lower or "cml" in disease_lower:
        return [
            {
                "target": "BCR-ABL",
                "full_name": "BCR-ABL Fusion Kinase",
                "druggability": "High",
                "rationale": "Philadelphia chromosome translocation creates constitutively active tyrosine kinase.",
                "known_drugs": ["Imatinib", "Dasatinib", "Nilotinib"],
                "score": 0.98,
            },
        ]

    else:
        return [
            {
                "target": "Unknown",
                "full_name": f"Target for {disease}",
                "druggability": "To be assessed",
                "rationale": "Further analysis needed with TxGemma-Chat.",
                "known_drugs": [],
                "score": 0.0,
            },
        ]


def print_targets(targets: List[Dict]):
    """Pretty-print target identification results."""
    print("\n" + "=" * 65)
    print("  Stage 1: Target Identification Results")
    print("=" * 65)

    table_data = []
    for t in targets:
        table_data.append([
            t.get("target", "N/A"),
            t.get("full_name", "N/A")[:30],
            t.get("druggability", "N/A"),
            f"{t.get('score', 0):.2f}",
            ", ".join(t.get("known_drugs", [])[:2]) or "None",
        ])

    print(tabulate(
        table_data,
        headers=["Target", "Full Name", "Druggability", "Score", "Known Drugs"],
        tablefmt="rounded_outline",
    ))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    targets = identify_targets("Non-Small Cell Lung Cancer")
    print_targets(targets)
