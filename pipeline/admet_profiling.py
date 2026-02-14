"""
Stage 4: ADMET Profiling
Predicts Absorption, Distribution, Metabolism, Excretion, and Toxicity
properties using TxGemma-Predict for each drug candidate.
"""

import logging
from typing import Dict, List
from tabulate import tabulate

from .config import DEFAULT_MODELS, PROMPT_TEMPLATES, ADMET_THRESHOLDS
from .model_loader import get_model_loader

logger = logging.getLogger("hai-def-pipeline")

ADMET_PROPERTIES = [
    {"key": "solubility",     "name": "Solubility",       "category": "Absorption",    "prompt": "admet_solubility"},
    {"key": "lipophilicity",  "name": "Lipophilicity",    "category": "Absorption",    "prompt": "admet_lipophilicity"},
    {"key": "bbb",            "name": "BBB Penetration",  "category": "Distribution",  "prompt": "admet_bbb"},
    {"key": "cyp3a4",         "name": "CYP3A4 Inhibition","category": "Metabolism",     "prompt": "admet_cyp"},
    {"key": "herg",           "name": "hERG Blocking",    "category": "Toxicity",      "prompt": "admet_herg"},
]


def profile_compound(smiles: str, compound_name: str = "Unknown") -> Dict:
    """
    Run full ADMET profiling for a single compound.

    Args:
        smiles: SMILES string of the compound.
        compound_name: Human-readable compound name.

    Returns:
        Dict with ADMET predictions and overall assessment.
    """
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["admet_profiling"]
    loader.load_model(model_id)

    results = {
        "compound_name": compound_name,
        "smiles": smiles,
        "properties": {},
        "flags": [],
    }

    for prop in ADMET_PROPERTIES:
        prompt = PROMPT_TEMPLATES[prop["prompt"]].format(smiles=smiles)
        response = loader.predict(model_id, prompt)

        assessment = _assess_property(prop["key"], response)

        results["properties"][prop["key"]] = {
            "name": prop["name"],
            "category": prop["category"],
            "prediction": response,
            "status": assessment["status"],
            "value": assessment["value"],
        }

        if assessment["status"] == "⚠️ Flag":
            results["flags"].append(prop["name"])

    # Overall ADMET verdict
    n_flags = len(results["flags"])
    if n_flags == 0:
        results["overall"] = "✅ Pass"
    elif n_flags <= 2:
        results["overall"] = "⚠️ Review"
    else:
        results["overall"] = "❌ Fail"

    return results


def batch_admet_profiling(compounds: List[Dict]) -> List[Dict]:
    """
    Profile a batch of compounds.

    Args:
        compounds: List of dicts with 'name' and 'smiles'.

    Returns:
        List of ADMET profile results.
    """
    results = []
    for compound in compounds:
        profile = profile_compound(
            smiles=compound["smiles"],
            compound_name=compound.get("name", "Unknown"),
        )
        results.append(profile)
    return results


def _assess_property(key: str, response: str) -> Dict:
    """Assess whether a property value is acceptable."""
    response_lower = response.lower()

    if key == "solubility":
        if "high" in response_lower:
            return {"status": "✅ Good", "value": "High"}
        elif "medium" in response_lower:
            return {"status": "✅ Good", "value": "Medium"}
        else:
            return {"status": "⚠️ Flag", "value": "Low"}

    elif key == "lipophilicity":
        # Extract logP value
        import re
        match = re.search(r"(-?\d+\.?\d*)", response)
        if match:
            logp = float(match.group(1))
            ideal = ADMET_THRESHOLDS["lipophilicity"]
            if ideal["ideal_min"] <= logp <= ideal["ideal_max"]:
                return {"status": "✅ Good", "value": f"logP={logp:.1f}"}
            else:
                return {"status": "⚠️ Flag", "value": f"logP={logp:.1f}"}
        return {"status": "✅ Good", "value": "Within range"}

    elif key == "bbb":
        if "yes" in response_lower:
            return {"status": "ℹ️ Info", "value": "Yes (penetrates)"}
        else:
            return {"status": "ℹ️ Info", "value": "No (does not penetrate)"}

    elif key == "cyp3a4":
        if "yes" in response_lower:
            return {"status": "⚠️ Flag", "value": "Inhibitor"}
        else:
            return {"status": "✅ Good", "value": "Non-inhibitor"}

    elif key == "herg":
        if "yes" in response_lower:
            return {"status": "⚠️ Flag", "value": "Blocker (risk)"}
        else:
            return {"status": "✅ Good", "value": "Non-blocker"}

    return {"status": "✅ Good", "value": "OK"}


def print_admet_profile(profile: Dict):
    """Pretty-print a single compound's ADMET profile."""
    print(f"\n{'─' * 60}")
    print(f"  ADMET Profile: {profile['compound_name']}")
    print(f"  SMILES: {profile['smiles'][:50]}...")
    print(f"  Overall: {profile['overall']}")
    print(f"{'─' * 60}")

    table_data = []
    for prop in ADMET_PROPERTIES:
        p = profile["properties"][prop["key"]]
        table_data.append([
            p["category"],
            p["name"],
            p["value"],
            p["status"],
        ])

    print(tabulate(
        table_data,
        headers=["Category", "Property", "Value", "Status"],
        tablefmt="rounded_outline",
    ))

    if profile["flags"]:
        print(f"\n  ⚠️  Flagged: {', '.join(profile['flags'])}")
    print()


def print_batch_summary(profiles: List[Dict]):
    """Print summary table for batch profiling."""
    print("\n" + "=" * 70)
    print("  Stage 4: ADMET Profiling Summary")
    print("=" * 70)

    table_data = []
    for p in profiles:
        table_data.append([
            p["compound_name"],
            p["overall"],
            len(p["flags"]),
            ", ".join(p["flags"][:3]) if p["flags"] else "None",
        ])

    print(tabulate(
        table_data,
        headers=["Compound", "Verdict", "# Flags", "Flagged Properties"],
        tablefmt="rounded_outline",
    ))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    profile = profile_compound(
        smiles="COc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
        compound_name="Erlotinib",
    )
    print_admet_profile(profile)
