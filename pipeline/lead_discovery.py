"""
Stage 2: Lead Discovery
Screens compound libraries and scores drug candidates using TxGemma-Predict.
Computes molecular descriptors via RDKit and predicts properties via TxGemma.
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
from tabulate import tabulate

from .config import (
    DEFAULT_MODELS,
    PROMPT_TEMPLATES,
    SAMPLE_COMPOUNDS_FILE,
)
from .model_loader import get_model_loader

logger = logging.getLogger("hai-def-pipeline")


def compute_molecular_descriptors(smiles: str) -> Dict:
    """
    Compute molecular descriptors using RDKit.
    Returns a dict with MW, LogP, HBD, HBA, TPSA, rotatable bonds.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "Invalid SMILES"}

        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logP": round(Descriptors.MolLogP(mol), 2),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
            "tpsa": round(Descriptors.TPSA(mol), 2),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "num_rings": rdMolDescriptors.CalcNumRings(mol),
            "lipinski_violations": _count_lipinski_violations(mol),
        }

    except ImportError:
        logger.warning("RDKit not available — using estimated descriptors.")
        return _estimate_descriptors(smiles)


def _count_lipinski_violations(mol) -> int:
    """Count Lipinski's Rule of Five violations."""
    from rdkit.Chem import Descriptors, rdMolDescriptors

    violations = 0
    if Descriptors.MolWt(mol) > 500:
        violations += 1
    if Descriptors.MolLogP(mol) > 5:
        violations += 1
    if rdMolDescriptors.CalcNumHBD(mol) > 5:
        violations += 1
    if rdMolDescriptors.CalcNumHBA(mol) > 10:
        violations += 1
    return violations


def _estimate_descriptors(smiles: str) -> Dict:
    """Rough estimation when RDKit is unavailable."""
    length = len(smiles)
    return {
        "molecular_weight": round(length * 8.5, 2),  # very rough estimate
        "logP": round(length * 0.05, 2),
        "hbd": smiles.count("O") + smiles.count("N"),
        "hba": smiles.count("O") + smiles.count("N") + smiles.count("F"),
        "tpsa": round(length * 2.0, 2),
        "rotatable_bonds": smiles.count("C") // 3,
        "num_rings": smiles.count("c1") + smiles.count("C1"),
        "lipinski_violations": 0,
        "estimated": True,
    }


def screen_compounds(
    csv_path: Optional[str] = None,
    target_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Screen a library of compounds. Computes descriptors, applies Lipinski
    filter, and scores with TxGemma.

    Args:
        csv_path: Path to compounds CSV (default: sample_compounds.csv)
        target_filter: Filter compounds by target name (optional)

    Returns:
        DataFrame with screening results.
    """
    if csv_path is None:
        csv_path = SAMPLE_COMPOUNDS_FILE

    # Resolve relative path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), csv_path)

    df = pd.read_csv(csv_path)

    if target_filter:
        df = df[df["Target"].str.contains(target_filter, case=False, na=False)]

    # Compute descriptors for each compound
    descriptors_list = []
    for _, row in df.iterrows():
        desc = compute_molecular_descriptors(row["SMILES"])
        desc["compound_name"] = row["Compound_Name"]
        desc["smiles"] = row["SMILES"]
        desc["target"] = row["Target"]
        desc["disease_area"] = row.get("Disease_Area", "Unknown")
        desc["source"] = row.get("Source", "Unknown")
        descriptors_list.append(desc)

    results_df = pd.DataFrame(descriptors_list)

    # Score with TxGemma
    results_df["txgemma_score"] = results_df.apply(
        lambda r: _score_compound(r["smiles"]), axis=1
    )

    # Drug-likeness score (Lipinski-based)
    results_df["drug_likeness"] = results_df["lipinski_violations"].apply(
        lambda v: "✅ Pass" if v <= 1 else "⚠️ Review" if v == 2 else "❌ Fail"
    )

    # Sort by score
    results_df = results_df.sort_values("txgemma_score", ascending=False)

    return results_df


def _score_compound(smiles: str) -> float:
    """Score a compound using TxGemma-Predict."""
    loader = get_model_loader()
    model_id = DEFAULT_MODELS["lead_discovery"]
    loader.load_model(model_id)

    prompt = PROMPT_TEMPLATES["molecule_description"].format(smiles=smiles)
    response = loader.predict(model_id, prompt)

    # In real mode, parse TxGemma's response for a score.
    # In simulated mode, use descriptor-based heuristic.
    if loader.simulated_mode:
        desc = compute_molecular_descriptors(smiles)
        mw = desc.get("molecular_weight", 300)
        logp = desc.get("logP", 2.5)
        violations = desc.get("lipinski_violations", 0)

        score = 1.0
        # Penalize for Lipinski violations
        score -= violations * 0.15
        # Penalize for extreme MW
        if mw > 500:
            score -= 0.1
        if mw < 150:
            score -= 0.1
        # Penalize for extreme logP
        if logp > 5 or logp < -1:
            score -= 0.15

        return round(max(0.0, min(1.0, score)), 2)

    # Parse actual model response for numeric score
    try:
        import re
        numbers = re.findall(r"(\d+\.?\d*)", response)
        if numbers:
            return round(min(float(numbers[0]) / 100, 1.0), 2)
    except Exception:
        pass

    return 0.5


def print_screening_results(df: pd.DataFrame):
    """Pretty-print compound screening results."""
    print("\n" + "=" * 80)
    print("  Stage 2: Lead Discovery — Compound Screening Results")
    print("=" * 80)

    cols = ["compound_name", "target", "molecular_weight", "logP",
            "lipinski_violations", "drug_likeness", "txgemma_score"]
    display_df = df[cols].copy()
    display_df.columns = ["Compound", "Target", "MW", "LogP",
                          "Lipinski Viol.", "Drug-like", "Score"]

    print(tabulate(display_df, headers="keys", tablefmt="rounded_outline",
                   showindex=False, floatfmt=".2f"))
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = screen_compounds()
    print_screening_results(results)
