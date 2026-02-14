"""
HAI-DEF Drug Discovery Pipeline — Main Orchestrator
Runs all five stages of the pipeline end-to-end.

Usage:
    python -m pipeline.main
    python -m pipeline.main --disease "Non-Small Cell Lung Cancer" --target EGFR
"""

import argparse
import logging
import sys
import time
import io
import os

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from .target_identification import identify_targets, print_targets
from .lead_discovery import screen_compounds, print_screening_results
from .binding_affinity import batch_binding_prediction, print_binding_results
from .admet_profiling import batch_admet_profiling, print_batch_summary, print_admet_profile
from .clinical_reasoning import assess_clinical_viability, print_clinical_assessment
from .visualization import print_final_report, create_pipeline_summary_chart
from .pathology_analysis import run_pathology_analysis, print_pathology_results
from .medical_imaging import run_medsiglip_analysis, print_medsiglip_results
from .derm_analysis import run_derm_analysis, print_derm_results
from .cxr_analysis import run_cxr_analysis, print_cxr_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hai-def-pipeline")


def run_pipeline(disease: str = "Non-Small Cell Lung Cancer", target: str = "EGFR"):
    """Execute the full drug discovery pipeline."""

    print("\n")
    print("╔" + "═" * 60 + "╗")
    print("║" + "  HAI-DEF Drug Discovery Pipeline".center(60) + "║")
    print("║" + "  TxGemma + MedGemma + Path Foundation".center(60) + "║")
    print("╚" + "═" * 60 + "╝")
    print(f"\n  Disease: {disease}")
    print(f"  Target:  {target}")
    print(f"  Time:    {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    # ──────────────────────────────────────────────────────────
    # Stage 1: Target Identification
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🎯 Stage 1: Target Identification")
    print("━" * 60)
    targets = identify_targets(disease)
    print_targets(targets)

    # ──────────────────────────────────────────────────────────
    # Stage 2: Lead Discovery (Compound Screening)
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  💊 Stage 2: Lead Discovery")
    print("━" * 60)
    screening_df = screen_compounds(target_filter=target)
    print_screening_results(screening_df)

    # Prepare compounds for downstream stages
    compounds = []
    for _, row in screening_df.iterrows():
        compounds.append({
            "name": row["compound_name"],
            "smiles": row["smiles"],
        })

    if not compounds:
        print("  ⚠️  No compounds found for target. Using full dataset.")
        screening_df = screen_compounds()
        print_screening_results(screening_df)
        for _, row in screening_df.head(5).iterrows():
            compounds.append({
                "name": row["compound_name"],
                "smiles": row["smiles"],
            })

    # ──────────────────────────────────────────────────────────
    # Stage 3: Binding Affinity Prediction
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🔬 Stage 3: Binding Affinity Prediction")
    print("━" * 60)
    binding_results = batch_binding_prediction(compounds, target)
    print_binding_results(binding_results)

    # ──────────────────────────────────────────────────────────
    # Stage 4: ADMET Profiling
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  ⚗️ Stage 4: ADMET Profiling")
    print("━" * 60)
    admet_profiles = batch_admet_profiling(compounds)
    print_batch_summary(admet_profiles)

    for profile in admet_profiles[:3]:  # Show detailed view for top 3
        print_admet_profile(profile)

    # ──────────────────────────────────────────────────────────
    # Stage 5: Clinical Reasoning
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🧪 Stage 5: Clinical Viability Assessment")
    print("━" * 60)

    clinical_assessments = []
    for i, compound in enumerate(compounds[:5]):
        admet = admet_profiles[i] if i < len(admet_profiles) else None
        binding = binding_results[i] if i < len(binding_results) else None

        assessment = assess_clinical_viability(
            smiles=compound["smiles"],
            compound_name=compound["name"],
            target_name=target,
            disease=disease,
            admet_profile=admet,
            binding_result=binding,
        )
        clinical_assessments.append(assessment)
        print_clinical_assessment(assessment)

    # ──────────────────────────────────────────────────────────
    # Stage 6: Pathology Analysis (Path Foundation)
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🔬 Stage 6: Pathology Analysis (Path Foundation)")
    print("━" * 60)
    pathology_results = run_pathology_analysis(
        image_dir="data/pathology",
        drug_name=compounds[0]["name"] if compounds else "Unknown",
        target_name=target,
    )
    print_pathology_results(pathology_results)

    # ──────────────────────────────────────────────────────────
    # Stage 7: Medical Image Analysis (MedSigLIP)
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🏥 Stage 7: Medical Image Analysis (MedSigLIP)")
    print("━" * 60)
    medsiglip_results = run_medsiglip_analysis(
        drug_name=compounds[0]["name"] if compounds else "Unknown",
        target_name=target,
        disease=disease,
    )
    print_medsiglip_results(medsiglip_results)

    # ──────────────────────────────────────────────────────────
    # Stage 8: Dermatology Analysis (Derm Foundation)
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🪨 Stage 8: Dermatology Analysis (Derm Foundation)")
    print("━" * 60)
    derm_results = run_derm_analysis(
        drug_name=compounds[0]["name"] if compounds else "Unknown",
        target_name=target,
    )
    print_derm_results(derm_results)

    # ──────────────────────────────────────────────────────────
    # Stage 9: Chest X-Ray Analysis (CXR Foundation)
    # ──────────────────────────────────────────────────────────
    print("━" * 60)
    print("  🏥 Stage 9: Chest X-Ray Analysis (CXR Foundation)")
    print("━" * 60)
    cxr_results = run_cxr_analysis(
        drug_name=compounds[0]["name"] if compounds else "Unknown",
        target_name=target,
        disease=disease,
    )
    print_cxr_results(cxr_results)

    # ──────────────────────────────────────────────────────────
    # Final Report
    # ──────────────────────────────────────────────────────────
    print_final_report(
        targets=targets,
        screening_results=screening_df,
        binding_results=binding_results,
        admet_profiles=admet_profiles,
        clinical_assessments=clinical_assessments,
    )

    # Generate chart
    chart_data = []
    for i, c in enumerate(clinical_assessments):
        chart_data.append({
            "compound_name": c["compound_name"],
            "binding_score": binding_results[i].get("confidence", 0.5) if i < len(binding_results) else 0.5,
            "admet_score": 1.0 - len(admet_profiles[i].get("flags", [])) * 0.2 if i < len(admet_profiles) else 0.5,
            "clinical_score": c.get("avg_probability", 0.5),
        })
    create_pipeline_summary_chart(chart_data)

    elapsed = time.time() - start_time
    print(f"\n  ⏱️  Pipeline completed in {elapsed:.1f} seconds")
    print(f"  📊 Chart saved to output/pipeline_summary.png")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="HAI-DEF Drug Discovery Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m pipeline.main
  python -m pipeline.main --disease "CML" --target "BCR-ABL"
  python -m pipeline.main --disease "NSCLC" --target "EGFR"
        """,
    )
    parser.add_argument(
        "--disease", default="Non-Small Cell Lung Cancer",
        help="Disease indication (default: NSCLC)"
    )
    parser.add_argument(
        "--target", default="EGFR",
        help="Drug target gene symbol (default: EGFR)"
    )

    args = parser.parse_args()
    run_pipeline(disease=args.disease, target=args.target)


if __name__ == "__main__":
    main()
