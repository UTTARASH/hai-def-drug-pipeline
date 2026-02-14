"""
Interactive Gradio Demo for HAI-DEF Drug Discovery Pipeline
============================================================
Provides a web-based interface for judges and users to:
  - Select disease target and compound library
  - Run the full 11-stage pipeline interactively
  - View live results from each stage
  - Explore compound rankings and safety profiles
  - View 3D protein structure links from AlphaFold

Launch: python app.py → opens at http://localhost:7860
"""

import sys
import os
import time
import io
import contextlib
from typing import List, Dict

# Ensure pipeline is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np

from pipeline.config import DEFAULT_MODELS
from pipeline.target_identification import run_target_identification, format_target_results
from pipeline.lead_discovery import run_lead_discovery
from pipeline.binding_affinity import run_binding_affinity
from pipeline.admet_profiling import run_admet_profiling
from pipeline.clinical_reasoning import run_clinical_reasoning
from pipeline.pathology_analysis import run_pathology_analysis
from pipeline.medical_imaging import run_medsiglip_analysis
from pipeline.derm_analysis import run_derm_analysis
from pipeline.cxr_analysis import run_cxr_analysis
from pipeline.deepchem_analysis import run_deepchem_analysis
from pipeline.alphafold_analysis import run_alphafold_analysis
from pipeline.cross_stage_intelligence import CrossStageIntelligence
from pipeline.federated_learning import run_federated_pipeline, FederatedDrugDiscovery


# ─── Pipeline Runner ──────────────────────────────────────────────────

def run_full_pipeline(disease: str, target: str, enable_federated: bool = False, progress=gr.Progress()):
    """Run the complete 11-stage pipeline with progress updates."""

    results = {}
    log_lines = []

    def log(msg):
        log_lines.append(msg)

    # Optional: Federated Learning
    federated_output = "*Federated learning not enabled for this run.*"
    if enable_federated:
        progress(0.02, desc="Stage 0: Federated Learning")
        log("🏥 Stage 0: Federated Learning (5 Hospital Nodes)")
        fed_results = run_federated_pipeline(disease=disease, target=target)
        federated_output = format_federated(fed_results)
        results["federated"] = fed_results
        log(f"   → {fed_results['num_hospitals']} hospitals, "
            f"{fed_results['num_rounds']} rounds, "
            f"{len(fed_results['consensus_biomarkers'])} consensus biomarkers")

    # Stage 1
    progress(0.05, desc="Stage 1: Target Identification")
    log("🎯 Stage 1: Target Identification (TxGemma-27B Chat)")
    targets = run_target_identification(disease)
    target_text = format_target_results(targets)
    results["targets"] = targets
    log(f"   → Identified {len(targets)} targets")

    # Stage 2
    progress(0.15, desc="Stage 2: Lead Discovery")
    log("💊 Stage 2: Lead Discovery (TxGemma-2B + RDKit)")
    compounds = run_lead_discovery(target)
    results["compounds"] = compounds
    log(f"   → Screened {len(compounds)} compounds")

    # Stage 3
    progress(0.25, desc="Stage 3: Binding Affinity")
    log("🔬 Stage 3: Binding Affinity (TxGemma-9B)")
    binding = run_binding_affinity(compounds[:5], target)
    results["binding"] = binding
    log(f"   → {sum(1 for b in binding if b.get('binds'))} compounds show binding")

    # Stage 4
    progress(0.35, desc="Stage 4: ADMET Profiling")
    log("⚗️ Stage 4: ADMET Profiling (TxGemma-2B)")
    admet = run_admet_profiling(compounds[:5])
    results["admet"] = admet
    log(f"   → {len(admet)} ADMET profiles generated")

    # Stage 5
    progress(0.45, desc="Stage 5: Clinical Viability")
    log("🧪 Stage 5: Clinical Viability (TxGemma-27B Chat)")
    clinical = run_clinical_reasoning(compounds[:5], target, disease)
    results["clinical"] = clinical
    log(f"   → {len(clinical)} clinical assessments")

    # Stage 6
    progress(0.55, desc="Stage 6: Pathology Analysis")
    log("🔬 Stage 6: Pathology Analysis (Path Foundation)")
    pathology = run_pathology_analysis()
    results["pathology"] = pathology
    log(f"   → Analyzed {len(pathology)} tissue patches")

    # Stage 7
    progress(0.60, desc="Stage 7: Medical Imaging")
    log("🏥 Stage 7: Medical Imaging (MedSigLIP)")
    imaging = run_medsiglip_analysis()
    results["imaging"] = imaging
    log(f"   → Classified {len(imaging)} medical images")

    # Stage 8
    progress(0.68, desc="Stage 8: Dermatology")
    log("🪨 Stage 8: Dermatology (Derm Foundation)")
    derm = run_derm_analysis()
    results["derm"] = derm
    log(f"   → Assessed {len(derm)} skin samples")

    # Stage 9
    progress(0.75, desc="Stage 9: Chest X-Ray")
    log("🫁 Stage 9: Chest X-Ray (CXR Foundation)")
    cxr = run_cxr_analysis()
    results["cxr"] = cxr
    log(f"   → Analyzed {len(cxr)} chest X-rays")

    # Stage 10
    progress(0.83, desc="Stage 10: DeepChem GNN")
    log("⚗️ Stage 10: Molecular Properties (DeepChem GCN)")
    deepchem = run_deepchem_analysis(compounds=compounds[:5])
    results["deepchem"] = deepchem
    log(f"   → {len(deepchem)} GNN property predictions")

    # Stage 11
    progress(0.90, desc="Stage 11: AlphaFold")
    log("🧬 Stage 11: Protein Structure (AlphaFold)")
    alphafold = run_alphafold_analysis(target_name=target, disease=disease)
    results["alphafold"] = alphafold
    log(f"   → {len(alphafold)} protein structures analyzed")

    # Cross-Stage Intelligence
    progress(0.95, desc="Cross-Stage Intelligence")
    log("🧠 Cross-Stage Intelligence: Integrating all results...")
    csi = CrossStageIntelligence()
    rankings = csi.rank_compounds(results)
    report = csi.generate_report(results, disease, target)
    results["rankings"] = rankings

    progress(1.0, desc="Complete!")
    log(f"\n✅ Pipeline complete — {len(rankings)} compounds ranked")

    # Format outputs
    log_output = "\n".join(log_lines)
    ranking_output = format_rankings(rankings)
    safety_output = format_safety(results)
    structure_output = format_structures(alphafold)

    return log_output, ranking_output, safety_output, structure_output, report, federated_output


def format_rankings(rankings: List[Dict]) -> str:
    """Format compound rankings as markdown table."""
    lines = ["## 🏆 Compound Rankings (Cross-Stage Intelligence)\n"]
    lines.append("| Rank | Compound | Score | Binding | ADMET | Clinical | Verdict |")
    lines.append("|------|----------|-------|---------|-------|----------|---------|")
    for i, r in enumerate(rankings[:10], 1):
        verdict = "✅ Advance" if r["overall_score"] > 0.65 else "⚠️ Optimize" if r["overall_score"] > 0.4 else "❌ Reject"
        lines.append(
            f"| {i} | **{r['compound']}** | {r['overall_score']:.2f} | "
            f"{r.get('binding_score', 0):.2f} | {r.get('safety_score', 0):.2f} | "
            f"{r.get('clinical_score', 0):.2f} | {verdict} |"
        )
    lines.append(f"\n*Scoring integrates data from all 11 pipeline stages.*")
    return "\n".join(lines)


def format_safety(results: Dict) -> str:
    """Format safety profiles."""
    lines = ["## ⚗️ Safety Dashboard\n"]
    admet = results.get("admet", [])
    for a in admet[:5]:
        name = a.get("compound", "Unknown")
        flags = a.get("flags", 0) if isinstance(a.get("flags"), int) else 0
        status = "🟢 Safe" if flags == 0 else "🟡 Monitor" if flags <= 1 else "🔴 Concern"
        lines.append(f"**{name}**: {status} ({flags} safety flags)")
    return "\n".join(lines)


def format_structures(alphafold_results: List[Dict]) -> str:
    """Format AlphaFold structure results."""
    lines = ["## 🧬 Protein Structures (AlphaFold)\n"]
    for r in alphafold_results:
        plddt = r.get("mean_plddt", 0)
        quality = "High" if plddt > 80 else "Medium" if plddt > 60 else "Low"
        lines.append(f"**{r['target']}** ({r['protein_name']})")
        lines.append(f"- UniProt: `{r['uniprot_id']}` | pLDDT: {plddt:.0f} ({quality})")
        lines.append(f"- Binding pockets: {r.get('num_pockets', 0)}")
        lines.append(f"- Druggability: {r.get('druggability_score', 0):.2f}")
        if r.get("pdb_url"):
            lines.append(f"- [View 3D Structure]({r['pdb_url']})")
        lines.append("")
    return "\n".join(lines)


def format_federated(fed_results: Dict) -> str:
    """Format federated learning results."""
    lines = ["## 🏥 Federated Learning Results\n"]
    lines.append(f"**Hospitals**: {fed_results['num_hospitals']} | "
                 f"**Rounds**: {fed_results['num_rounds']} | "
                 f"**Privacy ε**: {fed_results['privacy_epsilon']}\n")

    lines.append("### Participating Hospitals\n")
    lines.append("| ID | Hospital | Location | Specialization | Patients |")
    lines.append("|-----|----------|----------|----------------|----------|")
    for h in fed_results["hospital_summary"]:
        lines.append(f"| {h['id']} | {h['name']} | {h['location']} | "
                     f"{h['specialization']} | {h['patients']:,} |")

    gm = fed_results["global_model"]
    lines.append(f"\n### Convergence\n")
    lines.append(f"- Final loss: **{gm['final_avg_loss']:.4f}**")
    lines.append(f"- Privacy budget remaining: **{fed_results['privacy_budget_remaining']:.2f}** / 10.0")

    lines.append("\n### Consensus Biomarkers\n")
    if fed_results["consensus_biomarkers"]:
        lines.append("| Biomarker | Type | Prevalence | Confidence | Hospitals |")
        lines.append("|-----------|------|------------|------------|-----------|")
        for b in fed_results["consensus_biomarkers"]:
            lines.append(f"| {b['name']} | {b['type']} | {b['avg_prevalence']:.1%} | "
                         f"{b['avg_confidence']:.1%} | {b['reporting_hospitals']} |")
    else:
        lines.append("*No consensus biomarkers found.*")

    strat = fed_results["global_stratification"]
    lines.append(f"\n### Patient Stratification\n")
    lines.append(f"- Total patients: **{strat['total_patients']:,}**")
    lines.append(f"- Responder rate: **{strat['responder_rate']:.1%}**")
    lines.append(f"- Non-responder rate: **{strat['non_responder_rate']:.1%}**")

    lines.append("\n---")
    lines.append("*🔒 All data processed locally. Only weight updates were shared.*")
    return "\n".join(lines)


# ─── Gradio UI ────────────────────────────────────────────────────────

DISEASES = [
    "Non-Small Cell Lung Cancer",
    "Chronic Myeloid Leukemia",
    "Hepatocellular Carcinoma",
    "Type 2 Diabetes",
    "Myelofibrosis",
    "COVID-19",
    "Asthma",
]

TARGETS = ["EGFR", "BCR-ABL", "VEGFR", "KRAS", "ALK", "JAK2", "MET", "BRAF", "AMPK"]

CSS = """
.gradio-container {
    max-width: 1200px !important;
    font-family: 'Inter', sans-serif;
}
.pipeline-header {
    text-align: center;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}
.pipeline-header h1 {
    font-size: 2rem;
    margin: 0;
}
.pipeline-header p {
    opacity: 0.85;
    margin-top: 0.5rem;
}
.stage-log {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
"""

with gr.Blocks(
    css=CSS,
    title="HAI-DEF Drug Discovery Pipeline",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    ),
) as demo:

    gr.HTML("""
    <div class="pipeline-header">
        <h1>💊 HAI-DEF Drug Discovery Pipeline</h1>
        <p>10 AI Models · 11 Stages · Cross-Stage Intelligence</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">
            TxGemma · MedGemma · Path Foundation · MedSigLIP · Derm Foundation · CXR Foundation · DeepChem · AlphaFold
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            disease_input = gr.Dropdown(
                choices=DISEASES,
                value="Non-Small Cell Lung Cancer",
                label="🦠 Disease",
                info="Select disease area for the pipeline",
            )
            target_input = gr.Dropdown(
                choices=TARGETS,
                value="EGFR",
                label="🎯 Protein Target",
                info="Primary drug target",
            )
            run_btn = gr.Button(
                "🚀 Run Full Pipeline",
                variant="primary",
                size="lg",
            )
            federated_checkbox = gr.Checkbox(
                label="🏥 Enable Federated Learning",
                value=False,
                info="Run privacy-preserving multi-hospital training before pipeline",
            )

        with gr.Column(scale=2):
            pipeline_log = gr.Textbox(
                label="📋 Pipeline Execution Log",
                lines=16,
                interactive=False,
                elem_classes=["stage-log"],
            )

    with gr.Tabs():
        with gr.TabItem("🏆 Rankings"):
            rankings_output = gr.Markdown(label="Compound Rankings")

        with gr.TabItem("⚗️ Safety"):
            safety_output = gr.Markdown(label="Safety Dashboard")

        with gr.TabItem("🧬 Structures"):
            structures_output = gr.Markdown(label="AlphaFold Structures")

        with gr.TabItem("🏥 Federated"):
            federated_output = gr.Markdown(label="Federated Learning")

        with gr.TabItem("📊 Full Report"):
            report_output = gr.Markdown(label="Cross-Stage Intelligence Report")

    run_btn.click(
        fn=run_full_pipeline,
        inputs=[disease_input, target_input, federated_checkbox],
        outputs=[pipeline_log, rankings_output, safety_output, structures_output, report_output, federated_output],
    )

    gr.Markdown("""
    ---
    **Models**: TxGemma-2B/9B/27B, MedGemma-4B, Path Foundation, MedSigLIP, Derm Foundation, CXR Foundation, DeepChem GCN, AlphaFold  
    **Repository**: [github.com/UTTARASH/hai-def-drug-pipeline](https://github.com/UTTARASH/hai-def-drug-pipeline)  
    *Research use only — not for clinical decisions.*
    """)


if __name__ == "__main__":
    demo.launch(share=False)
