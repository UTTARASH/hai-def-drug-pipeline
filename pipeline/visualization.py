"""
Visualization Module
Generates charts and visual summaries for the drug discovery pipeline.
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("hai-def-pipeline")


def create_pipeline_summary_chart(
    candidates: List[Dict],
    output_path: str = "output/pipeline_summary.png",
):
    """
    Create a summary bar chart comparing candidates across metrics.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        names = [c.get("compound_name", f"Comp {i}") for i, c in enumerate(candidates)]
        binding_scores = [c.get("binding_score", 0.5) for c in candidates]
        admet_scores = [c.get("admet_score", 0.5) for c in candidates]
        clinical_scores = [c.get("clinical_score", 0.5) for c in candidates]

        x = np.arange(len(names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#111827")

        bars1 = ax.bar(x - width, binding_scores, width, label="Binding Affinity",
                       color="#06b6d4", alpha=0.9, edgecolor="none")
        bars2 = ax.bar(x, admet_scores, width, label="ADMET Score",
                       color="#8b5cf6", alpha=0.9, edgecolor="none")
        bars3 = ax.bar(x + width, clinical_scores, width, label="Clinical Viability",
                       color="#10b981", alpha=0.9, edgecolor="none")

        ax.set_xlabel("Drug Candidates", color="#94a3b8", fontsize=12)
        ax.set_ylabel("Score", color="#94a3b8", fontsize=12)
        ax.set_title("HAI-DEF Drug Discovery Pipeline — Candidate Comparison",
                     color="#f1f5f9", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(names, color="#94a3b8", rotation=30, ha="right")
        ax.tick_params(axis="y", colors="#94a3b8")
        ax.set_ylim(0, 1.1)
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#f1f5f9")

        for spine in ax.spines.values():
            spine.set_color("#334155")

        ax.grid(axis="y", alpha=0.1, color="#94a3b8")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

        logger.info(f"📊 Chart saved to {output_path}")
        return output_path

    except ImportError:
        logger.warning("matplotlib not available — skipping chart generation.")
        return None


def create_admet_radar_chart(
    profile: Dict,
    output_path: str = "output/admet_radar.png",
):
    """
    Create a radar/spider chart for ADMET properties.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        properties = list(profile.get("properties", {}).keys())
        values = []
        for key in properties:
            prop = profile["properties"][key]
            status = prop.get("status", "")
            if "Good" in status:
                values.append(0.9)
            elif "Flag" in status:
                values.append(0.3)
            else:
                values.append(0.6)

        N = len(properties)
        if N < 3:
            logger.warning("Need at least 3 properties for radar chart.")
            return None

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values += values[:1]  # Complete the polygon
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#111827")

        ax.plot(angles, values, "o-", linewidth=2, color="#06b6d4", markersize=8)
        ax.fill(angles, values, alpha=0.15, color="#06b6d4")

        labels = [profile["properties"][k]["name"] for k in properties]
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, color="#94a3b8", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["Poor", "Fair", "Good", "Excellent"],
                           color="#64748b", fontsize=8)
        ax.tick_params(axis="y", colors="#64748b")

        ax.set_title(
            f"ADMET Profile: {profile.get('compound_name', 'Unknown')}",
            color="#f1f5f9", fontsize=14, fontweight="bold", pad=20,
        )

        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(color="#334155", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

        logger.info(f"📊 Radar chart saved to {output_path}")
        return output_path

    except ImportError:
        logger.warning("matplotlib not available — skipping chart generation.")
        return None


def print_final_report(
    targets: List[Dict],
    screening_results,
    binding_results: List[Dict],
    admet_profiles: List[Dict],
    clinical_assessments: List[Dict],
):
    """Print a final consolidated pipeline report to the terminal."""

    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  HAI-DEF Drug Discovery Pipeline — Final Report".center(68) + "║")
    print("║" + "  Powered by Google Health AI Developer Foundations".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    # Top candidates
    print("\n🏆 Top Candidates (Ranked by Overall Score)")
    print("─" * 60)

    ranked = sorted(clinical_assessments,
                    key=lambda x: x.get("avg_probability", 0),
                    reverse=True)

    for i, c in enumerate(ranked[:5], 1):
        prob = c.get("avg_probability", 0)
        bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
        print(f"  #{i} {c['compound_name']:<15} {bar} {prob:.0%}")
        print(f"     Target: {c['target']} | {c['recommendation']}")
        print()

    print("─" * 60)
    print("  ⚠️  For research purposes only. Not for clinical use.")
    print("  📦 Models: TxGemma (2B/9B/27B) + MedGemma 4B")
    print("  🔗 HAI-DEF: https://health.google/developers/")
    print("─" * 60)
