"""
Federated Drug Discovery — Privacy-Preserving Multi-Hospital Pipeline
======================================================================
Implements the federated learning workflow described in the HAI-DEF write-up:

  1. Hospital nodes train MedGemma + MedSigLIP locally on private patient data
  2. A federated server aggregates model updates (not raw data)
  3. The global model produces cross-institutional biomarkers and insights
  4. Results feed into the existing 11-stage drug discovery pipeline

All operations run in simulated/demo mode (no real patient data or GPU required).
This module is designed to demonstrate the architecture and data flow of a
federated drug discovery system using HAI-DEF models.

Usage:
    from pipeline.federated_learning import run_federated_pipeline
    results = run_federated_pipeline(disease="NSCLC", target="EGFR")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("hai-def-pipeline")


# ═══════════════════════════════════════════════════════════════════════
# Simulated Hospital Profiles
# ═══════════════════════════════════════════════════════════════════════

HOSPITAL_PROFILES = [
    {
        "id": "H-001",
        "name": "Memorial Oncology Center",
        "location": "Boston, USA",
        "specialization": "Oncology",
        "patient_count": 12_500,
        "data_types": ["clinical_notes", "histopathology", "ct_scans"],
        "compute": "NVIDIA A100 × 4",
    },
    {
        "id": "H-002",
        "name": "Cardiology Institute of Mumbai",
        "location": "Mumbai, India",
        "specialization": "Cardiology",
        "patient_count": 8_200,
        "data_types": ["clinical_notes", "ecg", "chest_xray"],
        "compute": "NVIDIA T4 × 2",
    },
    {
        "id": "H-003",
        "name": "Berlin University Hospital",
        "location": "Berlin, Germany",
        "specialization": "Pulmonology",
        "patient_count": 6_800,
        "data_types": ["clinical_notes", "chest_xray", "ct_scans"],
        "compute": "NVIDIA V100 × 2",
    },
    {
        "id": "H-004",
        "name": "Tokyo Medical Research Center",
        "location": "Tokyo, Japan",
        "specialization": "Rare Diseases",
        "patient_count": 4_100,
        "data_types": ["clinical_notes", "histopathology", "genomics"],
        "compute": "NVIDIA A100 × 2",
    },
    {
        "id": "H-005",
        "name": "São Paulo Clinical Research Hub",
        "location": "São Paulo, Brazil",
        "specialization": "Infectious Disease",
        "patient_count": 9_600,
        "data_types": ["clinical_notes", "dermatology", "chest_xray"],
        "compute": "NVIDIA T4 × 4",
    },
]


# ═══════════════════════════════════════════════════════════════════════
# Hospital Node — Local Training Simulation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class HospitalNode:
    """
    Simulates a hospital participating in federated learning.

    Each hospital trains MedGemma (text) and MedSigLIP (image) on its
    local patient data. Only model weight updates are shared — raw
    patient data never leaves the institution.
    """
    hospital_id: str
    name: str
    location: str
    specialization: str
    patient_count: int
    data_types: List[str]
    compute: str

    # Internal state
    _local_loss: float = field(default=2.5, init=False, repr=False)
    _training_rounds: int = field(default=0, init=False, repr=False)
    _rng: np.random.Generator = field(default=None, init=False, repr=False)

    def __post_init__(self):
        # Deterministic RNG per hospital for reproducible simulations
        seed = int.from_bytes(self.hospital_id.encode(), "little") % (2**31)
        self._rng = np.random.default_rng(seed)

    def local_train(self, global_weights: Optional[np.ndarray] = None) -> Dict:
        """
        Simulate local training on private patient data.

        Returns model weight updates (deltas), NOT raw data.
        This is the core privacy guarantee of federated learning.
        """
        self._training_rounds += 1

        # Simulate training loss convergence
        noise = self._rng.normal(0, 0.05)
        decay = 0.85 ** self._training_rounds
        self._local_loss = max(0.15, 2.5 * decay + noise)

        # Simulate weight updates (deltas)
        # In a real system, these would be gradient-based model updates
        weight_dim = 384  # MedSigLIP embedding dimension
        weight_delta = self._rng.normal(0, 0.01 * decay, size=weight_dim)

        if global_weights is not None:
            # Incorporate global model — simulates starting from aggregated weights
            weight_delta += self._rng.normal(0, 0.005, size=weight_dim)

        # Simulate discovered biomarkers from local patient data
        biomarkers = self._discover_biomarkers()

        # Simulate patient stratification patterns
        stratification = self._stratify_patients()

        return {
            "hospital_id": self.hospital_id,
            "hospital_name": self.name,
            "round": self._training_rounds,
            "weight_delta": weight_delta,
            "local_loss": round(self._local_loss, 4),
            "patients_used": min(self.patient_count, 500 * self._training_rounds),
            "biomarkers": biomarkers,
            "stratification": stratification,
            "privacy_budget_used": round(0.1 * self._training_rounds, 2),
            "data_types_used": self.data_types,
        }

    def _discover_biomarkers(self) -> List[Dict]:
        """Simulate biomarker discovery from local patient data."""
        markers = [
            {"name": "PD-L1 Expression", "prevalence": 0.35, "type": "protein"},
            {"name": "EGFR T790M Mutation", "prevalence": 0.18, "type": "genomic"},
            {"name": "ALK Rearrangement", "prevalence": 0.05, "type": "genomic"},
            {"name": "KRAS G12C", "prevalence": 0.13, "type": "genomic"},
            {"name": "TP53 Mutation", "prevalence": 0.50, "type": "genomic"},
            {"name": "Tumor Mutational Burden", "prevalence": 0.22, "type": "composite"},
            {"name": "CRP Elevation", "prevalence": 0.42, "type": "serum"},
        ]
        # Each hospital discovers a subset based on its specialization
        n = self._rng.integers(2, min(5, len(markers)) + 1)
        selected = self._rng.choice(len(markers), size=n, replace=False)
        result = []
        for idx in selected:
            m = markers[idx].copy()
            # Add hospital-specific noise to prevalence
            m["prevalence"] = round(
                max(0.01, m["prevalence"] + self._rng.normal(0, 0.05)), 3
            )
            m["confidence"] = round(0.7 + self._rng.random() * 0.25, 3)
            m["source_hospital"] = self.hospital_id
            result.append(m)
        return result

    def _stratify_patients(self) -> Dict:
        """Simulate patient stratification from local data."""
        total = self.patient_count
        responders = int(total * (0.25 + self._rng.random() * 0.2))
        non_responders = int(total * (0.3 + self._rng.random() * 0.15))
        unknown = total - responders - non_responders
        return {
            "total_patients": total,
            "predicted_responders": responders,
            "predicted_non_responders": non_responders,
            "undetermined": unknown,
            "stratification_confidence": round(0.65 + self._rng.random() * 0.25, 3),
        }


# ═══════════════════════════════════════════════════════════════════════
# Federated Server — Secure Aggregation
# ═══════════════════════════════════════════════════════════════════════

class FederatedServer:
    """
    Central aggregation server for federated learning.

    Collects model weight updates from hospital nodes and produces
    a global model using Federated Averaging (FedAvg).
    Raw patient data is NEVER transmitted — only weight deltas.
    """

    def __init__(self, weight_dim: int = 384, privacy_epsilon: float = 1.0):
        self.weight_dim = weight_dim
        self.privacy_epsilon = privacy_epsilon
        self.global_weights = np.zeros(weight_dim)
        self.round_history: List[Dict] = []
        self.total_rounds = 0

    def aggregate(self, updates: List[Dict]) -> Dict:
        """
        Perform Federated Averaging (FedAvg) across hospital updates.

        Each hospital's contribution is weighted by its patient count.
        Differential privacy noise is added to the aggregated weights.
        """
        self.total_rounds += 1

        # Weighted average by patient count (FedAvg)
        total_patients = sum(u["patients_used"] for u in updates)
        weighted_delta = np.zeros(self.weight_dim)

        for update in updates:
            weight = update["patients_used"] / max(total_patients, 1)
            weighted_delta += weight * update["weight_delta"]

        # Add differential privacy noise (Gaussian mechanism)
        dp_noise = np.random.normal(
            0,
            self.privacy_epsilon * 0.01,
            size=self.weight_dim,
        )
        self.global_weights += weighted_delta + dp_noise

        # Compute convergence metrics
        avg_loss = np.mean([u["local_loss"] for u in updates])
        loss_std = np.std([u["local_loss"] for u in updates])

        # Aggregate biomarkers across hospitals
        all_biomarkers = []
        for u in updates:
            all_biomarkers.extend(u.get("biomarkers", []))
        consensus_biomarkers = self._consensus_biomarkers(all_biomarkers)

        # Aggregate patient stratification
        total_responders = sum(
            u["stratification"]["predicted_responders"] for u in updates
        )
        total_non_responders = sum(
            u["stratification"]["predicted_non_responders"] for u in updates
        )
        total_all = sum(u["stratification"]["total_patients"] for u in updates)

        round_result = {
            "round": self.total_rounds,
            "hospitals_participating": len(updates),
            "total_patients_used": total_patients,
            "avg_loss": round(avg_loss, 4),
            "loss_std": round(loss_std, 4),
            "convergence_delta": round(float(np.linalg.norm(weighted_delta)), 6),
            "privacy_budget_remaining": round(
                max(0, 10.0 - self.privacy_epsilon * self.total_rounds * 0.1), 2
            ),
            "consensus_biomarkers": consensus_biomarkers,
            "global_stratification": {
                "total_patients": total_all,
                "responder_rate": round(total_responders / max(total_all, 1), 3),
                "non_responder_rate": round(total_non_responders / max(total_all, 1), 3),
            },
        }
        self.round_history.append(round_result)
        return round_result

    def _consensus_biomarkers(self, all_biomarkers: List[Dict]) -> List[Dict]:
        """Find biomarkers reported by multiple hospitals (consensus)."""
        from collections import Counter

        name_counts = Counter(b["name"] for b in all_biomarkers)
        consensus = []
        seen = set()

        for b in all_biomarkers:
            if b["name"] in seen:
                continue
            count = name_counts[b["name"]]
            if count >= 2:  # Reported by ≥2 hospitals
                matching = [x for x in all_biomarkers if x["name"] == b["name"]]
                avg_prev = np.mean([m["prevalence"] for m in matching])
                avg_conf = np.mean([m["confidence"] for m in matching])
                consensus.append({
                    "name": b["name"],
                    "type": b["type"],
                    "avg_prevalence": round(float(avg_prev), 3),
                    "avg_confidence": round(float(avg_conf), 3),
                    "reporting_hospitals": count,
                    "consensus_strength": "Strong" if count >= 3 else "Moderate",
                })
                seen.add(b["name"])

        consensus.sort(key=lambda x: x["reporting_hospitals"], reverse=True)
        return consensus

    def get_global_model_state(self) -> Dict:
        """Return the current state of the global model."""
        return {
            "weights_norm": round(float(np.linalg.norm(self.global_weights)), 4),
            "total_rounds": self.total_rounds,
            "final_avg_loss": (
                self.round_history[-1]["avg_loss"] if self.round_history else None
            ),
            "convergence_history": [
                {"round": r["round"], "loss": r["avg_loss"]}
                for r in self.round_history
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# Federated Drug Discovery — End-to-End Orchestrator
# ═══════════════════════════════════════════════════════════════════════

class FederatedDrugDiscovery:
    """
    Orchestrates the full federated drug discovery workflow:

      1. Initialize hospital nodes with diverse patient populations
      2. Run federated training rounds (local train → aggregate → redistribute)
      3. Extract cross-institutional biomarkers via consensus
      4. Generate insights to feed into the 11-stage pipeline
      5. Produce a comprehensive federated learning report
    """

    def __init__(
        self,
        num_rounds: int = 10,
        hospital_profiles: Optional[List[Dict]] = None,
        privacy_epsilon: float = 1.0,
    ):
        self.num_rounds = num_rounds
        self.privacy_epsilon = privacy_epsilon

        profiles = hospital_profiles or HOSPITAL_PROFILES
        self.hospitals = [
            HospitalNode(
                hospital_id=p["id"],
                name=p["name"],
                location=p["location"],
                specialization=p["specialization"],
                patient_count=p["patient_count"],
                data_types=p["data_types"],
                compute=p["compute"],
            )
            for p in profiles
        ]
        self.server = FederatedServer(
            privacy_epsilon=privacy_epsilon,
        )

    def run(
        self,
        disease: str = "Non-Small Cell Lung Cancer",
        target: str = "EGFR",
    ) -> Dict:
        """
        Execute the full federated learning workflow.

        Returns a comprehensive results dictionary containing:
        - Per-round training metrics and convergence history
        - Consensus biomarkers from cross-institutional analysis
        - Global patient stratification insights
        - Privacy budget tracking
        """
        logger.info(
            "Starting federated learning: %d hospitals × %d rounds",
            len(self.hospitals),
            self.num_rounds,
        )

        all_round_results = []

        for round_num in range(1, self.num_rounds + 1):
            # Step 1: Each hospital trains locally
            updates = []
            for hospital in self.hospitals:
                update = hospital.local_train(
                    global_weights=self.server.global_weights
                )
                updates.append(update)

            # Step 2: Server aggregates updates
            round_result = self.server.aggregate(updates)

            # Step 3: Track per-hospital details for this round
            round_result["hospital_details"] = [
                {
                    "hospital_id": u["hospital_id"],
                    "hospital_name": u["hospital_name"],
                    "local_loss": u["local_loss"],
                    "patients_used": u["patients_used"],
                    "biomarkers_found": len(u["biomarkers"]),
                }
                for u in updates
            ]
            all_round_results.append(round_result)

            logger.info(
                "  Round %2d/%d — avg_loss=%.4f, convergence_delta=%.6f",
                round_num,
                self.num_rounds,
                round_result["avg_loss"],
                round_result["convergence_delta"],
            )

        # Final global model state
        global_state = self.server.get_global_model_state()

        # Compile final results
        final_round = all_round_results[-1]
        results = {
            "disease": disease,
            "target": target,
            "num_hospitals": len(self.hospitals),
            "num_rounds": self.num_rounds,
            "privacy_epsilon": self.privacy_epsilon,
            "global_model": global_state,
            "round_history": all_round_results,
            "consensus_biomarkers": final_round["consensus_biomarkers"],
            "global_stratification": final_round["global_stratification"],
            "privacy_budget_remaining": final_round["privacy_budget_remaining"],
            "hospital_summary": [
                {
                    "id": h.hospital_id,
                    "name": h.name,
                    "location": h.location,
                    "specialization": h.specialization,
                    "patients": h.patient_count,
                    "data_types": h.data_types,
                    "compute": h.compute,
                }
                for h in self.hospitals
            ],
        }

        logger.info(
            "Federated learning complete: final_loss=%.4f, biomarkers=%d",
            global_state["final_avg_loss"],
            len(results["consensus_biomarkers"]),
        )

        return results

    def generate_report(self, results: Dict) -> str:
        """Generate a comprehensive markdown report of federated results."""
        lines = [
            "# 🏥 Federated Drug Discovery Report\n",
            f"**Disease**: {results['disease']}  ",
            f"**Target**: {results['target']}  ",
            f"**Hospitals**: {results['num_hospitals']}  ",
            f"**Rounds**: {results['num_rounds']}  ",
            f"**Privacy ε**: {results['privacy_epsilon']}  \n",
            "---\n",
            "## Participating Hospitals\n",
            "| ID | Hospital | Location | Specialization | Patients |",
            "|-----|----------|----------|----------------|----------|",
        ]

        for h in results["hospital_summary"]:
            lines.append(
                f"| {h['id']} | {h['name']} | {h['location']} | "
                f"{h['specialization']} | {h['patients']:,} |"
            )

        lines.append("\n## Convergence\n")
        gm = results["global_model"]
        lines.append(f"- Final average loss: **{gm['final_avg_loss']:.4f}**")
        lines.append(f"- Total rounds: **{gm['total_rounds']}**")
        lines.append(
            f"- Privacy budget remaining: **{results['privacy_budget_remaining']:.2f}** / 10.0"
        )

        lines.append("\n## Consensus Biomarkers\n")
        if results["consensus_biomarkers"]:
            lines.append(
                "| Biomarker | Type | Prevalence | Confidence | Hospitals | Strength |"
            )
            lines.append(
                "|-----------|------|------------|------------|-----------|----------|"
            )
            for b in results["consensus_biomarkers"]:
                lines.append(
                    f"| {b['name']} | {b['type']} | {b['avg_prevalence']:.1%} | "
                    f"{b['avg_confidence']:.1%} | {b['reporting_hospitals']} | "
                    f"{b['consensus_strength']} |"
                )
        else:
            lines.append("*No consensus biomarkers found across hospitals.*")

        strat = results["global_stratification"]
        lines.append("\n## Global Patient Stratification\n")
        lines.append(f"- Total patients: **{strat['total_patients']:,}**")
        lines.append(f"- Predicted responder rate: **{strat['responder_rate']:.1%}**")
        lines.append(
            f"- Predicted non-responder rate: **{strat['non_responder_rate']:.1%}**"
        )

        lines.append("\n---\n")
        lines.append(
            "*All data was processed locally at each hospital. "
            "Only model weight updates were shared with the aggregation server. "
            "Patient privacy was preserved throughout.*"
        )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Public API — Entry Points
# ═══════════════════════════════════════════════════════════════════════

def run_federated_pipeline(
    disease: str = "Non-Small Cell Lung Cancer",
    target: str = "EGFR",
    num_rounds: int = 10,
    privacy_epsilon: float = 1.0,
) -> Dict:
    """
    Run the federated drug discovery pipeline.

    Returns a results dictionary that can be passed to the main pipeline
    or used independently.
    """
    fd = FederatedDrugDiscovery(
        num_rounds=num_rounds,
        privacy_epsilon=privacy_epsilon,
    )
    return fd.run(disease=disease, target=target)


def print_federated_results(results: Dict):
    """Print a summary of federated learning results to the console."""
    from tabulate import tabulate

    print("\n" + "=" * 70)
    print("  🏥 Federated Drug Discovery — Multi-Hospital Aggregation")
    print("  Privacy-preserving collaborative learning with HAI-DEF")
    print("=" * 70)

    # Hospital table
    print("\n  📋 Participating Hospitals:")
    h_table = []
    for h in results["hospital_summary"]:
        h_table.append([
            h["id"],
            h["name"],
            h["location"],
            h["specialization"],
            f"{h['patients']:,}",
        ])
    print(tabulate(
        h_table,
        headers=["ID", "Hospital", "Location", "Specialization", "Patients"],
        tablefmt="rounded_outline",
    ))

    # Convergence
    gm = results["global_model"]
    print(f"\n  📉 Convergence:")
    print(f"     Final avg loss:        {gm['final_avg_loss']:.4f}")
    print(f"     Total rounds:          {gm['total_rounds']}")
    print(f"     Privacy budget left:   {results['privacy_budget_remaining']:.2f} / 10.0")

    # Training history (abbreviated)
    print(f"\n  📊 Training History (loss per round):")
    history = gm["convergence_history"]
    hist_table = [[r["round"], f"{r['loss']:.4f}"] for r in history]
    print(tabulate(
        hist_table,
        headers=["Round", "Avg Loss"],
        tablefmt="rounded_outline",
    ))

    # Consensus biomarkers
    print(f"\n  🧬 Consensus Biomarkers (found by ≥2 hospitals):")
    if results["consensus_biomarkers"]:
        b_table = []
        for b in results["consensus_biomarkers"]:
            b_table.append([
                b["name"],
                b["type"],
                f"{b['avg_prevalence']:.1%}",
                f"{b['avg_confidence']:.1%}",
                b["reporting_hospitals"],
                b["consensus_strength"],
            ])
        print(tabulate(
            b_table,
            headers=["Biomarker", "Type", "Prevalence", "Confidence", "Hospitals", "Strength"],
            tablefmt="rounded_outline",
        ))
    else:
        print("     No consensus biomarkers found.")

    # Global stratification
    strat = results["global_stratification"]
    print(f"\n  👥 Global Patient Stratification:")
    print(f"     Total patients:           {strat['total_patients']:,}")
    print(f"     Predicted responders:     {strat['responder_rate']:.1%}")
    print(f"     Predicted non-responders: {strat['non_responder_rate']:.1%}")

    print(f"\n  🔒 Privacy: All data processed locally. Only model weight")
    print(f"     updates were shared. Patient data never left hospital nodes.")
    print()
