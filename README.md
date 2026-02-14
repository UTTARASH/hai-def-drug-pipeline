# HAI-DEF Drug Discovery Pipeline

> **A comprehensive drug discovery pipeline powered by Google's Health AI Developer Foundations (HAI-DEF) — TxGemma, MedGemma, and Path Foundation.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HAI-DEF](https://img.shields.io/badge/Google-HAI--DEF-4285F4)](https://health.google/developers/)

## Overview

This pipeline demonstrates **end-to-end drug discovery** using Google's open-weight Health AI Developer Foundations models:

| Stage | Model | Task |
|-------|-------|------|
| 🎯 Target Identification | TxGemma | Protein target analysis & disease association |
| 💊 Lead Discovery | TxGemma-Predict | Molecular property prediction (SMILES) |
| 🔬 Binding Affinity | TxGemma-Predict | Drug-target interaction scoring |
| ⚗️ ADMET Profiling | TxGemma-Predict | Absorption, Distribution, Metabolism, Excretion, Toxicity |
| 🧪 Clinical Viability | TxGemma-Chat | Conversational reasoning about drug candidates |
| 🔬 Pathology Analysis | Path Foundation | Histopathology embeddings & treatment response |
| 📊 Medical Literature | MedGemma | Evidence synthesis from medical text |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Drug Discovery Pipeline                 │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│  Target  │   Lead   │ Binding  │  ADMET   │  Clinical   │
│  ID      │ Discovery│ Affinity │ Profile  │  Reasoning  │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│              HAI-DEF Model Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ TxGemma-2B   │  │ TxGemma-9B   │  │ TxGemma-27B    │ │
│  │ (Predict)    │  │ (Predict)    │  │ (Chat/Predict) │ │
│  └──────────────┘  └──────────────┘  └────────────────┘ │
│              ┌──────────────┐                            │
│              │  MedGemma    │                            │
│              │  (Medical)   │                            │
│              └──────────────┘                            │
│              ┌──────────────────┐                        │
│              │ Path Foundation  │                        │
│              │ (Histopathology) │                        │
│              └──────────────────┘                        │
├─────────────────────────────────────────────────────────┤
│  Data: SMILES • Protein Sequences • H&E Tissue Patches  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/hai-def-drug-discovery.git
cd hai-def-drug-discovery
pip install -r requirements.txt
```

### 2. Authenticate with Hugging Face

```bash
huggingface-cli login
# Accept model terms at https://huggingface.co/google/txgemma-2b-predict
```

### 3. Run the Pipeline

```bash
# Full pipeline demo
python -m pipeline.main

# Individual stages
python -m pipeline.target_identification
python -m pipeline.lead_discovery
python -m pipeline.admet_profiling
python -m pipeline.clinical_reasoning
python -m pipeline.pathology_analysis
```

### 4. Run as Notebook

```bash
jupyter notebook notebooks/drug_discovery_pipeline.ipynb
```

## Project Structure

```
hai-def-drug-discovery/
├── pipeline/
│   ├── __init__.py
│   ├── main.py                    # Orchestrates the full pipeline
│   ├── config.py                  # Model IDs, hyperparameters, constants
│   ├── model_loader.py            # HAI-DEF model loading utilities
│   ├── target_identification.py   # Stage 1: Target ID & disease mapping
│   ├── lead_discovery.py          # Stage 2: Molecular screening & scoring
│   ├── binding_affinity.py        # Stage 3: Drug-target interaction
│   ├── admet_profiling.py         # Stage 4: ADMET property prediction
│   ├── clinical_reasoning.py      # Stage 5: TxGemma-Chat analysis
│   ├── pathology_analysis.py      # Stage 6: Path Foundation histopathology
│   └── visualization.py           # Charts, molecular visualization
├── data/
│   └── sample_compounds.csv       # Example drug candidates (SMILES)
├── notebooks/
│   └── drug_discovery_pipeline.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

## Example Output

```
═══════════════════════════════════════════════════════════
  HAI-DEF Drug Discovery Pipeline — Results Summary
═══════════════════════════════════════════════════════════

Target: EGFR (Epidermal Growth Factor Receptor)
Disease: Non-Small Cell Lung Cancer (NSCLC)

┌─── Top Candidates ───────────────────────────────────────
│ Rank │ Name         │ SMILES           │ Affinity │ ADMET
│ #1   │ Erlotinib    │ COc1cc2ncnc...   │ 0.92     │ ✅ Pass
│ #2   │ Gefitinib    │ COc1cc2c(Nc...   │ 0.87     │ ✅ Pass
│ #3   │ Afatinib     │ CN(C)C/C=C/...   │ 0.85     │ ⚠️ Review
│ #4   │ Candidate-X  │ CC(=O)Nc1cc...   │ 0.78     │ ✅ Pass
└──────────────────────────────────────────────────────────
```

## Models Used

| Model | Hugging Face ID | Parameters | Use Case |
|-------|----------------|------------|----------|
| TxGemma Predict (2B) | `google/txgemma-2b-predict` | 2B | Fast screening |
| TxGemma Predict (9B) | `google/txgemma-9b-predict` | 9B | Detailed predictions |
| TxGemma Chat (27B) | `google/txgemma-27b-chat` | 27B | Scientific reasoning |
| MedGemma 4B | `google/medgemma-4b-it` | 4B | Medical context |
| Path Foundation | `google/path-foundation` | ViT-S | Histopathology embeddings |

## Disclaimer

⚠️ **For research and educational purposes only.** This pipeline is not validated for clinical use. Drug development requires extensive regulatory testing. Always consult qualified professionals for therapeutic decisions.

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.
