"""
Pipeline Configuration
Central configuration for model IDs, hyperparameters, and constants.
"""

# ═══════════════════════════════════════════════════════════
# HAI-DEF Model IDs (Hugging Face)
# ═══════════════════════════════════════════════════════════

TXGEMMA_2B_PREDICT = "google/txgemma-2b-predict"
TXGEMMA_9B_PREDICT = "google/txgemma-9b-predict"
TXGEMMA_27B_PREDICT = "google/txgemma-27b-predict"
TXGEMMA_27B_CHAT = "google/txgemma-27b-chat"
MEDGEMMA_4B = "google/medgemma-4b-it"
PATH_FOUNDATION = "google/path-foundation"
MEDSIGLIP = "google/medsiglip"
DERM_FOUNDATION = "google/derm-foundation"
CXR_FOUNDATION = "google/cxr-foundation"
DEEPCHEM_GCN = "deepchem/graphconv"
ALPHAFOLD_DB = "alphafold-ebi/structure-db"

# Default model for each pipeline stage
DEFAULT_MODELS = {
    "target_identification": TXGEMMA_27B_CHAT,
    "lead_discovery": TXGEMMA_2B_PREDICT,
    "binding_affinity": TXGEMMA_9B_PREDICT,
    "admet_profiling": TXGEMMA_2B_PREDICT,
    "clinical_reasoning": TXGEMMA_27B_CHAT,
    "medical_literature": MEDGEMMA_4B,
    "pathology_analysis": PATH_FOUNDATION,
    "medical_imaging": MEDSIGLIP,
    "derm_analysis": DERM_FOUNDATION,
    "cxr_analysis": CXR_FOUNDATION,
    "deepchem_analysis": DEEPCHEM_GCN,
    "alphafold_analysis": ALPHAFOLD_DB,
}

# ═══════════════════════════════════════════════════════════
# TxGemma Prompt Templates
# Derived from the official TxGemma documentation for
# Therapeutic Data Commons (TDC) tasks.
# ═══════════════════════════════════════════════════════════

PROMPT_TEMPLATES = {
    # Drug-Target Interaction (DTI) — binary binding prediction
    "dti_binding": (
        "Given the drug SMILES string: {smiles}\n"
        "And the protein target amino acid sequence: {protein_seq}\n"
        "Predict whether this drug binds to the target. "
        "Answer with 'Yes' or 'No' and a confidence score between 0 and 1."
    ),

    # Binding Affinity — Kd prediction
    "binding_affinity": (
        "Predict the binding affinity (Kd in nM) for the following:\n"
        "Drug SMILES: {smiles}\n"
        "Target protein: {target_name}\n"
        "Provide a numerical prediction."
    ),

    # ADMET — Absorption
    "admet_solubility": (
        "Predict the aqueous solubility for the molecule with SMILES: {smiles}\n"
        "Classify as: 'High', 'Medium', or 'Low'."
    ),

    # ADMET — Toxicity (hERG inhibition)
    "admet_herg": (
        "Predict whether the molecule with SMILES: {smiles} "
        "is a hERG channel blocker (cardiotoxicity risk). "
        "Answer 'Yes' (blocker) or 'No' (non-blocker)."
    ),

    # ADMET — Blood-Brain Barrier
    "admet_bbb": (
        "Predict whether the molecule with SMILES: {smiles} "
        "can penetrate the blood-brain barrier. "
        "Answer 'Yes' or 'No'."
    ),

    # ADMET — CYP450 inhibition
    "admet_cyp": (
        "Predict whether the molecule with SMILES: {smiles} "
        "inhibits CYP3A4 enzyme. Answer 'Yes' or 'No'."
    ),

    # ADMET — Lipophilicity
    "admet_lipophilicity": (
        "Predict the lipophilicity (logP value) for the molecule "
        "with SMILES: {smiles}. Provide a numerical value."
    ),

    # Clinical Trial Approval
    "clinical_approval": (
        "Given a drug candidate with SMILES: {smiles}\n"
        "Target: {target_name}\n"
        "Disease: {disease}\n"
        "Predict the probability of this drug passing Phase {phase} "
        "clinical trials. Provide a probability between 0 and 1."
    ),

    # Molecular description
    "molecule_description": (
        "Describe the molecular properties and potential therapeutic "
        "applications of the compound with SMILES: {smiles}"
    ),
}

# ═══════════════════════════════════════════════════════════
# ADMET Property Thresholds
# ═══════════════════════════════════════════════════════════

ADMET_THRESHOLDS = {
    "solubility": {"high": -2.0, "medium": -4.0},  # logS
    "lipophilicity": {"ideal_min": 1.0, "ideal_max": 3.0},  # logP
    "molecular_weight": {"max": 500},  # Lipinski rule
    "hbd": {"max": 5},  # Hydrogen bond donors
    "hba": {"max": 10},  # Hydrogen bond acceptors
}

# ═══════════════════════════════════════════════════════════
# Sample Protein Sequences (Targets)
# ═══════════════════════════════════════════════════════════

SAMPLE_TARGETS = {
    "EGFR": {
        "name": "Epidermal Growth Factor Receptor",
        "uniprot": "P00533",
        "disease": "Non-Small Cell Lung Cancer (NSCLC)",
        "sequence_fragment": (
            "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQL"
            "GTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKT"
        ),
    },
    "BCR-ABL": {
        "name": "BCR-ABL Fusion Protein",
        "uniprot": "P00519",
        "disease": "Chronic Myeloid Leukemia (CML)",
        "sequence_fragment": (
            "MGCGCSSHPEDDWMENIDVCENCHYPIVPLDGKGTLLRNGS"
            "EVRDVRGAESGPPSPRQRLKFKAYQLAEKNEIPENLEYDFT"
        ),
    },
    "COX-2": {
        "name": "Cyclooxygenase-2",
        "uniprot": "P35354",
        "disease": "Pain / Inflammation",
        "sequence_fragment": (
            "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYK"
            "CDCTRTGFYGENCSTPEFLTRIKLFLKPTPNTVHYILTHFK"
        ),
    },
}

# ═══════════════════════════════════════════════════════════
# Pipeline Defaults
# ═══════════════════════════════════════════════════════════

DATA_DIR = "data"
SAMPLE_COMPOUNDS_FILE = "data/sample_compounds.csv"
OUTPUT_DIR = "output"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.1  # Low temperature for deterministic predictions
