# Agnosti-Path API Documentation

## Overview

Agnosti-Path provides both a Python API and a web interface for drug discovery pipelines.

## Python API

### Basic Usage

```python
from pipeline.main import DrugDiscoveryPipeline
from pipeline.config import PipelineConfig

# Initialize pipeline
config = PipelineConfig(
    use_gpu=True,
    quantization='4bit',
    simulation_mode=False
)

pipeline = DrugDiscoveryPipeline(config)

# Run full pipeline
results = pipeline.run(
    disease='non-small cell lung cancer',
    target='EGFR',
    compounds=['CCO', 'CCN', 'CCC']  # SMILES strings
)

# Get ranked compounds
for compound in results.ranked_compounds:
    print(f"{compound.name}: {compound.composite_score}")
```

### PipelineConfig

Configuration class for the pipeline.

```python
class PipelineConfig:
    use_gpu: bool = True              # Use GPU acceleration
    quantization: str = '4bit'        # '4bit', '8bit', or None
    simulation_mode: bool = False     # Use mock models
    cache_models: bool = True         # Cache loaded models
    batch_size: int = 8               # Processing batch size
```

### DrugDiscoveryPipeline

Main pipeline orchestrator.

#### Methods

##### `run(disease, target, compounds)`

Execute full 11-stage pipeline.

**Parameters:**
- `disease` (str): Disease name
- `target` (str): Protein target
- `compounds` (List[str]): List of SMILES strings

**Returns:** `PipelineResults`

##### `run_stage(stage_name, input_data)`

Run a specific pipeline stage.

**Parameters:**
- `stage_name` (str): One of the 11 stage names
- `input_data` (dict): Stage-specific input

**Returns:** Stage output

### PipelineResults

Container for pipeline results.

```python
class PipelineResults:
    ranked_compounds: List[CompoundResult]
    safety_flags: List[SafetyFlag]
    stage_outputs: Dict[str, Any]
    execution_time: float
```

### CompoundResult

Individual compound analysis.

```python
class CompoundResult:
    name: str
    smiles: str
    composite_score: float
    verdict: str  # 'Advance', 'Optimize', 'Monitor', 'Reject'
    stage_scores: Dict[str, float]
    safety_flags: List[str]
    recommendations: List[str]
```

## Web API (Gradio)

Launch the interactive demo:

```bash
python app.py
```

### Endpoints

#### POST /run_pipeline

Execute pipeline via HTTP.

**Request:**
```json
{
  "disease": "non-small cell lung cancer",
  "target": "EGFR",
  "compounds": ["CCO", "CCN"]
}
```

**Response:**
```json
{
  "ranked_compounds": [
    {
      "name": "Erlotinib",
      "composite_score": 0.87,
      "verdict": "Advance"
    }
  ],
  "execution_time": 120.5
}
```

## Stage-Specific APIs

### Stage 1: Target Identification

```python
from pipeline.target_identification import TargetIdentifier

identifier = TargetIdentifier(model='txgemma-27b')
targets = identifier.identify_targets('lung cancer')
```

### Stage 2: Lead Discovery

```python
from pipeline.lead_discovery import LeadDiscovery

discovery = LeadDiscovery(model='txgemma-2b')
candidates = discovery.generate_candidates(target='EGFR', num_candidates=100)
```

### Stage 3: Binding Affinity

```python
from pipeline.binding_affinity import BindingPredictor

predictor = BindingPredictor(model='txgemma-9b')
affinity = predictor.predict(smiles='CCO', protein='EGFR')
# Returns: {'pic50': 7.5, 'kd_nm': 31.6}
```

### Stage 4: ADMET Profiling

```python
from pipeline.admet_profiling import ADMETProfiler

profiler = ADMETProfiler(model='txgemma-2b')
profile = profiler.profile(smiles='CCO')
# Returns: {'solubility': 'high', 'bbb': 'low', 'hERG': 'safe'}
```

## Error Handling

```python
from pipeline.exceptions import PipelineError, ModelLoadError

try:
    results = pipeline.run(disease='cancer', target='EGFR')
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
except PipelineError as e:
    print(f"Pipeline failed: {e}")
```

## Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('agnosti-path')

# Pipeline will log progress
```

## Performance Tips

1. **Use quantization**: Set `quantization='4bit'` for faster inference
2. **Enable caching**: Set `cache_models=True` to avoid reloading
3. **Batch processing**: Increase `batch_size` for multiple compounds
4. **GPU utilization**: Ensure CUDA is available for best performance

## Examples

### Example 1: Single Compound Analysis

```python
from pipeline.main import DrugDiscoveryPipeline

pipeline = DrugDiscoveryPipeline()
result = pipeline.analyze_compound(
    smiles='CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4(C)C)O)C)C',
    name='Lanosterol'
)
print(f"Score: {result.composite_score}")
print(f"Verdict: {result.verdict}")
```

### Example 2: Batch Processing

```python
compounds = [
    {'smiles': 'CCO', 'name': 'Ethanol'},
    {'smiles': 'CCN', 'name': 'Ethylamine'},
    {'smiles': 'CCC', 'name': 'Propane'}
]

results = pipeline.analyze_batch(compounds)
for r in results:
    print(f"{r.name}: {r.verdict}")
```

### Example 3: Custom Scoring Weights

```python
from pipeline.cross_stage_intelligence import MCDAScorer

scorer = MCDAScorer(
    weights={
        'binding_affinity': 0.30,
        'admet_safety': 0.25,
        'clinical_viability': 0.20,
        'pathology': 0.15,
        'imaging': 0.10
    }
)
```

---

For more examples, see the `examples/` directory.
