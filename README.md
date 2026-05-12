# Unsupervised Defect Severity Grading

This repository contains the core implementation of the paper:  
**"Severity grading of industrial defects without any labels: an interpretable unsupervised approach"**  

The method extends unsupervised anomaly detection (PaDiM) to quantify defect severity and assign interpretable grades (normal / mild / moderate / severe) using entropy‑weighted fusion of seven spatially coupled features.

## Core Files

| File | Description |
|------|-------------|
| `config_sci.py` | All configuration parameters (dataset path, image size, quantiles, class list, etc.) |
| `core_pipeline.py` | PaDiM model, feature extraction, entropy weight method, connected component analysis |
| `run_experiments.py` | Main pipeline: train PaDiM, extract features, perform unsupervised grading, evaluate Kappa, save models/rules |
| `infer_defect_grade_sci.py` | Inference on a single image: load model and grading rules, output defect grade and visualization |
| `requirements.txt` | Python dependencies |

## Running Order

1. **Configure** – Edit `config_sci.py`:
   - Set `DATASET_ROOT` to your MVTec AD dataset path.
   - Adjust other parameters if needed (image size, calibration ratio, quantile cuts, etc.).

2. **Install dependencies** (recommended in a virtual environment):
   ```bash
   pip install -r requirements.txt