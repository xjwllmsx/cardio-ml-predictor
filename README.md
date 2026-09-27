# Cardiovascular Disease Prediction Using Machine Learning

A machine learning project comparing eight classification algorithms, plus a stacking ensemble, for predicting cardiovascular disease from patient health metrics — with leak-free pipelines, cross-validated tuning, threshold optimization, and SHAP-based interpretability.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

## Overview

This project develops and evaluates binary classification models to predict cardiovascular disease (CVD) presence based on patient health data. It demonstrates a complete, leak-free supervised learning workflow: rule-based data cleaning, pipeline-integrated feature engineering, cross-validated hyperparameter tuning across eight model families plus a stacking ensemble, decision-threshold tuning for a medical-screening cost trade-off, and a final evaluation that includes bootstrap confidence intervals, calibration, subgroup fairness, and SHAP explanations.

## Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early detection through routine health screenings could enable preventive interventions and improve patient outcomes. This project explores whether machine learning can effectively predict CVD using commonly collected health metrics such as blood pressure, cholesterol levels, and lifestyle factors.

**Research Questions:**
- Can machine learning models accurately predict CVD from basic health metrics?
- Which classification algorithm performs best for this medical prediction task?
- How does cross-validated hyperparameter tuning impact model performance?
- What does the cost of a missed diagnosis versus a false alarm mean for where the decision threshold should sit?
- Does model performance hold up evenly across patient subgroups (age, gender)?

## Dataset

**Source:** [Heart Failure Diagnosis Data for Machine Learning](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning) (Kaggle)

**Size:** 70,000 patient records; 68,584 (98.0%) remain after removing duplicates and physiologically implausible values (see [Methodology](#methodology))

**Features (11 total):**
- Age (in days)
- Gender (1: female, 2: male)
- Height (cm)
- Weight (kg)
- Systolic blood pressure (ap_hi)
- Diastolic blood pressure (ap_lo)
- Cholesterol level (1: normal, 2: above normal, 3: well above normal)
- Glucose level (1: normal, 2: above normal, 3: well above normal)
- Smoking status (binary)
- Alcohol consumption (binary)
- Physical activity (binary)

**Target Variable:**
- Cardio (0: no CVD, 1: CVD present)

**Class Distribution:** Balanced (~50% CVD, ~50% healthy)

## Methodology

Reusable code lives in the `src/cardio` package; the notebook is the narrative that calls into it (see [Project Structure](#project-structure)).

### 1. Data Cleaning
- Drop duplicate records and rows with physiologically implausible values, using fixed clinical bounds rather than anything learned from the data (`cardio.data.clean_data`):
  - Systolic blood pressure outside 80-250 mmHg
  - Diastolic blood pressure outside 40-160 mmHg, or diastolic ≥ systolic
  - Height outside 120-220 cm, weight outside 30-200 kg
- Because every rule is a fixed threshold, cleaning is safe to apply before the train/test split.

### 2. Feature Engineering
- Age converted from days to years; gender recoded to 0/1
- Derived features: BMI, pulse pressure (systolic − diastolic), and mean arterial pressure
- Implemented as a `FunctionTransformer` and run as the **first step of every model's pipeline** (`cardio.features`), so the exact transformation used in training is applied automatically at prediction time

### 3. Pipelines and Preprocessing
- Every model is a full `scikit-learn` `Pipeline`: feature engineering → preprocessing → classifier (`cardio.preprocessing.build_pipeline`)
- Continuous features are standardized for distance/gradient-based models (Logistic Regression, KNN, MLP) and passed through unscaled for tree-based models
- Because preprocessing lives inside the pipeline, cross-validation refits it on each fold's training data alone — the original version's single `StandardScaler.fit()` on the whole training set would leak information between CV folds

### 4. Train/Test Split
- 80% training set (used only for cross-validated tuning) / 20% test set (touched exactly once, at the end), stratified by target
- A larger training split than the original 70/30 is affordable now that model selection happens by cross-validation rather than by repeatedly checking the test set

### 5. Model Development
**Algorithms Tested**, each tuned with `StratifiedKFold(5)` cross-validation, scored on ROC-AUC, PR-AUC, Brier score, recall, F1, and accuracy simultaneously (`cardio.models`, `cardio.evaluation`):

- Dummy (stratified) — floor baseline
- Logistic Regression
- Gaussian Naive Bayes
- Decision Tree
- K-Nearest Neighbors (`k` searched from 5 to 101)
- Random Forest
- XGBoost
- Neural Network (MLP)
- **Stacking Ensemble** — the three strongest models combined via a Logistic Regression meta-learner

The final model is the simplest one whose mean cross-validated ROC-AUC is within one standard deviation of the best (`cardio.evaluation.select_final_model`, using the simplest-to-most-complex ranking in `cardio.models.COMPLEXITY_ORDER`) — a gain smaller than the fold-to-fold noise isn't worth the extra complexity.

### 6. Threshold Tuning
- The default 0.5 probability threshold is replaced with one tuned to optimize the F2 score (recall weighted twice as heavily as precision), using `TunedThresholdClassifierCV` on out-of-fold training predictions — reflecting that missing a CVD case is worse than an unnecessary follow-up test

### 7. Final Evaluation
- Confusion matrix, ROC-AUC, PR-AUC, Brier score, log loss, accuracy, balanced accuracy, sensitivity/specificity, F1/F2 — computed once, on the held-out test set
- 95% bootstrap confidence intervals (1,000 resamples) for ROC-AUC and recall
- ROC, precision-recall, and calibration curve overlays across models
- Subgroup breakdown by gender and age band

### 8. Interpretability
- Permutation importance on the test set (model-agnostic)
- SHAP `TreeExplainer` (XGBoost) and `LinearExplainer` (Logistic Regression) summary plots, plus per-patient waterfall plots for a true positive and a false negative

## Results

### Cross-Validated Model Comparison (5-fold, training set)

| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| Dummy (floor) | 0.500 | 50.0% | 0.490 |
| Naive Bayes | 0.784 | 72.1% | 0.687 |
| Logistic Regression | 0.792 | 72.8% | 0.708 |
| Decision Tree | 0.793 | 73.1% | 0.708 |
| K-Nearest Neighbors | 0.796 | 73.1% | 0.711 |
| **Random Forest** (selected) | **0.800** | 73.6% | 0.720 |
| Neural Network (MLP) | 0.800 | 73.4% | 0.717 |
| XGBoost | 0.802 | 73.7% | 0.721 |
| Stacking Ensemble (XGBoost + Random Forest + MLP) | 0.802 | 73.6% | 0.722 |

The Stacking Ensemble had the highest mean ROC-AUC (0.8022 ± 0.0035), but XGBoost, Random Forest, and the MLP all fall within one standard deviation of it — differences smaller than the fold-to-fold noise. Random Forest is the simplest of those candidates, so it was selected.

### Final Test Set Evaluation

| Metric | Random Forest (tuned threshold) | Random Forest (default 0.5) | Logistic Regression |
|---|---|---|---|
| Accuracy | 57.5% | 73.2% | 72.4% |
| Recall (Sensitivity) | 97.8% | 68.5% | 66.3% |
| Specificity | 18.1% | 77.7% | 78.4% |
| Precision | 53.9% | 75.1% | 75.0% |
| F1 / F2 | 0.695 / 0.841 | 0.716 / 0.697 | 0.704 / 0.679 |
| ROC-AUC | 0.798 (95% CI 0.791-0.806) | 0.798 | 0.789 |

**Decision threshold:** tuned to 0.172 (from a default of 0.5) to optimize F2.

### Key Outcomes

- **Final model:** Random Forest, cross-validated ROC-AUC 0.800 versus 0.792 for tuned Logistic Regression — chosen over the Stacking Ensemble (0.802) because the ensemble's edge was within one standard deviation.
- **Threshold tuning is a real trade-off, not a free win.** Optimizing for F2 cuts missed CVD cases from roughly 1 in 3 (default threshold) to **2.2%** (146 of 6,787), a 95% CI of 97.5-98.2% recall — but it also drops overall accuracy from 73.2% to 57.5% and specificity to 18%, meaning most healthy patients would now be flagged for follow-up too.
- **That cost isn't distributed evenly.** At the tuned threshold, patients 55+ end up with a specificity **under 1%** (nearly everyone flagged), while patients under 45 keep a much more balanced 61.1% specificity — a fairness consideration a single aggregate F2 score hides. See the notebook's subgroup analysis for the full breakdown.

## Technologies Used

- **Python 3.12**
- **pandas** / **NumPy** — data manipulation and numerical computing
- **scikit-learn** — pipelines, cross-validation, models, and evaluation metrics
- **XGBoost** — gradient-boosted trees
- **SHAP** — model interpretability
- **Matplotlib** — data visualization
- **joblib** — model persistence
- **pytest** — unit tests for cleaning, feature engineering, and model fitting
- **Jupyter Notebook** — interactive development environment

## Installation

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xjwllmsx/cardio-ml-predictor.git
cd cardio-ml-predictor
```

2. Choose your preferred installation method:

#### Option A: Using uv (Recommended - Faster)

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies (including the local cardio package)
uv sync
```

#### Option B: Using pip (Traditional Method)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages, then the local cardio package in editable mode
pip install -r requirements.txt
pip install -e .
```

See `requirements.txt` for the complete pinned dependency list (pandas, scikit-learn, XGBoost, SHAP, matplotlib, joblib, and Jupyter tooling).

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning)

2. Place the CSV file in the project root directory as `HeartFailureDataset.csv`

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `cardio-ml-predictor.ipynb` and run all cells (a full run — including hyperparameter search across all 8 models and SHAP — takes roughly 10-15 minutes on a modern multi-core machine)

The notebook executes the complete pipeline: data cleaning, feature engineering, cross-validated model comparison, ensembling, threshold tuning, final test evaluation, and interpretability — and saves the final tuned pipeline to `models/best_pipeline.joblib` and its metrics to `results/metrics.csv`.

### Running the tests

```bash
uv run pytest
```

Covers data cleaning rules, feature engineering, every model spec fitting and producing probabilities, and the cross-validation comparison helpers.

## Project Structure

```
cardio-ml-predictor/
│
├── cardio-ml-predictor.ipynb        # Main analysis notebook (imports from src/cardio)
├── src/cardio/                      # Reusable pipeline code
│   ├── config.py                    # Paths, column groups, plausibility bounds
│   ├── data.py                      # Loading and rule-based cleaning
│   ├── features.py                  # Domain feature engineering (BMI, pulse pressure, MAP)
│   ├── preprocessing.py             # ColumnTransformer + Pipeline builder
│   ├── models.py                    # Model specs, search spaces, stacking ensemble
│   ├── evaluation.py                # CV tuning, metrics, bootstrap CIs, plots
│   └── explain.py                   # Permutation importance and SHAP helpers
├── tests/                           # pytest coverage for the src/cardio package
├── models/                          # Saved best pipeline (generated; gitignored)
├── results/                         # Saved metrics table (generated; gitignored)
├── HeartFailureDataset.csv          # Dataset (download separately)
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project metadata and dependencies (uv)
├── uv.lock                          # Locked dependency versions (uv)
├── .gitignore                       # Git ignore rules
├── .python-version                  # Python version specification
├── README.md                        # Project documentation
├── LICENSE                          # MIT License
```

## Key Findings

### Tree ensembles and boosting edge out the linear baseline, but not by much

XGBoost, Random Forest, and the MLP all land within about a point of ROC-AUC of each other (0.800-0.802), and all beat Logistic Regression's 0.792. This suggests there's a modest amount of non-linear structure in the relationship between health metrics and CVD risk that tree- and gradient-based models can capture, but the gap is far smaller than the original project's "KNN vs. Logistic Regression" comparison implied — most of the achievable signal in this feature set is roughly linear.

### Cross-validation replaces tuning against the test set

The original version selected KNN's `k` by comparing test-set accuracy for `k=5` vs. `k=9` — a subtle leak that makes the reported "test" performance optimistic. Every model here is tuned on the training set alone via 5-fold cross-validation, and the test set is used exactly once, at the end.

### Threshold tuning surfaces a real, uneven trade-off

Optimizing the decision threshold for F2 substantially reduces missed CVD cases (down to 2.2%), but at a real cost to overall accuracy and specificity — and that cost falls much harder on older patients than younger ones (see [Results](#results)). This is a concrete illustration of why a single aggregate metric can hide subgroup-level problems.

### Medical AI Still Requires Higher Standards

At either threshold, accuracy stays well under the roughly 90%+ (with a very low false-negative rate) often cited for autonomous clinical decision-making — a heuristic, not a hard regulatory number, but a useful gut check. This model, like the original, is best framed as a triage/screening aid that flags patients for clinician review, not a diagnostic tool.

## Limitations

### Model Performance
- Accuracy remains well short of what's typically expected for autonomous clinical decisions, at either the default or the tuned threshold
- The tuned threshold's low specificity (18% overall, under 1% for patients 55+) would generate a large number of unnecessary follow-ups if deployed as-is
- SHAP and permutation importance describe what the model relies on, not whether those relationships are causal

### Data and Scope
- Limited to 11 basic health features — no lab work, family history, or ECG data
- No temporal data (single point-in-time measurements)
- Dataset from a single source without external validation
- Subgroup analysis covers gender and age band only; the dataset has no race, ethnicity, or socioeconomic data, so a fuller fairness audit isn't possible here

### Clinical Considerations
- Model intended as an educational demonstration, not a clinical tool
- No regulatory approval or clinical validation
- Lacks integration with electronic health records
- No physician-in-the-loop design for real-world deployment

## Future Work

### Validation and Robustness
- External validation on a second, independently collected dataset
- Temporal validation using more recent patient data
- Recalibrating and re-tuning the threshold for a deployment population with different CVD prevalence than this dataset's roughly 50/50 split

### Deployment Considerations
- A physician-in-the-loop workflow where the model flags high-risk patients for review rather than making autonomous decisions
- Monitoring for model and data drift in an ongoing screening pipeline
- Address regulatory compliance (HIPAA, FDA medical device classification)
- A fuller fairness audit if demographic data beyond age and gender become available

## Author

**Joseph Williams**
- Graduate Student, Master of Data Science, University of North Texas
- LinkedIn: [linkedin.com/in/josephedgarwilliams](https://www.linkedin.com/in/josephedgarwilliams/)
- GitHub: [github.com/xjwllmsx](https://github.com/xjwllmsx)
- Email: hello@joseph-williams.me

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Dataset provided by [Kaggle user alamshihab075](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning)
- Project completed as part of graduate coursework in Principles of Data Science; later updated to a cross-validated, pipeline-based workflow with additional model families and interpretability
- Inspired by the potential for machine learning to support healthcare decision-making

## Citation

If you use this work, please cite:

```
@misc{cvd_prediction_2025,
  author = {Williams, Joseph},
  title = {Cardiovascular Disease Prediction Using Supervised Learning},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/xjwllmsx/cardio-ml-predictor}
}
```
