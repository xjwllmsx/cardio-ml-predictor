# Cardiovascular Disease Prediction Using Machine Learning

A machine learning project comparing classification algorithms for predicting cardiovascular disease from patient health metrics.

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

This project develops and evaluates binary classification models to predict cardiovascular disease (CVD) presence based on patient health data. The analysis demonstrates a complete supervised learning workflow, from data preprocessing through model comparison and evaluation, with careful consideration of medical context and error costs.

## Problem Statement

Cardiovascular disease is one of the leading causes of death globally. Early detection through routine health screenings could enable preventive interventions and improve patient outcomes. This project explores whether machine learning can effectively predict CVD using commonly collected health metrics such as blood pressure, cholesterol levels, and lifestyle factors.

**Research Questions:**
- Can machine learning models accurately predict CVD from basic health metrics?
- Which classification algorithm performs best for this medical prediction task?
- How does hyperparameter tuning impact model performance?
- What are the implications of prediction errors in a medical screening context?

## Dataset

**Source:** [Heart Failure Diagnosis Data for Machine Learning](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning) (Kaggle)

**Size:** 70,000 patient records

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

**Class Distribution:** Balanced (50% CVD, 50% healthy)

## Methodology

### 1. Data Preparation
- Exploratory data analysis to understand feature distributions and data quality
- Train-test split (70-30) with stratification to maintain class balance
- Removal of non-predictive ID column

### 2. Feature Preprocessing
- Standardization using StandardScaler (mean = 0, std = 1)
- Critical for K-Nearest Neighbors algorithm performance
- Scaler fit on training data only to prevent data leakage

### 3. Model Development
**Algorithms Tested:**
- K-Nearest Neighbors (KNN) with hyperparameter tuning
- Logistic Regression

**Hyperparameter Tuning:**
- Tested KNN with K=5 and K=9
- Selected optimal K value based on test set performance

### 4. Evaluation Metrics
- Confusion Matrix (visual and numerical)
- Accuracy
- Precision
- Recall
- F1 Score

All metrics calculated on held-out test set to assess generalization performance.

## Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| KNN (K=5) | 65.8% | 66.5% | 63.6% | 65.0% |
| KNN (K=9) | 67.8% | 69.2% | 64.1% | 66.5% |
| Logistic Regression | 72.2% | 74.4% | 67.7% | 70.9% |

### Key Outcomes

**Best Performing Model:** Logistic Regression
- 72.2% accuracy on test set (15,167 correct predictions out of 21,000)
- 4.4 percentage point improvement over tuned KNN
- 555 fewer false positives than KNN (K=9)
- 217 fewer false negatives than KNN (K=9)

**Hyperparameter Tuning Impact:**
- Increasing K from 5 to 9 improved KNN accuracy by 2.0 percentage points
- Demonstrates the value of systematic hyperparameter optimization

**Medical Context:**
- False negative rate: 24% (2,552 CVD cases missed)
- False positive rate: 24% (2,480 healthy patients incorrectly flagged)
- Current performance insufficient for autonomous clinical decision-making

## Technologies Used

- **Python 3.12**
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and evaluation
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - Interactive development environment

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

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option B: Using pip (Traditional Method)

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### Required Packages

The `requirements.txt` file contains:
```
ipykernel>=7.1.0
jinja2>=3.1.6
matplotlib>=3.10.8
pandas>=2.3.3
scikit-learn>=1.8.0
```

## Usage

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/alamshihab075/heart-failure-diagnosis-data-for-machine-learning)

2. Place the CSV file in the project root directory as `HeartFailureDataset.csv`

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open `cardio-ml-predictor.ipynb` and run all cells

The notebook will execute the complete pipeline:
- Load and explore the data
- Preprocess features
- Train multiple models
- Generate evaluation metrics and visualizations
- Display model comparisons

## Project Structure

```
cardio-ml-predictor/
│
├── cardio-ml-predictor.ipynb        # Main analysis notebook
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

### Algorithm Selection Matters

Logistic Regression significantly outperformed K-Nearest Neighbors, suggesting that the relationship between health metrics and CVD risk is relatively linear. This aligns with medical understanding where factors like blood pressure and cholesterol typically show monotonic relationships with disease risk.

### Hyperparameter Tuning Provides Incremental Gains

Optimizing K from 5 to 9 improved KNN performance by 2 percentage points. While modest, this demonstrates the importance of systematic hyperparameter search rather than accepting default values.

### Multiple Evaluation Metrics Are Essential

In medical applications, accuracy alone is insufficient. The confusion matrix reveals that both models miss approximately 24% of CVD cases (false negatives), which would be unacceptable in clinical screening where false negatives could delay critical treatment.

### Medical AI Requires Higher Standards

The 72% accuracy achieved by Logistic Regression, while respectable for a baseline model, highlights the gap between academic performance and clinical deployment requirements. Medical decision support systems typically require 90%+ accuracy with very low false negative rates.

## Limitations

### Model Performance
- 28% overall error rate too high for autonomous clinical decisions
- 24% false negative rate could lead to missed diagnoses
- No analysis of performance across demographic subgroups
- Single train-test split rather than cross-validation

### Data and Scope
- Limited to 11 basic health features
- No temporal data (single point-in-time measurements)
- Dataset from single source without external validation
- Potential sampling bias if data not representative of diverse populations

### Clinical Considerations
- Model intended as educational demonstration, not clinical tool
- No regulatory approval or clinical validation
- Lacks integration with electronic health records
- No physician-in-the-loop design for real-world deployment

## Future Work

### Model Improvements
- Test ensemble methods (Random Forest, Gradient Boosting)
- Implement neural networks for potential non-linear pattern detection
- Feature engineering (BMI calculation, age groups, interaction terms)
- Feature importance analysis to identify most predictive health metrics
- Threshold optimization to balance precision and recall

### Validation and Robustness
- K-fold cross-validation for more reliable performance estimates
- External validation on independent datasets
- Temporal validation using more recent patient data
- Fairness analysis across demographic groups
- Calibration analysis for probability predictions

### Deployment Considerations
- Develop model interpretability tools (SHAP values, LIME)
- Design clinical decision support interface
- Implement real-time monitoring for model drift
- Address regulatory compliance (HIPAA, FDA medical device classification)
- Create physician feedback loop for continuous improvement

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
- Project completed as part of graduate coursework in Principles of Data Science
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