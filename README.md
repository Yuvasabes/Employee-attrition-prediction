# Employee Attrition Prediction using Machine Learning

Predicts whether an employee is likely to leave the company, using HR
features such as job satisfaction, income, overtime, tenure, and more.

## Project Structure

```
attrition_project/
├── data/
│   ├── employee_data.csv       # training dataset
│   └── new_employees.csv       # sample data for scoring (demo)
├── models/
│   └── attrition_model.pkl     # trained pipeline (preprocessing + classifier)
├── outputs/
│   ├── metrics.json            # model comparison results
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   └── predictions.csv         # sample scored output
├── src/
│   ├── generate_dataset.py     # creates the synthetic dataset (skip if using real data)
│   ├── preprocessing.py        # cleaning, encoding, scaling, train/test split
│   ├── train.py                # trains & compares models, saves the best one
│   ├── evaluate.py             # generates evaluation plots
│   └── predict.py              # scores new employees for attrition risk
├── requirements.txt
└── README.md
```

## About the Dataset

This project ships with a **synthetic dataset** (1,470 rows) generated to
mirror the structure and realistic correlations of the well-known IBM HR
Analytics Employee Attrition dataset (same columns: Age, JobSatisfaction,
OverTime, MonthlyIncome, YearsAtCompany, etc.), with an overall attrition
rate of ~16%, matching real-world HR benchmarks.

**To use your own company data instead:** just replace
`data/employee_data.csv` with your real HR export. Keep an `Attrition`
column with `Yes`/`No` values as the target. Column names don't need to
match exactly — `preprocessing.py` automatically detects numeric vs.
categorical columns and builds the pipeline accordingly.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset (skip if you have your own)
```bash
python3 src/generate_dataset.py
```

### 2. Train models
Trains Logistic Regression, Random Forest, and Gradient Boosting, compares
them with 5-fold cross-validation and a held-out test set, and saves the
best one automatically.
```bash
python3 src/train.py
```

### 3. Evaluate
Generates a confusion matrix, ROC curve, and feature importance chart.
```bash
python3 src/evaluate.py
```

### 4. Predict on new employees
```bash
python3 src/predict.py --input data/new_employees.csv --output outputs/predictions.csv
```
Output includes:
- `Attrition_Risk_Probability` — model's predicted probability of leaving
- `Predicted_Attrition` — Yes/No classification
- `Risk_Level` — Low / Medium / High bucket, useful for HR to prioritize retention conversations

You can also score a single employee via a JSON file:
```bash
python3 src/predict.py --input data/single_employee.json
```

### 5. Launch the interactive dashboard
```bash
streamlit run src/dashboard.py
```
This opens a browser tab (usually `http://localhost:8501`) with:
- **Model Overview tab** — metrics table, confusion matrix, ROC curve, feature importance (all rendered live, no need to open PNGs manually)
- **Upload & Predict tab** — drag-and-drop a CSV of employees and get color-coded risk predictions instantly, with a download button
- **Single Employee Calculator tab** — sliders/dropdowns to build a hypothetical employee profile and see their predicted attrition risk update live

## Current Model Performance

| Model               | CV ROC-AUC | Test Accuracy | Test F1 | Test ROC-AUC |
|---------------------|-----------:|---------------:|--------:|-------------:|
| Logistic Regression  | 0.623 | 0.643 | 0.323 | **0.667** (best) |
| Random Forest        | 0.576 | 0.844 | 0.042 | 0.650 |
| Gradient Boosting    | 0.543 | 0.830 | 0.039 | 0.616 |

Logistic Regression was selected as the best model — despite lower raw
accuracy, it has much better recall on the minority "Left" class (the one
that actually matters for HR intervention), whereas the tree models
achieve high accuracy mostly by predicting "Stayed" for almost everyone.

> Note: these numbers reflect the synthetic dataset. Once you plug in real
> company data, re-run `train.py` — performance will reflect genuine
> patterns in your workforce and should improve, since the synthetic
> data's signal is intentionally noisy/moderate.

## Next Steps / Extensions
- Add SHAP values for per-employee explainability
- Hyperparameter tuning (GridSearchCV / Optuna)
- Try XGBoost / LightGBM if available in your environment
- Build a simple Streamlit/Flask dashboard for HR to explore predictions
- Track model performance over time as new attrition data comes in
