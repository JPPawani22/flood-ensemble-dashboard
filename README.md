# Sri Lanka Flood Risk Ensemble Learning Dashboard

An individual assignment project for comparing **bagging, boosting, and stacking** on a Sri Lankan context dataset. The included app predicts the continuous `flood_risk_score` from tabular location and environmental features. It compares Random Forest, XGBoost, LightGBM, CatBoost, and a stacking ensemble with Random Forest, XGBoost, and LightGBM as base learners and an ANN meta-learner.

> **Dataset caveat:** the included 25,000-row flood CSV is synthetic (the `is_synthetic` column is true). It is useful for demonstrating the ML workflow, but cannot support claims about real flood risk or operational warning decisions. Mention this in the presentation.

## What the project demonstrates

- **Bagging:** Random Forest averages trees fit on bootstrap samples to reduce variance.
- **Boosting:** XGBoost, LightGBM, and CatBoost add trees sequentially to correct earlier errors.
- **Stacking ensemble:** out-of-fold predictions from Random Forest, XGBoost, and LightGBM feed an ANN meta-learner. Stacking is the ensemble method; combining tree models with an ANN makes this particular stack a hybrid implementation.
- Shared train/test split and pipeline-based preprocessing; compare R², MSE, and RMSE.
- Dashboard dataset overview, held-out metrics, model-family explanation, Random Forest permutation importance, and single-row predictions.
- Upload a CSV, choose a numeric regression target, and train the same comparison on another tabular dataset.

This starter focuses on **tabular regression**. Customer churn is normally a classification task (for example, `Churn` Yes/No), so it needs a classification pipeline and metrics such as precision, recall, F1, ROC-AUC, and a confusion matrix; do not treat its labels as a regression target. Healthcare data may contain sensitive personal information: only use properly anonymized, authorized data.

## Project layout

```text
backend/
  preprocessing.py       Dynamic tabular preprocessing and leakage exclusions
  train_models.py        Training, comparison, and model bundle creation
  models/                Generated serialized models (after training)
dashboard/
  dashboard.py           Streamlit UI
data/
  sri_lanka_flood_risk_dataset_25000.csv
notebooks/
  01_model_training_and_evaluation.ipynb
output/                  Generated metrics and metadata
requirements.txt
```

## Setup and run

Use Python 3.10 or 3.11 in a virtual environment. From the project root:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m backend.train_models
streamlit run dashboard/dashboard.py
```

The first training run creates `backend/models/training_bundle.pkl`, the standalone `backend/models/stacking_model.pkl`, `output/model_comparison.csv`, and run metadata. `hybrid_stack_model.pkl` is retained as a backwards-compatible copy of the same stacking pipeline. Training includes five-fold stacking and may take a few minutes. Keep the virtual environment active when running the dashboard because the pickled models depend on the installed package versions.

### Train on another numeric-target CSV

```bash
python -m backend.train_models --data path/to/data.csv --target numeric_target
```

Optional flags: `--drop column_a column_b`, `--test-size 0.25`, `--seed 7`. The dashboard also allows CSV upload and target selection. Before training another dataset, review `FLOOD_LEAKAGE_COLUMNS` and `IDENTIFIER_COLUMNS` in `backend/preprocessing.py`; add any dataset-specific outcome leakage or identifier fields. Never include information that would not be known at prediction time.

The pipeline infers numeric and categorical predictors, imputes missing values, scales numeric columns, and one-hot encodes categories. Identifiers and known flood outcome fields are excluded. Unknown categories at prediction time are safely ignored. Numeric target rows that are blank or non-numeric are removed.

## Notebook guide

Open `notebooks/01_model_training_and_evaluation.ipynb` from Jupyter. Run its cells in order to inspect the dataset, prepare predictors, train and compare the models, inspect metrics, and create evaluation plots. The Python modules in `backend/` are the reusable implementation; the notebook calls the same training function so results do not drift between notebook and app.

## Evaluation and interpretation

- **R²:** proportion of variance explained; higher is better and can be negative on a poor test set.
- **MSE:** average of squared prediction errors; lower is better. It is in squared target units and penalizes large errors strongly.
- **RMSE:** square root of MSE, in target units; lower is better and large errors weigh more.
- Compare all metrics on the same held-out split. A model winning one random split does not establish generalization. For stronger coursework evidence, report repeated cross-validation or a time/group-aware split when rows share locations or dates.
- Permutation importance is a predictive association diagnostic, not causation. Correlated features can share or mask importance.
- The ANN stack is more complex and computationally expensive; it is only useful if its test performance improves and its gain justifies the cost.

### Tabular vs image/text data

Tree ensembles are strong on structured tables with mixed numeric/categorical fields, non-linear relationships, and moderate data sizes. Raw images and text are high-dimensional unstructured inputs: CNNs/vision transformers and language/transformer models are usually a better fit. Ensembles can still work on engineered image/text features plus metadata, but this dashboard does not ingest raw images or text.

## Dataset sources and alternatives

- **Included Sri Lankan flood data:** `data/sri_lanka_flood_risk_dataset_25000.csv` (synthetic; already included, no download required).
- **Sri Lanka Open Data portal:** [data.gov.lk](https://data.gov.lk/) — search for a relevant government dataset and inspect its license, fields, coverage, and provenance before use.
- **Hydro-meteorological information:** [Sri Lanka HMIS](https://hmis.gov.lk/) — historical and current hydro-meteorological information; confirm access conditions and export options for the specific data needed.
- **River level and flood-warning records:** [Lanka Data Foundation repository](https://github.com/nuuuwan/lk_dmc/tree/data_lk_dmc_river_water_level_and_flood_warnings/data/lk_dmc_river_water_level_and_flood_warnings) — inspect documentation and terms before treating records as model labels.
- **Telecom churn example:** [IBM Telco Customer Churn dataset listing](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) is widely used but represents a US service, not Sri Lankan subscribers. Kaggle login may be needed. It is classification, and requires adapting the backend before use.

## Suggested implementation steps for the assignment

1. State the research question, e.g. “How do bagging, boosting, and a hybrid stack compare for predicting flood-risk score from environmental and location data?”
2. Document dataset provenance, license, unit of analysis, features, target, missing values, and any synthetic/sensitive nature.
3. Identify and remove IDs, direct target proxies, and post-outcome fields before splitting. Explain the decision in the report.
4. Establish a reproducible hold-out split. Fit imputation, scaling, and encoding on training rows only using pipelines.
5. Train the RF bagging baseline, three boosting variants, and the stacking ensemble. Record R², MSE, RMSE and runtime if you measure it.
6. Inspect leaderboard, residual behavior, permutation importance, and examples of large errors. Explain plausible causes without claiming causation.
7. Discuss limitations: synthetic labels, random split assumptions, location/date leakage risks, generalization, model complexity, and lack of uncertainty bounds.
8. Run the dashboard, capture screenshots, and record a concise 3-minute walkthrough.

## Three-minute presentation outline

- **0:00–0:25:** Problem, Sri Lankan flood-risk context, and exact numeric prediction target.
- **0:25–0:55:** Dataset, feature examples, target, and the fact that the provided data is synthetic.
- **0:55–1:35:** Explain bagging vs boosting and show the RF, XGBoost, LightGBM, and CatBoost comparison.
- **1:35–2:05:** Explain the RF + boosting + ANN stacking hybrid and when it helps or adds cost.
- **2:05–2:40:** Demonstrate dashboard metrics, interpretation, feature importance, and one prediction.
- **2:40–3:00:** Best result on the held-out split, limitations, and next steps with real validated data.

## Hosting

For a simple demo, deploy the Streamlit app to a Python-capable host (for example, Streamlit Community Cloud) with `requirements.txt`, source code, and the trained bundle. Large pickle files may exceed free-host repository limits; retrain during deployment or use managed artifact storage if needed. Pickle files should only be loaded from trusted sources. Do not commit private patient or customer data.

## References

- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24, 123–140.
- Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS*.
