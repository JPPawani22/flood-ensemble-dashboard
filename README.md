# Sri Lanka Flood Risk Dashboard

A Streamlit dashboard for tabular regression. It compares Random Forest, XGBoost, LightGBM, CatBoost, and a stacking model that combines tree predictions with an ANN. The dashboard shows held-out R², MSE, and RMSE, feature importance, and single-row predictions.

> **Dataset note:** The included flood-risk dataset is synthetic. Results are for demonstrating the machine-learning workflow and should not be used as real flood forecasts.

## Run the system

Requirements: Python 3.10 or 3.11.

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Train the models and start the dashboard:

```powershell
python -m backend.train_models
streamlit run dashboard/dashboard.py
```

Open the local URL printed by Streamlit in your browser. Model training, including the stacking model, may take a few minutes. Keep the environment active while using the dashboard.

## Project structure

```text
backend/
  preprocessing.py       Data preparation
  train_models.py        Model training and evaluation
dashboard/
  dashboard.py           Streamlit dashboard
notebooks/
  01_model_training_and_evaluation.ipynb
data/
  sri_lanka_flood_risk_dataset_25000.csv
output/                  Evaluation charts and results
requirements.txt         Python dependencies
```

## Use another dataset

Upload a CSV in the dashboard, select a numeric target, and choose **Train and compare models**.

Alternatively, train from the command line:

```powershell
python -m backend.train_models --data path/to/data.csv --target target_column
```

This project supports tabular regression. Review `backend/preprocessing.py` to ensure dataset-specific identifiers or target leakage fields are excluded before training.

## Support of Gen AI tools

The project was implemented with the support of the GEN AI tools (GPT).
All the test runs and values validation are done by only myself.