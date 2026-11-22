"""Streamlit dashboard for comparing tabular regression ensembles."""
from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from backend.preprocessing import prepare_features
from backend.train_models import train

DATA_PATH = ROOT / "data" / "sri_lanka_flood_risk_dataset_25000.csv"
BUNDLE_PATH = ROOT / "backend" / "models" / "training_bundle.pkl"

st.set_page_config(page_title="Sri Lanka Flood Risk | Ensemble Lab", page_icon="🌧️", layout="wide")
st.title("Sri Lanka Flood Risk · Ensemble Learning Lab")
st.caption("Compare bagging, boosting, and an ANN based stacking hybrid on tabular data.")

with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV for a new regression run", type="csv")
    if uploaded is not None:
        source = pd.read_csv(uploaded)
        st.success(f"Loaded {len(source):,} rows")
        target_options = source.columns.tolist()
        suggested = target_options.index("flood_risk_score") if "flood_risk_score" in target_options else len(target_options)-1
        target = st.selectbox("Numeric target", target_options, index=suggested)
        st.caption("For flood risk, keep outcome/leakage columns excluded. Add any dataset-specific leakage fields to the exclusion list in backend/preprocessing.py.")
        if st.button("Train and compare models", type="primary"):
            with st.spinner("Training five models. Stacking uses cross-validation and may take a few minutes…"):
                try:
                    st.session_state["bundle"] = train(source, target)
                    st.session_state["run_bundle"] = joblib.load(BUNDLE_PATH)
                    st.session_state["run_data"] = source
                    st.success("Training complete.")
                except Exception as exc:
                    st.error(f"Training could not complete: {exc}")
    else:
        source = pd.read_csv(DATA_PATH)
        st.caption("Using the included Sri Lankan flood risk CSV.")

if "run_bundle" not in st.session_state and BUNDLE_PATH.exists() and uploaded is None:
    st.session_state["run_bundle"] = joblib.load(BUNDLE_PATH)
    st.session_state["run_data"] = source

if uploaded is None and "run_bundle" not in st.session_state:
    st.warning("Train the included models first: `python -m backend.train_models`. Then reload this page.")
    st.stop()
if uploaded is not None and "run_bundle" not in st.session_state:
    st.info("Choose a numeric target and click **Train and compare models** to create this dataset's results.")
    st.stop()

bundle = st.session_state.get("run_bundle")
if bundle is not None:
    run_data = st.session_state.get("run_data", source)
    target_name = bundle["target"]
    results = bundle["comparison"]
else:
    run_data = source
    target_name = "flood_risk_score"
    results = pd.read_csv(ROOT / "output" / "model_comparison.csv")

if target_name in run_data.columns:
    top = run_data[target_name]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(run_data):,}")
    c2.metric("Predictor columns", str(len(bundle["feature_columns"]) if bundle else run_data.shape[1]-1))
    c3.metric("Target", target_name)
    c4.metric("Target mean", f"{pd.to_numeric(top, errors='coerce').mean():.2f}")

tab_compare, tab_explain, tab_predict, tab_about = st.tabs(
    ["Model comparison", "Interpretation", "Prediction", "Dataset & method"])

with tab_compare:
    st.subheader("Held-out test set results")
    st.caption("R² higher is better. MSE and RMSE lower are better; both penalize large errors.")
    st.dataframe(results.style.format({"R2":"{:.3f}", "MSE":"{:.3f}", "RMSE":"{:.3f}"}), use_container_width=True)
    
    metric = st.selectbox("Compare metric", ["RMSE", "MSE", "R2"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ranked = results.sort_values(metric, ascending=(metric != "R2"))
    bars = ax.barh(ranked["Model"], ranked[metric], color=["#0f766e" if x == ranked.iloc[0]["Model"] else "#94a3b8" for x in ranked["Model"]])
    ax.set_xlabel(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"))
    ax.grid(axis="x", alpha=.2)
    st.pyplot(fig)
    best = results.sort_values("R2", ascending=False).iloc[0]
    st.info(f"Best R² in this held-out split: **{best['Model']}** (R² {best['R2']:.3f}, RMSE {best['RMSE']:.3f}). This describes this split only; it is not evidence of performance on all Sri Lankan locations.")

with tab_explain:
    st.subheader("Why do the models behave differently?")
    st.markdown("""
| Family | This project | How it learns | Strengths | Limitations |
|---|---|---|---|---|
| Bagging | Random Forest | Fits trees on bootstrap samples, averages their predictions | Robust baseline; reduces variance; captures nonlinear tabular patterns | Can be large; less effective at extrapolating beyond training ranges |
| Boosting | XGBoost, LightGBM, CatBoost | Adds trees sequentially to correct previous errors | Often excellent on structured/tabular data; captures interactions | Needs tuning; can overfit noisy or small data; less transparent than a single tree |
| Stacking | RF + XGBoost + LightGBM → ANN | Learns a meta-model from out-of-fold base predictions | Can combine complementary error patterns | More training cost and complexity; may not beat best base model |

For text and image inputs, these models need a meaningful feature representation first. CNNs and vision transformers are generally better suited to raw images; transformer or language models usually suit raw text. Tree ensembles remain useful with engineered text/image features and small structured metadata. This dashboard accepts **tabular regression CSVs** with a numeric target.
""")
    if bundle:
        st.markdown("#### Feature signal (Random Forest permutation importance)")
        st.caption("Importance measures test-score drop after shuffling a feature. It indicates association, not causation.")
        if st.button("Calculate feature importance"):
            try:
                from sklearn.inspection import permutation_importance
                model = bundle["models"]["Random Forest (Bagging)"]
                raw_X = bundle["test_features"]
                raw_y = bundle["test_actual"]
                # Bound the interactive calculation so larger uploads remain usable.
                if len(raw_X) > 1000:
                    raw_X = raw_X.sample(1000, random_state=42)
                    raw_y = raw_y.loc[raw_X.index]
                imp = permutation_importance(model, raw_X, raw_y, n_repeats=2,
                                             random_state=42, scoring="r2", n_jobs=-1)
                imp_df = pd.DataFrame({"Feature": raw_X.columns, "Importance": imp.importances_mean}).nlargest(12, "Importance").sort_values("Importance")
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(imp_df["Feature"], imp_df["Importance"], color="#0f766e")
                ax.set_xlabel("Mean R² drop when shuffled")
                st.pyplot(fig)
                if len(imp_df):
                    st.write(f"The strongest measured signal is **{imp_df.iloc[-1]['Feature']}**. Check whether this makes domain sense and whether it is available at prediction time.")
            except Exception as exc:
                st.caption(f"Feature importance is unavailable for this run: {exc}")
    st.warning("The included flood data appears synthetic (`is_synthetic` is true). Results are an educational demonstration, not a real flood warning or a causal explanation.")

with tab_predict:
    st.subheader("Estimate a target value for one row")
    if not bundle:
        st.info("Load a trained model bundle to use prediction.")
    else:
        model_name = st.selectbox("Model", list(bundle["models"].keys()))
        feature_cols = bundle["feature_columns"]
        defaults = run_data.drop(columns=[target_name], errors="ignore")
        # Remove excluded fields exactly as training did, then show safe editable inputs.
        defaults = defaults.drop(columns=bundle["dropped_columns"], errors="ignore")
        defaults = defaults.reindex(columns=feature_cols)
        row = {}
        with st.form("predict_form"):
            cols = st.columns(2)
            for i, name in enumerate(feature_cols):
                series = defaults[name]
                if pd.api.types.is_numeric_dtype(series):
                    value = float(pd.to_numeric(series, errors="coerce").median())
                    row[name] = cols[i % 2].number_input(name, value=value)
                else:
                    options = series.dropna().astype(str).value_counts().index.tolist()
                    row[name] = cols[i % 2].selectbox(name, options or ["Unknown"])
            submit = st.form_submit_button("Predict")
        if submit:
            try:
                estimate = bundle["models"][model_name].predict(pd.DataFrame([row]))[0]
                st.success(f"Predicted {target_name}: **{estimate:.3f}**")
                st.caption("A point estimate from the selected model; no uncertainty interval is calculated.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

with tab_about:
    st.subheader("Dataset overview")
    st.write(f"Current data: **{len(run_data):,} rows × {run_data.shape[1]} columns**. Target: **{target_name}**.")
    st.dataframe(run_data.head(10), use_container_width=True)
    if bundle:
        st.write("Excluded from predictors:", ", ".join(bundle["dropped_columns"]) or "none")
        st.write(f"Train/test: {1-bundle['test_size']:.0%}/{bundle['test_size']:.0%}, random seed {bundle['split_seed']}.")
    st.markdown("#### Model workflow")
    st.write("A held-out random split is shared across models. Preprocessing is fitted inside each pipeline using training data only. The stack's base predictions are generated with five-fold cross-validation before fitting its ANN meta-learner.")
