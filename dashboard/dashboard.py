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

st.set_page_config(page_title="Flood Risk | Ensemble Lab", page_icon="\U0001F327\uFE0F", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { color-scheme:dark; --ink:#e6edf5; --muted:#a0b2c5; --teal:#49c5c1; --line:#2b4055; --panel:#14283a; --field:#1b3247; }
.stApp { background:linear-gradient(180deg,#102235 0,#0b1724 380px,#0b1724 100%); color:var(--ink); }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3,h4,p,li,label,small { color:var(--ink); }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
.block-container { max-width:1400px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stSidebar"] { background:#101f30; border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); padding:1.05rem 1.2rem; border-radius:14px; box-shadow:0 8px 24px rgba(0,0,0,.18); min-height:112px; }
[data-testid="stMetricLabel"] { color:var(--muted); font-size:.8rem; }
[data-testid="stMetricValue"] { color:#f2f7fb; font-family:'Manrope',sans-serif; font-size:1.45rem; overflow-wrap:anywhere; }
[data-testid="stTabs"] [role="tab"] { font-weight:600; }
[data-testid="stTabs"] [aria-selected="true"] { color:var(--teal); }
.hero { display:block; width:100%; box-sizing:border-box; background:transparent; color:#f3fbfd; border:0; border-radius:0; padding:1.8rem 0 1.5rem; margin-bottom:1.35rem; box-shadow:none; overflow:visible; }
.hero h1 { color:#fff !important; font-size:clamp(2.7rem,5vw,4rem); line-height:1.08; margin:0 0 .55rem 0; }
.hero p { color:#d7edf1; margin:0; font-size:1rem; }
.eyebrow { display:block; text-transform:uppercase; letter-spacing:.12em; font-size:1rem; line-height:1.5; font-weight:700; color:#72e0d8; margin:0 0 .85rem; padding-top:.1rem; overflow:visible; }
div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:var(--panel); }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p { color:var(--muted) !important; }
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stDateInput"] input, [data-testid="stTextArea"] textarea { background:#1b3247 !important; color:#f2f7fb !important; border-color:#456078 !important; }
[data-baseweb="select"] *, [role="listbox"] *, [role="option"] { color:#f2f7fb !important; }
[role="listbox"], [data-baseweb="popover"] > div { background:#1b3247 !important; }
input::placeholder, textarea::placeholder { color:#a0b2c5 !important; }
button, [data-testid="stFileUploader"] section { border-color:#456078 !important; }
[data-testid="stButton"] button, [data-testid="stFormSubmitButton"] button { background:#1b3247; color:#f2f7fb; border:1px solid #456078; }
[data-testid="stButton"] button:hover, [data-testid="stFormSubmitButton"] button:hover { border-color:#49c5c1; color:#ffffff; }
button[kind="primary"] { background:#087f8c !important; color:#ffffff !important; border-color:#49c5c1 !important; }
[data-testid="stFileUploader"] section { background:#14283a; }
[data-testid="stAlert"] { border:1px solid var(--line); }
[data-testid="stTabs"] button { color:#a0b2c5; }
hr { border-color:var(--line); }
a { color:#70d5d0 !important; }
</style>
<section class="hero"><div class="eyebrow">Ensemble learning lab | Index 224202B</div>
<h1>Sri Lanka Flood Risk</h1><p>Compare bagging, boosting, and a neural stacking hybrid on structured data.</p></section>
""", unsafe_allow_html=True)

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
    metric = st.selectbox("Compare metric", ["RMSE", "MSE", "R2"])
    ranked = results.sort_values(metric, ascending=(metric != "R2"))
    winner = ranked.iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Top model | " + metric, str(winner["Model"]))
    m2.metric("Best " + metric, f"{winner[metric]:.3f}")
    m3.metric("Held-out rows", f"{len(bundle['test_actual']):,}" if bundle else "N/A")
    st.dataframe(results.style.format({"R2":"{:.3f}", "MSE":"{:.3f}", "RMSE":"{:.3f}"})
                 .set_properties(**{"background-color":"#14283a", "color":"#e6edf5"})
                 .highlight_max(subset=["R2"], props="background-color:#174b42; color:#ffffff; font-weight:700;")
                 .highlight_min(subset=["MSE", "RMSE"], props="background-color:#174b42; color:#ffffff; font-weight:700;"),
                 use_container_width=True, hide_index=True)
    st.markdown("#### Model ranking")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#14283a")
    bars = ax.barh(ranked["Model"], ranked[metric], color=["#35d0c5" if x == ranked.iloc[0]["Model"] else "#40586d" for x in ranked["Model"]], height=.62)
    ax.set_facecolor("#14283a")
    ax.invert_yaxis()
    ax.set_xlabel(metric + (" (higher is better)" if metric == "R2" else " (lower is better)"), color="#a0b2c5")
    ax.grid(axis="x", color="#30475b", linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#456078")
    ax.tick_params(axis="y", length=0, colors="#f2f7fb", labelsize=10)
    ax.tick_params(axis="x", colors="#c3d3e0", labelsize=9)
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")
    ax.bar_label(bars, fmt="%.3f", padding=7, color="#ffffff", fontsize=10, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
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
                fig, ax = plt.subplots(figsize=(8, 4), facecolor="#14283a")
                ax.set_facecolor("#14283a")
                bars = ax.barh(imp_df["Feature"], imp_df["Importance"], color="#087f8c", height=.62)
                ax.set_xlabel("Mean R\u00b2 drop when shuffled", color="#a0b2c5")
                ax.grid(axis="x", color="#30475b", linewidth=.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0, colors="#f2f7fb", labelsize=9)
                for label in ax.get_yticklabels():
                    label.set_fontweight("semibold")
                ax.bar_label(bars, fmt="%.3f", padding=5, fontsize=9, color="#ffffff", fontweight="bold")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
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
