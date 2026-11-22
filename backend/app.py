"""Backend entry-point notes.

Model training and artifact creation live in :mod:`backend.train_models`:

    python -m backend.train_models --data path/to/data.csv --target numeric_target

The interactive application is the Streamlit module at
``dashboard/dashboard.py``; run it from the project root with
``streamlit run dashboard/dashboard.py``.
"""
