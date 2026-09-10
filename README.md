# Industrial Data Curation & Quality Tool

A Streamlit prototype simulating an internal data-foundry workflow: ingest
noisy industrial IoT/maintenance data, profile its quality, and run it
through a human-in-the-loop AI-assisted curation queue.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
app.py                  # Streamlit UI — 3 tabs (Data Health, Duplicate
                         # Review, AI Curation Queue), session_state-driven
src/data_generator.py   # Synthetic industrial dataset with injected
                         # duplicates, missing values, noisy text, anomalies
src/quality_checks.py   # Missingness / exact & near-duplicate detection /
                         # per-equipment-type z-score anomaly detection
src/ai_suggestions.py   # Simulated AI/LLM normalization + categorization
                         # engine (rule + fuzzy-match based), with a stable
                         # interface a real LLM call could later drop into
requirements.txt
```

See the accompanying chat response for the full architecture explanation
and a guide on framing this project's business value in an interview.
