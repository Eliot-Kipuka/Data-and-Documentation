SHAP supplement aligned to the current manuscript

The four model outcomes are Y1 HER, Y2 HSR, Y3 LER and Y4 LSR. Results contain 714 held-out observations per category, signed SHAP values, base values, predictions and observed counts. All four categories share one input table, results/600m_shap_inputs.csv. The four formerly identical input tables have been consolidated; no observations have been removed.

Saved predictions reproduce current Table 3; SHAP contributions sum to prediction minus base value within numerical precision. Mean absolute SHAP is not the normalized gain percentage in Table 4. LOWESS is a descriptive smoother; fallback iterations are documented in shap_smoothing_checks.csv.

Run python run_shap.py with the sibling O2O_reproducibility_notebook present and its recorded dependencies installed. It loads that package's saved models and input data; no independent copy of the model code or input data is required here. It does not rerun the parameter searches. KDE checks are in the response letter.
