# O2O model reproducibility supplement

This package replaces the earlier supplement whose XGBoost and spatial-validation numbers did not match the current manuscript and response. The four original notebooks are preserved unchanged in original_notebooks.

Outcomes are shop counts: Y1 HER, Y2 HSR, Y3 LER and Y4 LSR. The selected 600 m sample contains 3,570 grids and 20,543 shops. Category-specific zero counts are retained. The 1,200 m sample has 1,648 aggregation units.

The substantive source correction is X17 at FID 1690, changed from 0 to the final 600 m model input value 472.222222222222. The corresponding coarse-cell value changes from 114.58333325 to 232.6388888055555. See results/data_correction.csv. Non-additive coarse indicators are summaries of existing fine-cell indicators, not fresh POI calculations.

Model settings explicitly use histogram trees, seed 0 and a base score equal to the mean training outcome. These settings reproduce the archived notebook scores and gain importance; the original software version is not claimed to have been recovered. Train/test split seed is 42. Spatial folds are supplied explicitly to avoid GroupKFold tie-handling changes between library versions. Spatial blocks are validation groups for 600 m models, not new 3/5 km shop-count models.

Open O2O_robustness_analysis.ipynb from this directory and run its cells in order. Or run `python code/run_analysis.py`; add `--search` only to repeat the original candidate searches. The original Y4 candidate grid contains one combination. Install requirements.txt in the recorded Python 3.7 environment. KDE maps and sensitivity checks are supplied in the response letter and are outside this package.

results contains baseline model JSON files, XGBoost/OLS holdout predictions, original gain importance, per-fold random/spatial predictions and metrics, scale sensitivity, rank stability and VIF/tolerance. Fivefold gain summaries are distinct from Table 4 original-fit gain scores; mean absolute SHAP is also a separate measure. Fold SDs are not confidence intervals.

The HTML is a read-only view of the notebook. Model-cell outputs retain their earlier recorded execution; the final numerical audit is independently executed during this packaging operation. No fresh training run is claimed by the packaging audit. `python code/validate_support.py` needs only numpy and pandas and checks model JSON inference, OLS fitting, saved predictions, data alignment, folds, numerical summaries and current manuscript/response reference values.

SHAP results/figures are in the sibling O2O_SHAP_supplement. The main analysis also regenerates SHAP outputs under this directory. The sibling SHAP wrapper can regenerate directly from the saved baseline model JSON files. Geometry and fold files are in the sibling O2O_spatial_grids. Keep the three directories together when running the cross-package audit.
