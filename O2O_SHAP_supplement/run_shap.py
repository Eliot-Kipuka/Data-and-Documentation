from pathlib import Path
import sys
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
CORE=ROOT.parent/'O2O_reproducibility_notebook'
sys.path.insert(0,str(CORE/'code'))
import run_analysis as analysis
fine=pd.read_csv(CORE/'data'/'600m_model_input.csv')
test_ids=pd.read_csv(CORE/'results'/'600m_holdout_assignments.csv')
test_ids=set(test_ids.loc[test_ids.subset=='test','FID'])
test_rows=np.flatnonzero(fine.FID.isin(test_ids).values)
models={}
for target in analysis.TARGETS:
    model=analysis.estimator(target)
    model.load_model(str(CORE/'results'/("600m_"+target+"_model.json")))
    models[target]=(model,test_rows)
analysis.ROOT=ROOT
analysis.OUT=ROOT/'results'
analysis.shap_analysis(fine,models)
common=ROOT/'results'/'600m_Y1_shap_inputs.csv'
common.replace(ROOT/'results'/'600m_shap_inputs.csv')
for target in ['Y2','Y3','Y4']:
    (ROOT/'results'/('600m_'+target+'_shap_inputs.csv')).unlink()
