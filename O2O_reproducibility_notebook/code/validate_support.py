"""Independent numerical audit; no XGBoost installation or training required."""
from pathlib import Path
import json, hashlib, ast, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
FEATURES=['X1','X2','X3','X4','X6','X7','X8','X19','X10','X14','X17','X18','X22']
TARGETS=['Y1','Y2','Y3','Y4']
checks=[]
def check(name,passed,details=''):
    checks.append(dict(check=name,passed=bool(passed),details=str(details)))
    if not passed: raise AssertionError(name+': '+str(details))
def scores(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    mse=np.mean((y-p)**2)
    return dict(r2=1-np.sum((y-p)**2)/np.sum((y-y.mean())**2),mse=mse,rmse=np.sqrt(mse),mae=np.mean(np.abs(y-p)))
def infer_json(learner,x):
    trees=learner['gradient_booster']['model']['trees']
    x=np.asarray(x,dtype=np.float32)
    output=np.full(len(x),float(learner['learner_model_param']['base_score']),dtype=np.float32)
    for tree in trees:
        for k,row in enumerate(x):
            node=0
            while tree['left_children'][node]!=-1:
                value=row[tree['split_indices'][node]]
                left=bool(tree['default_left'][node]) if np.isnan(value) else value<np.float32(tree['split_conditions'][node])
                node=tree['left_children'][node] if left else tree['right_children'][node]
            output[k]+=np.float32(tree['split_conditions'][node])
    return output

fine=pd.read_csv(ROOT/'data'/'600m_model_input.csv')
coarse=pd.read_csv(ROOT/'data'/'1200m_model_input.csv')
source=pd.read_csv(ROOT/'data'/'source_600m_grid_cells.csv')
check('600m and 1200m sample sizes',len(fine)==3570 and len(coarse)==1648)
check('shop counts and nonnegative integer outcomes',np.array_equal(fine[TARGETS].sum().values,[4422,9968,3923,2230]) and np.all(fine[TARGETS].values>=0) and np.all(fine[TARGETS].values%1==0))
check('600m/1200m count conservation',np.allclose(fine[TARGETS].sum(),coarse[TARGETS].sum()))
check('source-model predictors aligned',np.allclose(source.set_index('FID').loc[fine.FID,FEATURES],fine[FEATURES],rtol=0,atol=5.1e-7))
check('FID1690 corrected',np.isclose(source.loc[source.FID==1690,'X17'].iloc[0],472.222222222222))
check('coarse correction aligned',np.isclose(coarse.loc[(coarse.gx==33)&(coarse.gy==52),'X17'].iloc[0],232.6388888055555))
tree=ast.parse((ROOT/'code'/'run_analysis.py').read_text(encoding='utf-8'))
aggregation=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='aggregate_1200')
exec(compile(ast.fix_missing_locations(ast.Module(body=[aggregation],type_ignores=[])),'aggregation','exec'))
regen=aggregate_1200(source)
check('1200m aggregation reproduces supplied inputs',np.allclose(regen[FEATURES+TARGETS],coarse[FEATURES+TARGETS],rtol=1e-8,atol=1e-8))
split=pd.read_csv(RESULTS/'600m_holdout_assignments.csv')
train_ids=split.loc[split.subset=='train','FID'].values
test_ids=split.loc[split.subset=='test','FID'].values
check('holdout separation',len(train_ids)==2856 and len(test_ids)==714 and not(set(train_ids)&set(test_ids)))
check('holdout covers all model FIDs',set(split.FID)==set(fine.FID))
indexed=fine.set_index('FID')
expected_r2=[.4169,.5371,.0938,.1262]
expected_mse=[2.5945,9.7311,20.2466,13.9613]
expected_rmse=[1.6107,3.1195,4.4996,3.7365]
baseline=pd.read_csv(RESULTS/'600m_model_comparison.csv')
gain=pd.read_csv(RESULTS/'600m_gain_importance.csv')
vif=[]
x=fine[FEATURES].values.astype(float)
x=(x-x.mean(axis=0))/x.std(axis=0)
for j,name in enumerate(FEATURES):
    regressors=np.column_stack([np.ones(len(x)),np.delete(x,j,axis=1)])
    residual=x[:,j]-regressors@np.linalg.lstsq(regressors,x[:,j],rcond=None)[0]
    tolerance=np.sum(residual**2)/np.sum(x[:,j]**2)
    vif.append(dict(feature=name,tolerance=tolerance,VIF=1/tolerance))
vf=pd.DataFrame(vif)
vf.to_csv(RESULTS/'vif_tolerance.csv',index=False)
check('VIF/tolerance manuscript thresholds',vf.VIF.max()<10 and vf.tolerance.min()>.1, 'max VIF %.6f; min tolerance %.6f'%(vf.VIF.max(),vf.tolerance.min()))
for i,target in enumerate(TARGETS):
    pred=pd.read_csv(RESULTS/('600m_'+target+'_holdout_predictions.csv'))
    check(target+' test FIDs and observations',set(pred.FID)==set(test_ids) and np.array_equal(pred.observed,indexed.loc[pred.FID,target]))
    learner=json.loads((RESULTS/('600m_'+target+'_model.json')).read_text())['learner']
    native=infer_json(learner,indexed.loc[pred.FID,FEATURES].values)
    check(target+' saved XGBoost JSON reproduces predictions',np.allclose(native,pred.XGBoost_predicted,rtol=0,atol=2e-5),np.max(np.abs(native-pred.XGBoost_predicted)))
    avg=np.zeros(len(FEATURES)); counts=np.zeros(len(FEATURES))
    for t in learner['gradient_booster']['model']['trees']:
        for node,left in enumerate(t['left_children']):
            if left!=-1:
                j=t['split_indices'][node];avg[j]+=t['loss_changes'][node];counts[j]+=1
    normalized=np.divide(avg,counts,out=np.zeros_like(avg),where=counts>0)
    normalized/=normalized.sum()
    saved=gain[gain.category==target].set_index('feature').loc[FEATURES]
    check(target+' gain derived from model JSON',np.allclose(normalized,saved.gain_importance,rtol=2e-5,atol=1e-7))
    ranks=pd.Series(normalized).rank(ascending=False,method='min').values
    check(target+' gain ranks',np.array_equal(ranks,saved['rank'].values))
    for kind,column in [('XGBoost','XGBoost_predicted'),('OLS','OLS_predicted')]:
        stats=scores(pred.observed,pred[column]);savedrow=baseline[(baseline.category==target)&(baseline.model==kind)].iloc[0]
        check(target+' '+kind+' metrics from predictions',all(np.isclose(stats[k],savedrow[k],rtol=1e-8,atol=1e-8) for k in stats))
        if kind=='XGBoost':check(target+' current manuscript Table3',round(stats['r2'],4)==expected_r2[i] and round(stats['mse'],4)==expected_mse[i] and round(stats['rmse'],4)==expected_rmse[i])
    # Independent OLS fit with centering, matching an intercept-enabled regression.
    train=indexed.loc[train_ids,FEATURES].values.astype(float); response=indexed.loc[train_ids,target].values.astype(float)
    center=train.mean(axis=0);coeff=np.linalg.lstsq(train-center,response-response.mean(),rcond=None)[0]
    intercept=float(response.mean()-center@coeff)
    ols=indexed.loc[pred.FID,FEATURES].values@coeff+intercept
    check(target+' OLS independently fitted',np.allclose(ols,pred.OLS_predicted,rtol=1e-7,atol=1e-6))
    (RESULTS/('600m_'+target+'_OLS_model.json')).write_text(json.dumps(dict(features=FEATURES,intercept=intercept,coefficients=coeff.tolist(),fit_method='Centered numpy least squares on the same training FIDs'),indent=2),encoding='utf-8')
    shaproot=ROOT.parent/'O2O_SHAP_supplement'
    sv=pd.read_csv(shaproot/'results'/('600m_'+target+'_shap_values.csv'))
    check(target+' SHAP additivity',np.max(np.abs(sv[FEATURES].sum(axis=1)+sv.base_value-sv.prediction))<3e-5)
    check(target+' SHAP/baseline predictions',np.allclose(sv.set_index('FID').loc[pred.FID,'prediction'],pred.XGBoost_predicted))
    check(target+' SHAP observed counts',np.array_equal(sv.observed,indexed.loc[sv.FID,target]))
    inputs=pd.read_csv(shaproot/'results'/'600m_shap_inputs.csv')
    check(target+' shared SHAP inputs',np.allclose(inputs[FEATURES],indexed.loc[inputs.FID,FEATURES]))
    summary=pd.read_csv(shaproot/'results'/'600m_shap_importance.csv')
    check(target+' SHAP importance summary',np.allclose(sv[FEATURES].abs().mean(),summary[summary.category==target].set_index('feature').loc[FEATURES,'mean_abs_shap']))

sp=pd.read_csv(RESULTS/'spatial_predictions.csv')
folds=pd.read_csv(ROOT/'data'/'spatial_fold_assignments.csv')
spmetrics=pd.read_csv(RESULTS/'spatial_fold_metrics.csv')
for (size,target,fold),p in sp.groupby(['block_size','category','fold']):
    membership=folds[folds.block_size==size].set_index('FID')
    check('spatial block exclusion %s %s %s'%(size,target,fold),not(set(membership.loc[membership.test_fold==fold,'block'])&set(membership.loc[membership.test_fold!=fold,'block'])))
    check('spatial prediction membership %s %s %s'%(size,target,fold),set(p.FID)==set(membership.index[membership.test_fold==fold]))
    stats=scores(p.observed,p.predicted);row=spmetrics[(spmetrics.block_size==size)&(spmetrics.category==target)&(spmetrics.fold==fold)].iloc[0]
    check('spatial metrics %s %s %s'%(size,target,fold),all(np.isclose(stats[k],row[k],rtol=1e-8,atol=1e-8) for k in stats))
    check('spatial observed %s %s %s'%(size,target,fold),np.array_equal(p.observed,indexed.loc[p.FID,target]))
spmean=spmetrics.groupby(['block_size','category']).r2.mean()
for size,expected in [(3000,[.326,.430,.062,.021]),(5000,[.268,.347,.050,.122])]:
    check('response spatial summary '+str(size),np.array_equal(np.round(spmean.loc[size].loc[TARGETS].values,3),expected))
sensitivity=pd.read_csv(RESULTS/'sensitivity_predictions.csv')
sm=pd.read_csv(RESULTS/'sensitivity_fold_metrics.csv')
for key,p in sensitivity.groupby(['scale','category','model','scheme','fold']):
    selected=sm
    for name,value in zip(['scale','category','model','scheme','fold'],key):selected=selected[selected[name]==value]
    stats=scores(p.observed,p.predicted)
    check('scale/random metrics '+str(key),len(selected)==1 and all(np.isclose(stats[k],selected.iloc[0][k],rtol=1e-8,atol=1e-8) for k in stats))
    dataset=fine if key[0]=='600m' else coarse
    check('scale observations '+str(key),np.array_equal(p.observed,dataset.iloc[p.row_index][key[1]].values))
randommean=sm[(sm.scale=='600m')&(sm.model=='XGBoost')&(sm.scheme=='random_cv5')].groupby('category').r2.mean()
check('response random fivefold summary',np.array_equal(np.round(randommean.loc[TARGETS].values,3),[.372,.446,.067,-.014]))
geometry=ROOT.parent/'O2O_spatial_grids'
geo=pd.read_csv(geometry/'grid_600m_model_centres.csv').set_index('FID')
check('600m geometry count attributes',np.array_equal(geo.loc[fine.FID,TARGETS].values,fine[TARGETS].values))
gc=pd.read_csv(geometry/'grid_1200m_model_centres.csv').set_index(['gx','gy'])
check('1200m geometry count attributes',np.array_equal(gc.loc[list(zip(coarse.gx,coarse.gy)),TARGETS].values,coarse[TARGETS].values))
for size in [3000,5000]:
    f=folds[folds.block_size==size].set_index('FID')
    check('geometry fold map '+str(size),np.array_equal(geo.loc[fine.FID,'fold'+str(size)].values,f.loc[fine.FID,'test_fold'].values))
report=pd.DataFrame(checks)
report.to_csv(RESULTS/'package_validation_checks.csv',index=False)
(RESULTS/'package_validation_status.json').write_text(json.dumps(dict(success=True,checks=len(report),models_retrained=False,OLS_independently_refitted=True,XGBoost_JSON_inference_checked=True,KDE_excluded=True),indent=2),encoding='utf-8')
print('PASS: %d independent checks; baseline, importance, SHAP, folds and reply summaries match.'%len(report))
