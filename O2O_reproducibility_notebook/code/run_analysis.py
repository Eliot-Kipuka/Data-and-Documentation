"""Reproduce the original 600 m outputs and aligned supplementary analyses.

Run: python code/run_analysis.py
Original notebooks are archived byte-for-byte and are never edited or executed here.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import platform
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'results'
FEATURES = ['X1','X2','X3','X4','X6','X7','X8','X19','X10','X14','X17','X18','X22']
TARGETS = ['Y1','Y2','Y3','Y4']
NAMES = ['Population Density','Income Level','Education Level','Youth Population',
         'Express outlets density','High-traffic road density','Low-traffic road density',
         'Distance to Public Transit','Commercial Diversity','Retail Space Rent',
         'Density of Cultural and Leisure Facilities','Office Density','Distance to Shopping Mall']
TABLE_ORDER = ['X1','X2','X3','X4','X6','X7','X8','X19','X10','X14','X17','X18','X22']
PARAMETERS = {
    'Y1': dict(n_estimators=500,max_depth=2,learning_rate=.01),
    'Y2': dict(n_estimators=300,max_depth=3,learning_rate=.01),
    'Y3': dict(n_estimators=300,max_depth=1,learning_rate=.02),
    'Y4': dict(n_estimators=300,max_depth=3,learning_rate=.005),
}

class OriginalSettingsRegressor(xgb.XGBRegressor):
    """Make the settings reproducing archived results explicit for every fit.

    The intercept uses only the training labels of the current fold. This
    does not estimate the archived software version; it reproduces the
    archived parameters, scores and gain importance on supplied data.
    """
    def fit(self, X, y, **kwargs):
        self.set_params(base_score=float(np.mean(y)))
        return super().fit(X, y, **kwargs)

def estimator(target):
    return OriginalSettingsRegressor(tree_method='hist',random_state=0,
        eval_metric='rmse',n_jobs=1,**PARAMETERS[target])

def metrics(y, pred):
    mse = mean_squared_error(y,pred)
    return dict(r2=r2_score(y,pred),mse=mse,rmse=np.sqrt(mse),mae=mean_absolute_error(y,pred))

def export(frame, name):
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')

def aggregate_1200(cells):
    raw=cells.copy()
    raw['pop_weight']=np.where(raw.X1>0,raw.X1*raw.area,0)
    raw['active_area']=np.where(raw.X1>0,raw.area,0)
    raw['distance_area']=np.where((raw.X1>0)&(raw.X19>0)&(raw.X22>0),raw.area,0)
    area_cols=['X1','X6','X7','X8','X17','X18']
    for col in area_cols: raw[col+'_num']=raw[col]*raw.area
    for col in ['X2','X3','X4']: raw[col+'_num']=raw[col]*raw.pop_weight
    for col in ['X10','X14']: raw[col+'_num']=raw[col]*raw.active_area
    for col in ['X19','X22']: raw[col+'_num']=raw[col]*raw.distance_area
    sums=['area','total']+TARGETS+['pop_weight','active_area','distance_area']+[c+'_num' for c in FEATURES]
    coarse=raw.groupby(['gx','gy'],as_index=False)[sums].sum()
    for col in area_cols: coarse[col]=coarse[col+'_num']/coarse.area
    for col in ['X2','X3','X4']: coarse[col]=coarse[col+'_num']/coarse.pop_weight.replace(0,np.nan)
    for col in ['X10','X14']: coarse[col]=coarse[col+'_num']/coarse.active_area.replace(0,np.nan)
    for col in ['X19','X22']: coarse[col]=coarse[col+'_num']/coarse.distance_area.replace(0,np.nan)
    coarse=coarse[(coarse.total>=1)&(coarse.X1>0)].dropna(subset=FEATURES)
    return coarse[['gx','gy','area','total']+TARGETS+FEATURES].reset_index(drop=True)

def prepare():
    fine=pd.read_csv(DATA/'600m_model_input.csv')
    source=pd.read_csv(DATA/'source_600m_grid_cells.csv')
    assert fine.FID.is_unique and len(fine)==3570
    # A single documented data correction. Keep the six-decimal source
    # precision for all other fields to avoid unrelated numerical changes.
    old=source.loc[source.FID==1690,'X17'].iloc[0]
    new=fine.loc[fine.FID==1690,'X17'].iloc[0]
    source.loc[source.FID==1690,'X17']=new
    aligned=source.set_index('FID').loc[fine.FID,FEATURES+TARGETS]
    assert np.allclose(aligned.values,fine[FEATURES+TARGETS].values,rtol=0,atol=5.1e-7)
    source.to_csv(DATA/'source_600m_grid_cells.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    coarse=aggregate_1200(source)
    assert len(coarse)==1648 and np.allclose(fine[TARGETS].sum(),coarse[TARGETS].sum())
    assert np.isclose(coarse.loc[(coarse.gx==33)&(coarse.gy==52),'X17'].iloc[0],232.6388888055555)
    coarse.to_csv(DATA/'1200m_model_input.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
    export(pd.DataFrame([dict(FID=1690,feature='X17',source_value_before_correction=0,
        source_value_after_correction=new,coarse_gx=33,coarse_gy=52,
        old_coarse_value=114.58333325,new_coarse_value=coarse.loc[(coarse.gx==33)&(coarse.gy==52),'X17'].iloc[0])]),'data_correction.csv')
    print('3570 fine grids; 1648 coarse grids; store totals conserved. X17 source aligned.')
    return fine,coarse

def original_reference(target):
    nb=json.loads((ROOT/'original_notebooks'/('Analysis resource-'+target+'.ipynb')).read_text('utf-8'))
    import re
    text=''.join(''.join(o.get('data',{}).get('text/plain',[])) for o in nb['cells'][16]['outputs'])
    gain=dict((k,float(v)) for k,v in re.findall(r'\d+\s+(X\d+)\s+([\d.]+)',text))
    text=''.join(''.join(o.get('text',[])) for o in nb['cells'][14]['outputs'])
    scores=re.findall(r'R2=([\d.]+), RMSE=([\d.]+), MAE=([\d.]+)',text)[-1]
    return gain,np.array(scores,dtype=float)

def baseline(fine):
    rows=[]; gains=[]; models={}; splits=[]; checks=[]
    tr,te=train_test_split(np.arange(len(fine)),test_size=.2,random_state=42)
    for i in tr: splits.append(dict(FID=int(fine.FID.iloc[i]),subset='train'))
    for i in te: splits.append(dict(FID=int(fine.FID.iloc[i]),subset='test'))
    export(pd.DataFrame(splits),'600m_holdout_assignments.csv')
    for target in TARGETS:
        X=fine[FEATURES].values; y=fine[target].values
        trained=estimator(target).fit(X[tr],y[tr]);models[target]=(trained,te)
        trained.save_model(str(OUT/('600m_'+target+'_model.json')))
        predicted={'FID':fine.FID.iloc[te].values,'observed':y[te]}
        for kind,m in [('XGBoost',trained),('OLS',LinearRegression().fit(X[tr],y[tr]))]:
            pred=m.predict(X[te]);predicted[kind+'_predicted']=pred
            rows.append(dict(scale='600m',category=target,model=kind,train_n=len(tr),test_n=len(te),**metrics(y[te],pred)))
            if kind=='OLS':
                (OUT/('600m_'+target+'_OLS_model.json')).write_text(json.dumps(dict(features=FEATURES,intercept=float(m.intercept_),coefficients=m.coef_.tolist(),fit_method='LinearRegression on the same training FIDs'),indent=2))
        export(pd.DataFrame(predicted),'600m_'+target+'_holdout_predictions.csv')
        ref,score=original_reference(target)
        actual=metrics(y[te],trained.predict(X[te]))
        assert np.allclose([actual['r2'],actual['rmse'],actual['mae']],score,rtol=0,atol=5.1e-5)
        for col,val in zip(FEATURES,trained.feature_importances_):
            assert abs(float(val)-ref[col])<5.1e-7
            gains.append(dict(category=target,feature=col,gain_importance=float(val),relative_importance_percent=float(val)*100))
        checks.append(dict(category=target,max_gain_difference_to_saved_6dec=max(abs(float(v)-ref[c]) for c,v in zip(FEATURES,trained.feature_importances_)),archived_scores_reproduced=True))
    export(pd.DataFrame(rows),'600m_model_comparison.csv')
    gain=pd.DataFrame(gains)
    gain['rank']=gain.groupby('category').gain_importance.rank(ascending=False,method='min').astype(int)
    export(gain,'600m_gain_importance.csv');export(pd.DataFrame(checks),'original_output_checks.csv')
    export(pd.DataFrame([dict(category=t,tree_method='hist',base_score_rule='mean of training outcome',random_state=0,**PARAMETERS[t]) for t in TARGETS]),'model_settings.csv')
    print(pd.DataFrame(rows).to_string(index=False))
    return models

def shap_analysis(fine, models):
    """Exact native TreeSHAP, using the fitted trees and their path cover.

    This bypasses the incompatible installed shap/numba plotting dependency.
    approx_contribs=False; no interventional background dataset is supplied.
    The originals use shap.Explainer(model) without a background dataset.
    These files are numeric supplementary exports, not replacements for
    archived manuscript images. Original images were not compared pixelwise.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess
    folder=ROOT/'figures'/'shap';folder.mkdir(parents=True,exist_ok=True)
    summaries=[];checks=[];curve_rows=[];smooth_checks=[]
    for target,(model,te) in models.items():
        X=fine.iloc[te][FEATURES];dm=xgb.DMatrix(X.values)
        contrib=model.get_booster().predict(dm,pred_contribs=True,approx_contribs=False)
        pred=model.predict(X.values)
        delta=float(np.max(np.abs(contrib.sum(axis=1)-pred)))
        assert delta<1e-4
        values=pd.DataFrame(contrib[:,:-1],columns=FEATURES)
        values.insert(0,'FID',fine.FID.iloc[te].values)
        values['base_value']=contrib[:,-1];values['prediction']=pred
        values['observed']=fine.iloc[te][target].values
        export(values,'600m_'+target+'_shap_values.csv')
        test=fine.iloc[te][['FID']+FEATURES].copy();export(test,'600m_'+target+'_shap_inputs.csv')
        mean=np.abs(contrib[:,:-1]).mean(axis=0);order=np.argsort(mean)[::-1]
        for col,value in zip(FEATURES,mean):summaries.append(dict(category=target,feature=col,mean_abs_shap=float(value)))
        checks.append(dict(category=target,test_n=len(te),max_additivity_error=delta,algorithm='exact native TreeSHAP',feature_perturbation='tree_path_dependent'))
        fig,ax=plt.subplots(figsize=(7,5));ax.barh(np.array(FEATURES)[order][::-1],mean[order][::-1]);ax.set_xlabel('Mean absolute TreeSHAP value (shop counts)');ax.set_title(target+' / 600 m test set');fig.tight_layout();fig.savefig(str(folder/(target+'_importance.png')),dpi=180);plt.close(fig)
        fig,axes=plt.subplots(4,4,figsize=(16,13))
        for i,col in enumerate(FEATURES):
            ax=axes.flat[i];xx=X[col].values;yy=contrib[:,i]
            ax.scatter(xx,yy,s=9,alpha=.45)
            curve=lowess(yy,xx,frac=2./3.,it=3,return_sorted=True)
            iterations=3
            finite=np.isfinite(curve).all(axis=1)
            # Old statsmodels robust LOWESS can fail with repeated zero
            # predictors and piecewise-constant SHAP values. Preserve the
            # raw values and disclose the non-robust visual fallback.
            if finite.sum()<.8*len(xx):
                iterations=0
                curve=lowess(yy,xx,frac=2./3.,it=0,return_sorted=True)
                finite=np.isfinite(curve).all(axis=1)
            smooth_checks.append(dict(category=target,feature=col,lowess_iterations=iterations,finite_curve_points=int(finite.sum())))
            curve=curve[finite]
            ax.plot(curve[:,0],curve[:,1],color='coral');ax.axhline(0,color='black',linestyle='-.',linewidth=.7)
            ax.set_xlabel(col);ax.set_ylabel('TreeSHAP value')
            for xv,yv in curve:curve_rows.append(dict(category=target,feature=col,x=float(xv),lowess_shap=float(yv),lowess_iterations=iterations))
        for ax in list(axes.flat)[13:]:ax.set_visible(False)
        fig.tight_layout();fig.savefig(str(folder/(target+'_dependence.png')),dpi=180);plt.close(fig)
    export(pd.DataFrame(summaries),'600m_shap_importance.csv')
    export(pd.DataFrame(checks),'shap_additivity_checks.csv')
    export(pd.DataFrame(curve_rows),'shap_lowess_curves.csv')
    export(pd.DataFrame(smooth_checks),'shap_smoothing_checks.csv')
    print(pd.DataFrame(checks).to_string(index=False))

def sensitivity(fine,coarse):
    performance=[];importance=[];predictions=[];assignments=[]
    for scale,data in [('600m',fine),('1200m',coarse)]:
        X=data[FEATURES].values
        tr,te=train_test_split(np.arange(len(data)),test_size=.2,random_state=42)
        for i in range(len(data)):
            ids={'FID':int(data.FID.iloc[i])} if scale=='600m' else {'gx':int(data.gx.iloc[i]),'gy':int(data.gy.iloc[i])}
            assignments.append(dict(scale=scale,row_index=i,holdout_subset='test' if i in set(te) else 'train',**ids))
        for target in TARGETS:
            y=data[target].values
            for kind in ['XGBoost','OLS']:
                make=lambda: estimator(target) if kind=='XGBoost' else LinearRegression()
                fitted=make().fit(X[tr],y[tr]);pred=fitted.predict(X[te])
                performance.append(dict(scale=scale,category=target,model=kind,scheme='holdout',fold=0,train_n=len(tr),test_n=len(te),**metrics(y[te],pred)))
                for i,v in zip(te,pred):predictions.append(dict(scale=scale,category=target,model=kind,scheme='holdout',fold=0,row_index=int(i),observed=float(y[i]),predicted=float(v)))
                fold_gains=[]
                for fold,(train,test) in enumerate(KFold(5,shuffle=True,random_state=42).split(X),1):
                    m=make().fit(X[train],y[train]);p=m.predict(X[test])
                    performance.append(dict(scale=scale,category=target,model=kind,scheme='random_cv5',fold=fold,train_n=len(train),test_n=len(test),**metrics(y[test],p)))
                    for i,v in zip(test,p):predictions.append(dict(scale=scale,category=target,model=kind,scheme='random_cv5',fold=fold,row_index=int(i),observed=float(y[i]),predicted=float(v)))
                    if kind=='XGBoost':fold_gains.append(m.feature_importances_)
                if fold_gains:
                    arr=np.array(fold_gains);means=arr.mean(axis=0);ranks=pd.Series(means).rank(ascending=False,method='min').astype(int).values
                    for i,col in enumerate(FEATURES):importance.append(dict(scale=scale,category=target,feature=col,mean_gain=float(means[i]),sd_gain=float(arr[:,i].std(ddof=1)),rank=int(ranks[i])))
    perf=pd.DataFrame(performance);imp=pd.DataFrame(importance)
    export(perf,'sensitivity_fold_metrics.csv');export(pd.DataFrame(predictions),'sensitivity_predictions.csv');export(pd.DataFrame(assignments),'scale_holdout_assignments.csv');export(imp,'cv5_gain_importance.csv')
    summary=perf[perf.scheme=='random_cv5'].groupby(['scale','category','model'])[['r2','mse','rmse','mae']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];export(summary.reset_index(),'random_cv5_summary.csv')
    stability=[]
    for t in TARGETS:
        a=imp[(imp.scale=='600m')&(imp.category==t)].sort_values('rank').head(5).feature.tolist()
        b=imp[(imp.scale=='1200m')&(imp.category==t)].sort_values('rank').head(5).feature.tolist()
        stability.append(dict(category=t,top5_600m=', '.join(a),top5_1200m=', '.join(b),overlap=len(set(a)&set(b))))
    export(pd.DataFrame(stability),'top5_rank_stability.csv')
    print(summary.reset_index().to_string(index=False));print(pd.DataFrame(stability).to_string(index=False))

def spatial(fine):
    # Preserve the historical supplied fold map, including tie handling.
    # This prevents sklearn GroupKFold version changes altering the experiment.
    folds=pd.read_csv(DATA/'spatial_fold_assignments.csv')
    blocks=pd.read_csv(DATA/'spatial_blocks.csv')
    rows=[];predictions=[]
    for size in [3000,5000]:
        membership=folds[folds.block_size==size].set_index('FID').loc[fine.FID]
        assert np.array_equal(membership.block.values,blocks.set_index('FID').loc[fine.FID,'block_'+str(size)+'m'])
        assert membership.groupby('block').test_fold.nunique().max()==1
        for fold in range(1,6):
            test=np.flatnonzero(membership.test_fold.values==fold);train=np.flatnonzero(membership.test_fold.values!=fold)
            assert not set(membership.block.values[train])&set(membership.block.values[test])
            X=fine[FEATURES].values
            for target in TARGETS:
                y=fine[target].values;m=estimator(target).fit(X[train],y[train]);pred=m.predict(X[test])
                rows.append(dict(block_size=size,category=target,fold=fold,train_n=len(train),test_n=len(test),**metrics(y[test],pred)))
                for i,v in zip(test,pred):predictions.append(dict(block_size=size,category=target,fold=fold,FID=int(fine.FID.iloc[i]),observed=float(y[i]),predicted=float(v)))
    frame=pd.DataFrame(rows);export(frame,'spatial_fold_metrics.csv');export(pd.DataFrame(predictions),'spatial_predictions.csv')
    summary=frame.groupby(['block_size','category'])[['r2','mse','rmse','mae']].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    export(summary.reset_index(),'spatial_cv_summary.csv');print(summary.reset_index().to_string(index=False))

def search_original():
    fine=pd.read_csv(DATA/'600m_model_input.csv');rows=[]
    for target in TARGETS:
        nb=json.loads((ROOT/'original_notebooks'/('Analysis resource-'+target+'.ipynb')).read_text('utf-8'))
        code=''.join(nb['cells'][14]['source'])
        grid=next(ast.literal_eval(node.value) for node in ast.parse(code).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='param_grid' for t in node.targets))
        tr,te,yr,yt=train_test_split(fine[FEATURES].values,fine[target].values,test_size=.2,random_state=42)
        search=GridSearchCV(OriginalSettingsRegressor(tree_method='hist',random_state=0,eval_metric='rmse',n_jobs=1),grid,scoring='neg_mean_squared_error',cv=KFold(3,shuffle=True,random_state=0),n_jobs=4).fit(tr,yr)
        assert search.best_params_==PARAMETERS[target]
        export(pd.DataFrame(search.cv_results_),'original_search_'+target+'.csv')
        rows.append(dict(category=target,best_cv_negative_mse=search.best_score_,**search.best_params_,**metrics(yt,search.predict(te))))
    export(pd.DataFrame(rows),'original_search_summary.csv');print(pd.DataFrame(rows).to_string(index=False))

def environment():
    import sklearn,scipy,matplotlib,statsmodels
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__,matplotlib=matplotlib.__version__,statsmodels=statsmodels.__version__)
    (OUT/'environment.json').write_text(json.dumps(versions,indent=2),encoding='utf-8')
    return versions

def vif_analysis(fine):
    x=fine[FEATURES].values.astype(float)
    x=(x-x.mean(axis=0))/x.std(axis=0)
    rows=[]
    for j,name in enumerate(FEATURES):
        predictors=np.column_stack([np.ones(len(x)),np.delete(x,j,axis=1)])
        residual=x[:,j]-predictors.dot(np.linalg.lstsq(predictors,x[:,j],rcond=None)[0])
        tolerance=np.sum(residual**2)/np.sum(x[:,j]**2)
        rows.append(dict(feature=name,tolerance=tolerance,VIF=1/tolerance))
    export(pd.DataFrame(rows),'vif_tolerance.csv')

def main(search=False):
    warnings.filterwarnings('ignore')
    fine,coarse=prepare();vif_analysis(fine);models=baseline(fine);shap_analysis(fine,models)
    sensitivity(fine,coarse);spatial(fine)
    if search:search_original()
    print(environment())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--search',action='store_true',help='Also repeat the complete original parameter searches')
    main(parser.parse_args().search)
