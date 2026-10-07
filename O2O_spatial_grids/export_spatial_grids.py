from pathlib import Path
import argparse, shutil
import pandas as pd
import shapefile
ROOT=Path(__file__).resolve().parent
CORE=ROOT.parent/'O2O_reproducibility_notebook'
parser=argparse.ArgumentParser()
parser.add_argument('--output',default='regenerated')
args=parser.parse_args()
OUT=ROOT/args.output
if OUT.resolve()==ROOT.resolve():
    raise ValueError('Choose a separate export directory')
OUT.mkdir(parents=True,exist_ok=True)
fine=pd.read_csv(CORE/'data'/'600m_model_input.csv').set_index('FID')
folds=pd.read_csv(CORE/'data'/'spatial_fold_assignments.csv')
coarse=pd.read_csv(CORE/'data'/'1200m_model_input.csv').set_index(['gx','gy'])
for name in ['grid_600m_model','grid_1200m_model','cv_blocks_3000m','cv_blocks_5000m']:
    reader=shapefile.Reader(str(ROOT/name))
    fields=[f[0] for f in reader.fields[1:]]
    writer=shapefile.Writer(str(OUT/name),shapeType=reader.shapeType)
    for field in reader.fields[1:]: writer.field(*field)
    for shape_record in reader.iterShapeRecords():
        row=dict(zip(fields,list(shape_record.record)))
        if name=='grid_600m_model':
            for y in ['Y1','Y2','Y3','Y4']: row[y]=int(fine.loc[row['FID'],y])
            for size in [3000,5000]:
                membership=folds[(folds.block_size==size)&(folds.FID==row['FID'])].iloc[0]
                row['blk'+str(size)]=int(membership.block)
                row['fold'+str(size)]=int(membership.test_fold)
        elif name=='grid_1200m_model':
            cell=coarse.loc[(row['gx'],row['gy'])]
            for y in ['Y1','Y2','Y3','Y4']: row[y]=int(cell[y])
            row['area_m2']=float(cell['area'])
        else:
            size=3000 if name=='cv_blocks_3000m' else 5000
            membership=folds[(folds.block_size==size)&(folds.block==row['block_id'])]
            if membership.test_fold.nunique()!=1: raise ValueError('A spatial block must have one test fold')
            row['test_fold']=int(membership.test_fold.iloc[0])
            row['n_grids']=len(membership)
            for y in ['Y1','Y2','Y3','Y4']:row[y]=int(fine.loc[membership.FID,y].sum())
        writer.shape(shape_record.shape)
        writer.record(*[row[k] for k in fields])
    writer.close()
    reader.close()
    for extension in ['.prj','.cpg']:
        shutil.copy2(ROOT/(name+extension),OUT/(name+extension))
    centres=pd.read_csv(ROOT/(name+'_centres.csv'))
    if name=='grid_600m_model':
        for y in ['Y1','Y2','Y3','Y4']:centres[y]=fine.loc[centres.FID,y].values
        for size in [3000,5000]:
            membership=folds[folds.block_size==size].set_index('FID')
            centres['blk'+str(size)]=membership.loc[centres.FID,'block'].values
            centres['fold'+str(size)]=membership.loc[centres.FID,'test_fold'].values
    elif name=='grid_1200m_model':
        data=coarse.loc[list(zip(centres.gx,centres.gy))]
        for y in ['Y1','Y2','Y3','Y4']:centres[y]=data[y].values
        centres['area_m2']=data.area.values
    else:
        size=3000 if name=='cv_blocks_3000m' else 5000
        for k,centre in centres.iterrows():
            membership=folds[(folds.block_size==size)&(folds.block==centre.block_id)]
            centres.loc[k,'test_fold']=int(membership.test_fold.iloc[0])
            centres.loc[k,'n_grids']=len(membership)
            for y in ['Y1','Y2','Y3','Y4']:centres.loc[k,y]=int(fine.loc[membership.FID,y].sum())
    centres.to_csv(OUT/(name+'_centres.csv'),index=False)
shutil.copy2(ROOT/'600m_to_1200m_membership.csv',OUT/'600m_to_1200m_membership.csv')
print('Export complete:',OUT)
