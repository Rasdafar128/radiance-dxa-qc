"""Compare 14 frozen recipes per split. Run from the repository root.

python research/compare_geometry.py
Outputs stay in artifacts/ because intermediate tables contain data identifiers.
"""
import sys,json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'dxa-qc')]
import numpy as np,pandas as pd,pydicom
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
from dxaqc.geometry import spine_geometry,hip_geometry
from dxaqc.anatomy import hip_side_score,canonical_hip
from src.solution.dicom import pixels
from src import config as C
COLS={'spine_position':['crest_levels','above_crest_vertebrae','bottom_lat_frac','crest_h_min'], 'spine_axis':['angle_deg','angle_end_deg'], 'spine_objects':['tophat_top_p99','tophat_p999','sat_max_blob'], 'hip_position':['lt_prom_sw','shaft_tilt_deg','medial_resid_max_sw'], 'hip_roi':['h_px','lat_margin_px','gt_top_px','below_lt_mm','ok']}
cache=C.ARTIFACTS/'unified-geometry.csv'
df=pd.read_csv(C.ARTIFACTS/'e5-blend/manifest.csv'); rows=[]
for r in df.itertuples():
 im=pixels(pydicom.dcmread(C.DATA/r.path_to_study))
 geom=spine_geometry(im) if r.anatomical_region==C.REGION_SPINE else hip_geometry(canonical_hip(im,'R' if hip_side_score(im)>0 else 'L'))
 geom.pop('_viz',None); rows.append(geom)
g=pd.DataFrame(rows);g.to_csv(cache,index=False)
print('geometry',g.shape,flush=True)
from src.solution.model import targets,Model,Blend
from src.utils.train_blend import thresholds,cached_model
from functools import lru_cache
cached_model=lru_cache(None)(cached_model)
reports=[]
for split in ['e5-blend','e5-blend-split137']:
 parent=C.ARTIFACTS/split; df=pd.read_csv(parent/'manifest.csv')
 reference=pd.read_csv(C.ARTIFACTS/'e5-blend/manifest.csv')
 pd.testing.assert_frame_equal(df.drop(columns='fold'),reference.drop(columns='fold'))
 with np.load(parent/'features.npz') as c: x=c['features']; assert list(c['image_ids'])==list(df.image_id)
 for weight in [0,.25,.5,1.]:
  for mode in ['rare','all']:
   if weight==0 and mode=='all':continue
   records=[]
   for fold in range(3):
    tr=df[df.fold!=fold];te=df[df.fold==fold]
    inner=pd.read_csv(parent/f'fold_{fold}/inner_predictions.csv').set_index('image_id').reindex(tr.image_id)
    base=Blend([cached_model(parent/f'fold_{fold}/member_{i}') for i in range(2)],json.loads((parent/f'fold_{fold}/model.json').read_text())['heads']);_,score=base.classify(x[te.index]);outer=pd.DataFrame(score,index=te.image_id)
    assignments=tr.study.map(pd.read_csv(parent/f'inner_{fold}.csv').set_index('study').fold).to_numpy()
    for key,y in targets(tr).items():
     if key not in COLS or (mode=='rare' and key not in ['spine_axis','hip_roi']):continue
     X=g[COLS[key]].to_numpy();known=np.isfinite(y); pi=np.full(len(tr),np.nan)
     def fit(ix):
      yy=y[ix]; XX=X[tr.index[ix]]
      if np.unique(yy).size<2:return float(yy.mean())
      return make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',random_state=42,max_iter=2000)).fit(XX,yy)
     def pred(m,xx):return np.full(len(xx),m) if isinstance(m,float) else m.predict_proba(xx)[:,1]
     for k in [0,1]:
      a=known&(assignments!=k);b=known&(assignments==k);pi[b]=pred(fit(a),X[tr.index[b]])
     pg=pred(fit(known),X[te.index]); inner[key]=(1-weight)*inner[key]+weight*pi;outer[key]=(1-weight)*outer[key]+weight*pg
    for qmode in ['direct','mixed']:
     ins=inner.copy();out=outer.copy()
     if qmode=='mixed':
      for key,ts in [('spine_quality',C.TARGETS[:3]),('hip_quality',C.TARGETS[3:])]:
       ins[key]=.5*ins[key]+.5*(1-np.prod(1-ins[ts],axis=1));out[key]=.5*out[key]+.5*(1-np.prod(1-out[ts],axis=1))
     md=Model.__new__(Model);md.heads=thresholds(tr,ins);md.metadata={'quality_gate':True}
     result,s=md.decide({k:out[k].to_numpy() for k in out})
     rec=te[['image_id','anatomical_region','quality_class']].copy();rec['pred']=result.quality_class.to_numpy();rec['p']=result.quality_prob.to_numpy();rec['mode']=qmode
     for key,reg,label in zip(C.TARGETS,C.TARGET_REGIONS,sum(C.VIOLATIONS.values(),[])):
      rec['true_'+key]=targets(te)[key];rec['pred_'+key]=[label in st.split(C.VIOLATION_SEP) and rr==reg for st,rr in zip(result.violation_type,result.anatomical_region)]
     records.append(rec)
   for qm in ['direct','mixed']:
    r=pd.concat([r for r in records if r['mode'].iloc[0]==qm]); f={k:f1_score(r.loc[r['true_'+k].notna(),'true_'+k],r.loc[r['true_'+k].notna(),'pred_'+k],zero_division=0) for k in C.TARGETS}
    row=dict(split=split,weight=weight,mode=mode,quality=qm,f1=f1_score(r.quality_class,r.pred),auc=roc_auc_score(r.quality_class,r.p),macro=float(np.mean(list(f.values()))),types=f)
    reports.append(row);print(json.dumps(row),flush=True)
Path('artifacts/unified-comparison.json').write_text(json.dumps(reports,indent=2))
