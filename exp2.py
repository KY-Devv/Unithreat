import numpy as np, pandas as pd, sys, json, warnings, time
warnings.filterwarnings('ignore')
import exp, config
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, balanced_accuracy_score, average_precision_score, confusion_matrix
W=config.W; out={}
# ---- X3 leakage: record-random vs entity-grouped CV (RF, raw features)
rows=[]
for task,kind in [('apt_flow','bin'),('andmal_dyn','multi'),('unraveled','bin')]:
    D=np.load(W+f'task_{task}.npz',allow_pickle=True); X=D['X']; y=(D['yb'] if kind=='bin' else D['ym']).astype(int); g=D['groups']; K=len(np.unique(y))
    for proto,folds in [('record-random',exp.splitter(y,None,5)),('entity-grouped',exp.splitter(y,g,5))]:
        for fi,(tr,te) in enumerate(folds):
            m=RandomForestClassifier(100,class_weight='balanced_subsample',n_jobs=config.N_JOBS,random_state=42).fit(X[tr],y[tr]); P=m.predict_proba(X[te])
            rows.append(dict(task=task,protocol=proto,fold=fi,**exp.metrics(y[te],P.argmax(1),P,K)))
    print(task,'leak done',flush=True)
pd.DataFrame(rows).to_csv(W+'res_leakage.csv',index=False)
# ---- X4a universal schema-free detector on the 20-d record profile only
T=['andmal_static','maldroid','apt_flow','unraveled']; data={}
for t in T:
    D=np.load(W+f'task_{t}.npz',allow_pickle=True); y=D['yb'].astype(int); g=D['groups'] if len(D['groups']) else None
    data[t]=(D['RP'],y,exp.splitter(y,g,5))
def wts(parts):
    w=[]
    for (R,y) in parts:
        c=np.bincount(y,minlength=2); w.append(1.0/(len(parts)*2*c[y]))
    return np.concatenate(w)
rows=[]
for fi in range(5):
    parts=[(data[t][0][data[t][2][fi][0]],data[t][1][data[t][2][fi][0]]) for t in T]
    m=HistGradientBoostingClassifier(max_iter=200,random_state=42).fit(np.vstack([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),sample_weight=wts(parts)*1e4)
    for t in T:
        te=data[t][2][fi][1]; s=m.predict_proba(data[t][0][te])[:,1]; y=data[t][1][te]
        rows.append(dict(setting='pooled (one model, all domains)',test=t,fold=fi,auc=roc_auc_score(y,s),mcc=matthews_corrcoef(y,s>.5),bacc=balanced_accuracy_score(y,s>.5)))
for held in T:
    parts=[(data[t][0],data[t][1]) for t in T if t!=held]
    m=HistGradientBoostingClassifier(max_iter=200,random_state=42).fit(np.vstack([p[0] for p in parts]),np.concatenate([p[1] for p in parts]),sample_weight=wts(parts)*1e4)
    s=m.predict_proba(data[held][0])[:,1]; y=data[held][1]
    rows.append(dict(setting='leave-one-domain-out',test=held,fold=0,auc=roc_auc_score(y,s),mcc=matthews_corrcoef(y,s>.5),bacc=balanced_accuracy_score(y,s>.5)))
for a in T:
    m=HistGradientBoostingClassifier(max_iter=200,class_weight='balanced',random_state=42).fit(data[a][0],data[a][1])
    for b in T:
        if a!=b: s=m.predict_proba(data[b][0])[:,1]; rows.append(dict(setting='pairwise transfer',train=a,test=b,fold=0,auc=roc_auc_score(data[b][1],s)))
pd.DataFrame(rows).to_csv(W+'res_universal.csv',index=False); print('universal done',flush=True)
