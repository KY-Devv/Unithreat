import pandas as pd, numpy as np, glob, json, sys, warnings
warnings.filterwarnings('ignore'); import config
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, average_precision_score, confusion_matrix
W=config.W
B=pd.concat([pd.read_csv(f) for f in glob.glob(W+'res_*_bin.csv')+glob.glob(W+'res_*_multi.csv')]); B=B[B.model.isin(['LR','SVM','RF','DT','NB','KNN'])]
U=pd.concat([pd.read_csv(f) for f in glob.glob(W+'ut_*.csv')])
R=pd.concat([B,U],ignore_index=True); R['tk']=R.task+'|'+R.kind; R.to_csv(W+'all_folds.csv',index=False)
mets=['acc','bacc','prec','rec','f1','mcc','auc','prauc','tpr','fpr']
S=R.groupby(['tk','model'])[mets].agg(['mean','std']); S.columns=['_'.join(c) for c in S.columns]; S=S.reset_index(); S.to_csv(W+'summary.csv',index=False)
gate=U[U.model=='UniThreat'].groupby(['task','kind']).apply(lambda d: pd.Series({'view1_folds':int((d.sel_view==1).sum()),'balanced_folds':int((d.sel_cw=='balanced').sum())})).reset_index(); gate.to_csv(W+'gate.csv',index=False)
P=R.pivot_table(index=['tk','fold'],columns='model',values='mcc'); st={}
for b in ['LR','SVM','RF','DT','NB','KNN']:
    d=(P['UniThreat']-P[b]).dropna(); w=stats.wilcoxon(d)
    st[b]={'n':len(d),'mean_diff_mcc':float(d.mean()),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'p':float(w.pvalue)}
T=R[R.model.isin(['LR','SVM','RF','DT','NB','KNN','UniThreat'])].pivot_table(index='tk',columns='model',values='mcc')
fr=stats.friedmanchisquare(*[T[c] for c in T]); st['friedman']={'chi2':float(fr.statistic),'p':float(fr.pvalue),'ranks':T.rank(axis=1,ascending=False).mean().round(2).to_dict(),'n_tasks':len(T)}
json.dump(st,open(W+'stats.json','w'),indent=1)
# unified decision layer on UniThreat out-of-fold scores
def ece(y,p,nb=10):
    b=np.minimum((p*nb).astype(int),nb-1); return sum((b==k).mean()*abs(y[b==k].mean()-p[b==k].mean()) for k in range(nb) if (b==k).any())
Tk=['andmal_static','maldroid','apt_flow','unraveled']; rows=[]; pool={'raw':[],'cal':[],'y':[],'dom':[]}
for t in Tk:
    O=np.load(W+f'oofut_{t}.npz'); y=O['y']; s=O['p']; f=O['fold']; cal=np.zeros(len(y))
    for k in range(5): cal[f==k]=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(s[f!=k],y[f!=k]).predict(s[f==k])
    for nm,p in [('raw',s),('cal',cal)]:
        tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
        rows.append(dict(domain=t,score=nm,auc=roc_auc_score(y,p),prauc=average_precision_score(y,p),ece=ece(y,p),f1_mal=f1_score(y,p>=.5),mcc=matthews_corrcoef(y,p>=.5),tpr=tp/(tp+fn),fpr=fp/(fp+tn),n=len(y),prev=y.mean())); pool[nm].append(p)
    pool['y'].append(y); pool['dom'].append(np.full(len(y),t))
y=np.concatenate(pool['y']); dom=np.concatenate(pool['dom'])
for nm in ('raw','cal'):
    p=np.concatenate(pool[nm]); tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    rows.append(dict(domain='ALL',score=nm,auc=roc_auc_score(y,p),prauc=average_precision_score(y,p),ece=ece(y,p),f1_mal=f1_score(y,p>=.5),mcc=matthews_corrcoef(y,p>=.5),tpr=tp/(tp+fn),fpr=fp/(fp+tn),n=len(y),prev=y.mean()))
pd.DataFrame(rows).to_csv(W+'res_unified_layer.csv',index=False)
p=np.concatenate(pool['cal']); android=np.isin(dom,['andmal_static','maldroid'])
true=np.where(y==0,0,np.where(android,1,2)); pred=np.where(p<.5,0,np.where(android,1,2)); cm=confusion_matrix(true,pred,labels=[0,1,2])
json.dump({'joint_acc':accuracy_score(true,pred),'joint_macro_f1':f1_score(true,pred,average='macro'),'joint_mcc':matthews_corrcoef(true,pred),'cm':cm.tolist()},open(W+'joint.json','w'))
pd.set_option('display.width',250)
order=['LR','SVM','RF','DT','NB','KNN','HSE-native (balanced meta)','HSE-native (plain meta)','HSE+SABP (balanced meta)','HSE+SABP (plain meta)','UniThreat']
for tk,g in S.groupby('tk'):
    g=g.set_index('model').reindex(order); print('\n==',tk); print((g[[m+'_mean' for m in ['acc','bacc','prec','rec','f1','mcc','auc','tpr','fpr']]]*100).round(2).to_string())
print(gate.to_string()); print(json.dumps(st,indent=0)); print(pd.DataFrame(rows).round(4).to_string()); print(json.load(open(W+'joint.json')))
