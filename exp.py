import numpy as np, pandas as pd, time, json, sys, warnings, pickle, config
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
W=config.W; SEED=config.SEED; NJ=config.N_JOBS
slog=FunctionTransformer(lambda X: np.sign(X)*np.log1p(np.abs(X)))
def scaled(m): return make_pipeline(slog,StandardScaler(),m)
def baselines():
    return {'LR':scaled(LogisticRegression(C=1.0,max_iter=300,class_weight='balanced')),
            'SVM':scaled(SGDClassifier(loss='hinge',alpha=1e-4,class_weight='balanced',max_iter=50,tol=1e-3,random_state=SEED)),
            'RF':RandomForestClassifier(200,class_weight='balanced_subsample',n_jobs=NJ,random_state=SEED),
            'DT':DecisionTreeClassifier(class_weight='balanced',random_state=SEED),
            'NB':scaled(GaussianNB()),
            'KNN':scaled(KNeighborsClassifier(5,n_jobs=NJ))}
def bases(K=2):
    return {'rf':RandomForestClassifier(150,class_weight='balanced_subsample',n_jobs=NJ,random_state=SEED),
            'et':ExtraTreesClassifier(150,class_weight='balanced_subsample',n_jobs=NJ,random_state=SEED),
            'hgb':HistGradientBoostingClassifier(max_iter=120 if K==2 else 40,learning_rate=0.1 if K==2 else 0.2,class_weight='balanced',early_stopping=False,random_state=SEED),
            'lr':scaled(LogisticRegression(C=1.0,max_iter=300,class_weight='balanced'))}
def scores(m,X):
    if hasattr(m,'predict_proba'): return m.predict_proba(X)
    s=m.decision_function(X); return np.column_stack([-s,s]) if s.ndim==1 else s
def metrics(y,pred,S,K):
    p,r,f,_=precision_recall_fscore_support(y,pred,average='macro',zero_division=0)
    o={'acc':accuracy_score(y,pred),'bacc':balanced_accuracy_score(y,pred),'prec':p,'rec':r,'f1':f,'mcc':matthews_corrcoef(y,pred)}
    try:
        if K==2:
            o['auc']=roc_auc_score(y,S[:,1]); o['prauc']=average_precision_score(y,S[:,1])
            tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel(); o['tpr']=tp/max(tp+fn,1); o['fpr']=fp/max(fp+tn,1); o['f1_mal']=2*tp/max(2*tp+fp+fn,1)
        else:
            E=np.exp(S-S.max(1,keepdims=True)) if (S.min()<0 or S.max()>1) else S; E=E/E.sum(1,keepdims=True)
            o['auc']=roc_auc_score(y,E,multi_class='ovr',average='macro',labels=np.arange(K))
    except Exception as e: o['auc']=np.nan
    return o
def splitter(y,g,n,seed=SEED):
    if g is not None: return list(StratifiedGroupKFold(n,shuffle=True,random_state=seed).split(y,y,g))
    return list(StratifiedKFold(n,shuffle=True,random_state=seed).split(y,y))
def stack_fit_predict(Xtr,ytr,gtr,Xte,K,keep=None):
    """manual group-aware stacking; returns dict of test probabilities for every base, and the stacked meta output"""
    names=list(bases(K).keys()); oof=np.zeros((len(ytr),len(names)*K)); te=np.zeros((len(Xte),len(names)*K))
    for a,b in splitter(ytr,gtr,3,seed=7):
        for j,(n,m) in enumerate(bases(K).items()):
            m.fit(Xtr[a],ytr[a]); P=np.zeros((len(b),K)); P[:,m.classes_ if not hasattr(m,'steps') else m.steps[-1][1].classes_]=m.predict_proba(Xtr[b]); oof[b,j*K:(j+1)*K]=P
    out={}
    for j,(n,m) in enumerate(bases(K).items()):
        m.fit(Xtr,ytr); P=m.predict_proba(Xte); te[:,j*K:(j+1)*K]=P; out[n]=P
    meta=LogisticRegression(C=1.0,max_iter=500,class_weight='balanced').fit(oof,ytr)
    out['stack']=meta.predict_proba(te); out['meta_coef']=np.abs(meta.coef_).reshape(-1,len(names),K).mean((0,2)) if K>2 else np.abs(meta.coef_).reshape(len(names),K).mean(1)
    return out
def run(task,kind):
    """Six baseline classifiers on the native features, five leakage-aware folds."""
    D=np.load(W+f'task_{task}.npz',allow_pickle=True); X=D['X']
    y=(D['yb'] if kind=='bin' else D['ym']).astype(int); g=D['groups'] if len(D['groups']) else None; K=len(np.unique(y)); res=[]
    rng=np.random.RandomState(SEED)
    for fi,(tr,te) in enumerate(splitter(y,g,5)):
        for n,m in baselines().items():
            trn=tr
            if n=='KNN' and config.KNN_MAX_TRAIN and len(tr)>config.KNN_MAX_TRAIN: trn=rng.choice(tr,config.KNN_MAX_TRAIN,replace=False)
            t=time.time(); m.fit(X[trn],y[trn]); tf=time.time()-t; S=scores(m,X[te]); pred=m.predict(X[te])
            res.append(dict(task=task,kind=kind,model=n,fold=fi,fit_s=tf,**metrics(y[te],pred,S,K)))
        print(task,kind,'baselines fold',fi,'done',round(time.time()-T0),'s',flush=True)
    pd.DataFrame(res).to_csv(W+f'res_{task}_{kind}.csv',index=False)
T0=time.time()
if __name__=='__main__':
    task,kind=sys.argv[1],sys.argv[2]
    run(task,kind)
