"""Level-1 of UniThreat: validation-gated stacking. For every outer fold the gate picks, using training data only,
the feature view (native / native+SABP) and the meta-learner class weighting (balanced / none) that maximise
cross-fitted MCC of the meta-learner on the level-0 out-of-fold probabilities."""
import sys, glob, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
import exp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
W=exp.W
def mk(cw): return LogisticRegression(C=1.0,max_iter=500,class_weight=cw)
CANDS=[(v,cw) for v in (0,1) for cw in ('balanced',None)]
def run(task,kind):
    rows=[]; sel=[]; files=sorted(glob.glob(W+f'L0_{task}_{kind}_f*.npz'))
    if len(files)<5: return None
    n=sum(len(np.load(f)['yte']) for f in files); 
    for fi,f in enumerate(files):
        L=np.load(f); ytr=L['ytr']; yte=L['yte']; g=L['gtr'] if len(L['gtr']) else None; K=len(np.unique(ytr)); sc={}; out={}
        for v,cw in CANDS:
            O=L[f'oof{v}']; cv=np.zeros(len(ytr),int)
            for a,b in exp.splitter(ytr,g,3,seed=11): cv[b]=mk(cw).fit(O[a],ytr[a]).predict(O[b])
            sc[(v,cw)]=matthews_corrcoef(ytr,cv)
            P=mk(cw).fit(O,ytr).predict_proba(L[f'te{v}']); out[(v,cw)]=P
        best=max(sc,key=sc.get); sel.append(best)
        names={(0,'balanced'):'HSE-native (balanced meta)',(0,None):'HSE-native (plain meta)',(1,'balanced'):'HSE+SABP (balanced meta)',(1,None):'HSE+SABP (plain meta)'}
        for k,P in out.items(): rows.append(dict(task=task,kind=kind,model=names[k],fold=fi,**exp.metrics(yte,P.argmax(1),P,K)))
        P=out[best]; rows.append(dict(task=task,kind=kind,model='UniThreat',fold=fi,sel_view=best[0],sel_cw=str(best[1]),**exp.metrics(yte,P.argmax(1),P,K)))
        for j,b in enumerate(['RF','ET','HGB','LR']):
            for v in (0,1):
                P=L[f'te{v}'][:,j*K:(j+1)*K]; rows.append(dict(task=task,kind=kind,model=f'base{v}:{b}',fold=fi,**exp.metrics(yte,P.argmax(1),P,K)))
        if K==2:
            if fi==0: oof=np.zeros(n); yy=np.zeros(n,int); ff=np.zeros(n,int)
            oof[L['te_idx']]=out[best][:,1]; yy[L['te_idx']]=yte; ff[L['te_idx']]=fi
    R=pd.DataFrame(rows); R.to_csv(W+f'ut_{task}_{kind}.csv',index=False)
    if K==2: np.savez(W+f'oofut_{task}.npz',p=oof,y=yy,fold=ff)
    return R,sel
if __name__=='__main__':
    for task,kind in [('andmal_static','bin'),('maldroid','bin'),('apt_flow','bin'),('unraveled','bin'),('andmal_static','multi'),('andmal_dyn','multi'),('maldroid','multi')]:
        r=run(task,kind)
        if r is None: print(task,kind,'pending'); continue
        R,sel=r; print('\n==',task,kind,'gate picks:',sel)
        print((R.groupby('model')[['acc','bacc','f1','mcc','auc']].mean()*100).round(2).to_string())
