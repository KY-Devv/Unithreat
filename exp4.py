"""Level-0 of UniThreat: fit the four base learners on two feature views (native; native+SABP) with group-aware inner folds.
Saves, per outer fold, the out-of-fold and test probability matrices so that the level-1 gate/meta-learner can be studied offline."""
import sys, numpy as np, time
task,kind=sys.argv[1],sys.argv[2]
import exp
W=exp.W; D=np.load(W+f'task_{task}.npz',allow_pickle=True); X=D['X']; XA=np.hstack([X,D['RP'],D['ECP']])
y=(D['yb'] if kind=='bin' else D['ym']).astype(int); g=D['groups'] if len(D['groups']) else None; K=len(np.unique(y)); t0=time.time()
def level0(Xtr,ytr,gtr,Xte):
    names=list(exp.bases(K)); oof=np.zeros((len(ytr),len(names)*K),np.float32); te=np.zeros((len(Xte),len(names)*K),np.float32)
    for a,b in exp.splitter(ytr,gtr,3,seed=7):
        for j,(n,m) in enumerate(exp.bases(K).items()):
            m.fit(Xtr[a],ytr[a]); cl=m.steps[-1][1].classes_ if hasattr(m,'steps') else m.classes_
            P=np.zeros((len(b),K)); P[:,cl]=m.predict_proba(Xtr[b]); oof[b,j*K:(j+1)*K]=P
    for j,(n,m) in enumerate(exp.bases(K).items()):
        m.fit(Xtr,ytr); te[:,j*K:(j+1)*K]=m.predict_proba(Xte)
    return oof,te
for fi,(tr,te) in enumerate(exp.splitter(y,g,5)):
    gtr=g[tr] if g is not None else None
    o0,t0_=level0(X[tr],y[tr],gtr,X[te]); o1,t1_=level0(XA[tr],y[tr],gtr,XA[te])
    np.savez_compressed(W+f'L0_{task}_{kind}_f{fi}.npz',oof0=o0,te0=t0_,oof1=o1,te1=t1_,ytr=y[tr],yte=y[te],gtr=gtr if gtr is not None else np.zeros(0),te_idx=te)
    print(task,kind,'L0 fold',fi,round(time.time()-t0),flush=True)
