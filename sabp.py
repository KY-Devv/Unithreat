"""Schema-Agnostic Behavioural Profiling (SABP).
RP  : stateless record profile  - 20 descriptors of how activity is distributed across a record's features.
ECP : entity context profile    - log(#records), mean and std of RP over all records sharing an entity key."""
import numpy as np, pandas as pd
RP_NAMES=['breadth','log_total','mean','std','skew','kurt','max','entropy','gini','top1','top5',
          'q10','q25','q50','q75','q90','frac_gt1','frac_gt3','frac_gt6','eff_n']
def record_profile(X, chunk=50000):
    out=[]
    for i in range(0,X.shape[0],chunk):
        A=X[i:i+chunk]; A=A.toarray() if hasattr(A,'toarray') else np.asarray(A)
        V=np.log1p(np.abs(np.nan_to_num(A.astype(np.float64),posinf=0,neginf=0)))
        d=V.shape[1]; tot=V.sum(1); mu=V.mean(1); sd=V.std(1); z=(V-mu[:,None])/(sd[:,None]+1e-9)
        P=V/(tot[:,None]+1e-12); H=-(P*np.log(P+1e-12)).sum(1)
        S=np.sort(V,1); idx=np.arange(1,d+1)
        gini=(2*(S*idx).sum(1)/(d*tot+1e-12))-(d+1)/d
        q=np.quantile(V,[.1,.25,.5,.75,.9],axis=1)
        out.append(np.column_stack([(V>0).mean(1),np.log1p(tot),mu,sd,(z**3).mean(1),(z**4).mean(1),S[:,-1],
            H/np.log(d),np.clip(gini,0,1),S[:,-1]/(tot+1e-12),S[:,-5:].sum(1)/(tot+1e-12),q[0],q[1],q[2],q[3],q[4],
            (V>1).mean(1),(V>3).mean(1),(V>6).mean(1),np.exp(H)/d]).astype(np.float32))
    return np.vstack(out)
def entity_profile(RP, keys):
    df=pd.DataFrame(RP); g=df.groupby(np.asarray(keys),sort=False)
    cnt=g[0].transform('size').to_numpy(); m=g.transform('mean').to_numpy(); s=g.transform('std').fillna(0).to_numpy()
    return np.column_stack([np.log1p(cnt),m,s]).astype(np.float32)
ECP_NAMES=['ent_logn']+['ent_mean_'+n for n in RP_NAMES]+['ent_std_'+n for n in RP_NAMES]
