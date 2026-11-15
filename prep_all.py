import numpy as np, pandas as pd, pickle, re, config
from sabp import record_profile, entity_profile
from sklearn.feature_extraction.text import HashingVectorizer
W=config.W; rng=np.random.RandomState(config.SEED)
def save(name,X,RP,ECP,yb,ym,groups,classes,info):
    np.savez_compressed(W+f'task_{name}.npz',X=X.astype(np.float32),RP=RP,ECP=ECP if ECP is not None else np.zeros((len(X),0),np.float32),
        yb=yb if yb is not None else np.zeros(0), ym=ym if ym is not None else np.zeros(0), groups=groups if groups is not None else np.zeros(0),classes=np.array(classes,dtype=str))
    pickle.dump(info,open(W+f'info_{name}.pkl','wb')); print(name,X.shape,'RP',RP.shape,'ECP',None if ECP is None else ECP.shape,info)
# ---------- A1 AndMal2020 static
X=np.load(W+'andmal_static_X.npy'); m=pickle.load(open(W+'andmal_static_meta.pkl','rb'))
n_full=len(X); ok=~pd.Series(m['hash']).duplicated().values          # drop duplicate app hashes
X=X[ok]; y=m['y'][ok]
if config.STATIC_MAX_APPS and config.STATIC_MAX_APPS<len(y):         # optional class-stratified sample, floor of 300 apps per class
    N=config.STATIC_MAX_APPS
    sub=np.sort(np.concatenate([rng.choice(np.where(y==c)[0],size=min((y==c).sum(),max(300,int(round((y==c).sum()*N/len(y))))),replace=False) for c in np.unique(y)]))
    X=X[sub]; y=y[sub]
Xs=X
cl=sorted(np.unique(y)); ym=np.array([cl.index(v) for v in y]); yb=(y!='Benign').astype(int)
RP=record_profile(Xs)
save('andmal_static',Xs,RP,None,yb,ym,None,cl,{'n_full':n_full,'d_full':m['d_full'],'d_kept':Xs.shape[1],'n_used':len(y),'malicious_frac':float(yb.mean()),'dup_hash_full':int((~ok).sum()),'class_counts':pd.Series(y).value_counts().to_dict()})
# ---------- A2 AndMal2020 dynamic
d=pd.read_pickle(W+'andmal_dyn.pkl'); fc=[c for c in d.columns if c not in('Hash','Category','Family','Label')]
Xd=d[fc].to_numpy(np.float32); nun=(Xd.std(0)>0); Xd=Xd[:,nun]
cl=sorted(d.Category.unique()); ym=d.Category.map({c:i for i,c in enumerate(cl)}).values
RP=record_profile(Xd); ECP=entity_profile(RP,d.Hash.values)
save('andmal_dyn',Xd,RP,ECP,None,ym,pd.factorize(d.Hash)[0],cl,{'n':len(d),'d':int(nun.sum()),'d_const_removed':int((~nun).sum()),'n_hash':d.Hash.nunique(),'n_family':d.Family.nunique()})
# ---------- A3 CICMalDroid2020
d=pd.read_csv(config.find('*feature_vectors_syscallsbinders_frequency_5_Cat.csv')); n0=len(d); d=d.drop_duplicates().reset_index(drop=True)
names={1:'Adware',2:'Banking',3:'SMS',4:'Riskware',5:'Benign'}
Xm=d.drop(columns='Class').to_numpy(np.float32); nun=Xm.std(0)>0; Xm=Xm[:,nun]
cl=[names[i] for i in range(1,6)]; ym=d.Class.values-1; yb=(d.Class.values!=5).astype(int)
save('maldroid',Xm,record_profile(Xm),None,yb,ym,None,cl,{'n_raw':n0,'n':len(d),'d':int(nun.sum()),'malicious_frac':float(yb.mean()),'class_counts':d.Class.map(names).value_counts().to_dict()})
# ---------- A4 APT flows
d=pd.read_pickle(W+'apt_flow.pkl'); n0=len(d)
idc=['publicIP','FlowID','SrcIP','DstIP','Timestamp','SrcPort','Label']
num=d.drop(columns=idc).replace([np.inf,-np.inf],np.nan)
const=[c for c in num if num[c].nunique(dropna=True)<=1]; num=num.drop(columns=const).fillna(0)
dup=d.drop(columns=['Label']).duplicated().values
Xa=num.to_numpy(np.float32); RPa=record_profile(Xa); ECPa=entity_profile(RPa,d.publicIP.values)
per=d.groupby('publicIP').size(); print('flows/IP quantiles',per.quantile([.1,.5,.9,.99,1]).to_dict())
CAP=config.APT_FLOWS_PER_IP or 10**9; r=rng.rand(len(d)); rk=pd.Series(r).groupby(d.publicIP.values).rank(method='first').values
s=np.where((rk<=CAP)&(~dup))[0]
yb=(d.Label.values=='APT').astype(int)
save('apt_flow',Xa[s],RPa[s],ECPa[s],yb[s],None,pd.factorize(d.publicIP.values[s])[0],['BENIGN','APT'],
     {'n_full':n0,'n_used':len(s),'d':Xa.shape[1],'const_removed':const,'n_ip':int(d.publicIP.nunique()),'ip_apt':int(d[d.Label=='APT'].publicIP.nunique()),'malicious_frac_full':float(yb.mean()),'malicious_frac_used':float(yb[s].mean()),'cap':CAP,'features':list(num.columns)})
del d,num,Xa,RPa,ECPa
# ---------- A5 Unraveled host logs
E=pd.read_pickle(W+'unraveled_events.pkl')
def mask(t):
    t=re.sub(r'0x[0-9a-fA-F]+','HEX',t); t=re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b','IPADDR',t); t=re.sub(r'\{[0-9A-Fa-f-]{20,}\}','GUID',t)
    t=re.sub(r'S-1-[\d-]+','SID',t); return re.sub(r'\d+','N',t)
txt=(E.chan+'_SRC '+E.event_id.map(lambda s:'EID_'+re.sub(r'\W','_',s)[:30])+' '+E.text.map(mask))
hv=HashingVectorizer(n_features=256,alternate_sign=False,norm=None,token_pattern=r'[A-Za-z_][A-Za-z_\.\-]{1,}',lowercase=True)
H=hv.transform(txt)
chan=pd.get_dummies(E.chan).to_numpy(np.float32)
RP=record_profile(H.tocsr())
win=E.chan+'|'+E.ts.dt.floor('5min').astype(str); ECP=entity_profile(RP,win.values)
grp=E.chan+'|'+E.ts.dt.floor('30min').astype(str)
pos=np.where(E.y.values==1)[0]; neg=rng.choice(np.where(E.y.values==0)[0],config.UNRAVELED_BENIGN,replace=False); s=np.sort(np.concatenate([pos,neg]))
Xu=np.hstack([H[s].toarray().astype(np.float32),chan[s]])
save('unraveled',Xu,RP[s],ECP[s],E.y.values[s],None,pd.factorize(grp.values[s])[0],['Benign','APT'],
     {'n_events_full':len(E),'n_attack_full':int(E.y.sum()),'prevalence_full':float(E.y.mean()),'n_used':len(s),'prevalence_used':float(E.y.values[s].mean()),'pos_groups':int(grp[E.y==1].nunique()),'channels':pd.crosstab(E.chan,E.y).to_dict()})
