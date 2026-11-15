import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd, json, os, config
from matplotlib.colors import LinearSegmentedColormap
W=config.W; F=os.path.join(config.WORK_DIR,'figures',''); os.makedirs(F,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#8a8983','axes.labelcolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,'grid.color':'#e7e6e1','grid.linewidth':0.6,'axes.axisbelow':True})
C=['#2a78d6','#eb6834','#1baf7a']; INK='#0b0b0b'
seq=LinearSegmentedColormap.from_list('b',['#eef5fd','#cde2fb','#86b6ef','#2a78d6','#104281'])
TN={'andmal_static|bin':'AndMal static\n(binary)','maldroid|bin':'MalDroid\n(binary)','apt_flow|bin':'APT flows\n(binary)','unraveled|bin':'Unraveled logs\n(binary)','andmal_static|multi':'AndMal static\n(15-class)','andmal_dyn|multi':'AndMal dynamic\n(14-class)','maldroid|multi':'MalDroid\n(5-class)'}
S=pd.read_csv(W+'summary.csv'); M=['LR','SVM','DT','NB','KNN','RF','UniThreat']
# Fig 2 heatmap of MCC
T=S[S.model.isin(M)].pivot(index='model',columns='tk',values='mcc_mean').reindex(M)[list(TN)]
fig,ax=plt.subplots(figsize=(9,3.6)); ax.grid(False); im=ax.imshow(T.values,cmap=seq,vmin=0.2,vmax=1,aspect='auto')
ax.set_xticks(range(7),[TN[c] for c in T.columns],fontsize=8); ax.set_yticks(range(len(M)),['UniThreat (proposed)' if m=='UniThreat' else m for m in M])
for i in range(T.shape[0]):
    for j in range(T.shape[1]):
        v=T.values[i,j]; best=v==np.nanmax(T.values[:,j]); ax.text(j,i,f'{v:.3f}',ha='center',va='center',fontsize=8.5,color='white' if v>0.72 else INK,fontweight='bold' if best else 'normal')
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,7),minor=True); ax.set_yticks(np.arange(-.5,len(M)),minor=True); ax.grid(which='minor',color='white',linewidth=2); ax.tick_params(which='both',length=0)
cb=fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02); cb.set_label('MCC',fontsize=8); cb.outline.set_visible(False)
plt.savefig(F+'fig2_mcc_heatmap.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close()
# Fig 3 leakage
L=pd.read_csv(W+'res_leakage.csv').groupby(['task','protocol']).mcc.agg(['mean','std']).reset_index()
tasks=['apt_flow','andmal_dyn','unraveled']; lab=['APT flows\n(entity = remote IP)','AndMal dynamic\n(entity = app hash)','Unraveled logs\n(entity = channel × 30 min)']
fig,ax=plt.subplots(figsize=(6.6,3.3)); x=np.arange(3); w=0.36
for k,(p,nm) in enumerate([('record-random','Record-level random folds'),('entity-grouped','Entity-grouped folds')]):
    d=L[L.protocol==p].set_index('task').loc[tasks]; b=ax.bar(x+(k-.5)*(w+.03),d['mean'],w,yerr=d['std'],color=C[k],label=nm,error_kw=dict(ecolor='#52514e',lw=0.8,capsize=2))
    for xi,v,sd in zip(x+(k-.5)*(w+.03),d['mean'],d['std']): ax.text(xi,v+sd+0.012,f'{v:.3f}',ha='center',fontsize=8,color=INK)
ax.set_xticks(x,lab); ax.set_ylabel('MCC (Random Forest, native features)'); ax.set_ylim(0,1.15); ax.grid(axis='x',visible=False); ax.legend(frameon=False,loc='upper center',ncol=2,fontsize=8.5,bbox_to_anchor=(0.5,1.12))
plt.savefig(F+'fig3_leakage.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close()
# Fig 4 cross-domain
U=pd.read_csv(W+'res_universal.csv'); R=pd.read_csv(W+'res_unified_layer.csv'); D=['andmal_static','maldroid','apt_flow','unraveled']; DL=['AndMal static','MalDroid','APT flows','Unraveled logs']
spec=S[S.model=='UniThreat'].set_index('tk').loc[[d+'|bin' for d in D],'auc_mean']; pooled=U[U.setting.str.startswith('pooled')].groupby('test').auc.mean().loc[D]; lodo=U[U.setting=='leave-one-domain-out'].set_index('test').loc[D,'auc']
fig,ax=plt.subplots(figsize=(7.8,3.4)); x=np.arange(4); w=0.27
for k,(v,nm) in enumerate([(spec,'Domain specialist (UniThreat)'),(pooled,'One shared model on SABP profile, domain seen'),(lodo,'Shared model, domain held out')]):
    ax.bar(x+(k-1)*(w+.02),v.values,w,color=C[k],label=nm)
    for xi,val in zip(x+(k-1)*(w+.02),v.values): ax.text(xi,val+0.015,f'{val:.3f}',ha='center',fontsize=7.5,color=INK)
ax.axhline(0.5,color='#52514e',lw=0.9,ls='--'); pass
ax.set_xticks(x,DL); ax.set_ylabel('ROC-AUC'); ax.set_ylim(0,1.22); ax.set_yticks([0,.25,.5,.75,1]); ax.grid(axis='x',visible=False); ax.legend(frameon=False,loc='upper center',ncol=1,fontsize=8,bbox_to_anchor=(0.5,1.3))
plt.savefig(F+'fig4_crossdomain.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close()
# Fig 5 unified layer: ECE + joint confusion matrix
J=json.load(open(W+'joint.json')); cm=np.array(J['cm'])
fig,(a,b)=plt.subplots(1,2,figsize=(9.4,3.4),gridspec_kw={'width_ratios':[1.25,1]})
dd=D+['ALL']; x=np.arange(5); w=0.36
for k,(s,nm) in enumerate([('raw','Before calibration'),('cal','After per-domain isotonic calibration')]):
    v=R[R.score==s].set_index('domain').loc[dd,'ece'].values; a.bar(x+(k-.5)*(w+.03),v,w,color=C[k],label=nm)
    for xi,val in zip(x+(k-.5)*(w+.03),v): a.text(xi,val+0.00015,f'{val:.4f}',ha='center',fontsize=7,color=INK)
a.set_xticks(x,DL+['All domains\npooled'],fontsize=8); a.set_ylabel('Expected calibration error'); a.grid(axis='x',visible=False); a.legend(frameon=False,fontsize=8,loc='upper right'); a.set_ylim(0,0.0105); a.set_title('(a) Score comparability across domains',fontsize=9,loc='left',color=INK)
b.grid(False); rn=cm/cm.sum(1,keepdims=True); b.imshow(rn,cmap=seq,vmin=0,vmax=1); cl=['Benign','Android\nmalware','APT\nactivity']
b.set_xticks(range(3),cl,fontsize=8); b.set_yticks(range(3),cl,fontsize=8); b.set_xlabel('Predicted'); b.set_ylabel('True')
for i in range(3):
    for j in range(3): b.text(j,i,f'{cm[i,j]:,}\n({rn[i,j]*100:.1f}%)',ha='center',va='center',fontsize=8,color='white' if rn[i,j]>0.6 else INK)
for s in b.spines.values(): s.set_visible(False)
b.tick_params(length=0); b.set_title('(b) Joint three-way verdict',fontsize=9,loc='left',color=INK)
plt.tight_layout(); plt.savefig(F+'fig5_unified.png',dpi=220,bbox_inches='tight',facecolor='white'); plt.close()
