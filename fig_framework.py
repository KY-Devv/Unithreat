import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5})
fig,ax=plt.subplots(figsize=(10,6.2)); ax.set_xlim(0,100); ax.set_ylim(0,62); ax.axis('off')
INK='#0b0b0b'; SUB='#52514e'; EDGE='#8a8983'
def box(x,y,w,h,title,lines=(),fc='#ffffff',bold=True,fs=8.5):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=1.0',fc=fc,ec=EDGE,lw=0.9))
    ax.text(x+w/2,y+h-1.3,title,ha='center',va='top',fontsize=fs,fontweight='bold' if bold else 'normal',color=INK)
    for i,l in enumerate(lines): ax.text(x+w/2,y+h-4.3-i*2.35,l,ha='center',va='top',fontsize=7.3,color=SUB)
def arrow(x1,y1,x2,y2): ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='-|>',color=SUB,lw=0.9,shrinkA=0,shrinkB=0))
def band(y,h,label):
    ax.add_patch(FancyBboxPatch((0.5,y),99,h,boxstyle='round,pad=0.1,rounding_size=0.8',fc='#f3f2ee',ec='none'))
    ax.text(1.6,y+h/2,label,rotation=90,ha='center',va='center',fontsize=7.6,color=SUB,fontweight='bold')
band(50.5,11,'L1  Adapters'); band(39.3,9.7,'L2  Hygiene'); band(27,11,'L3  Level-0'); band(15.5,10,'L4  Gate'); band(1,13,'L5  Decision')
src=[('A1 Static app','CCCS-CIC-AndMal2020','331 manifest/code attrs'),('A2 Dynamic app','CCCS-CIC-AndMal2020','126 runtime counters'),('A3 Syscall/binder','CICMalDroid2020','470 call frequencies'),('A4 Network flow','APT flow dataset','66 flow statistics'),('A5 Host log','Unraveled','256 hashed tokens')]
for i,(t,a,b) in enumerate(src):
    x=4+i*19.2; box(x,51.5,17.6,9,t,(a,b)); arrow(x+8.8,51.5,x+8.8,48.6)
box(4,40.2,94.4,8.2,'Leakage-aware data hygiene',('de-duplication · constant / non-finite feature removal · entity key per record','(app hash, remote IP, log channel × time window) · entity-grouped stratified folds'))
arrow(51.2,40.5,51.2,37.6)
box(4,28,56.5,9,'Two feature views',('View 0: native adapter features','View 1: native + SABP (record profile, 20; entity context, 41)'))
box(63,28,35.4,9,'Base learners per view',('RF · Extra-Trees · Hist. Gradient Boosting · LR','→ out-of-fold class probabilities'))
arrow(60.5,32.5,63,32.5); arrow(80.7,28,80.7,25.1)
box(4,16.2,94.4,8.8,'Validation gate + meta-learner (level 1)',('gate: pick view ∈ {0, 1} and class weighting ∈ {balanced, none} by cross-fitted MCC on training folds only','meta-learner: logistic regression on the selected out-of-fold probabilities'))
arrow(51.2,16.5,51.2,13.7)
box(4,1.6,45,11.8,'Unified decision layer',('per-domain isotonic calibration (cross-fitted)','→ one risk score on a common probability scale','→ one operating threshold for all telemetry'))
box(53.4,1.6,45,11.8,'Outputs',('joint verdict: benign / Android malware / APT activity','native category (14-, 15-, 5-class) where labels exist','cross-domain probe: SABP-only transfer tests'))
arrow(49,7.6,53.4,7.6)
import os, config; os.makedirs(os.path.join(config.WORK_DIR,'figures'),exist_ok=True)
plt.savefig(os.path.join(config.WORK_DIR,'figures','fig1_framework.png'),dpi=220,bbox_inches='tight',facecolor='white')
