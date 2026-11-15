"""Adapter A5: parse the ten Unraveled host-log CSVs into one event table (source, ts, event_id, text, label)."""
import pandas as pd, numpy as np, glob, os, re, config
U=os.path.join(config.DATA_DIR,'')
TS_WIN=re.compile(r'^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2}:\d{2} [AP]M$')
TS_SYS=re.compile(r'^([A-Z][a-z]{2})\s+(\d{1,2}) (\d{2}:\d{2}:\d{2}) (\S+) (.*)$',re.S)
EMB=re.compile(r',([A-Za-z][^,\n]*),([A-Za-z][^,\n]*),([A-Za-z][^,\n]*),(None|APT|SH|CS|IDS|[A-Za-z ]+)\s*(?:\n|$)')
AUD=re.compile(r'msg=audit\((\d+)\.')
rows=[]
def lab(stage,sig):
    s=str(stage); g=str(sig)
    if s in('nan','None',''): return None
    return 0 if s=='Benign' else 1
for f in sorted(glob.glob(U+'*cleaned_*.csv')):
    src=os.path.basename(f).split('cleaned_')[1][:-4]
    d=pd.read_csv(f,low_memory=False,dtype=str)
    n0=len(d); kept=0
    if 'LogEvent' in d:
        for ev,act,st,sg in zip(d.LogEvent,d.Activity,d.Stage,d.Signature):
            y=lab(st,sg)
            if y is None or not isinstance(ev,str): continue
            m=TS_SYS.match(ev)
            if m:
                ts=pd.Timestamp(f'2021 {m.group(1)} {m.group(2)} {m.group(3)}'); body=m.group(5)
                prog=re.split(r'[\[:]',body,maxsplit=1)[0].strip()
            else:
                a=AUD.search(ev)
                if a: ts=pd.Timestamp(int(a.group(1)),unit='s'); body=ev; mm=re.match(r'type=(\S+)',ev); prog=mm.group(1) if mm else 'audit'
                else: continue   # fragment line, not a full event
            rows.append((src,ts,prog,body,y,str(st),str(act))); kept+=1
    else:
        for r in d.itertuples(index=False):
            c=[x if isinstance(x,str) else '' for x in r]
            ti=next((i for i in range(3) if TS_WIN.match(c[i].strip())),None)
            if ti is None: continue
            ts=pd.to_datetime(c[ti].strip(),format='%m/%d/%Y %I:%M:%S %p')
            eid=c[ti+2].strip(); task=c[ti+3].strip(); msg=c[ti+4] if ti+4<len(c) else ''
            if not eid.isdigit(): continue
            st,sg,act=c[-3] if False else c[6],c[8],c[5]
            y=lab(st if st else 'nan',sg)
            if ti==1:  # aligned row: labels embedded in the message's first line
                m=EMB.search(msg.split('\n')[0]+'\n')
                if not m: continue
                act,st=m.group(1),m.group(2); y=0 if st=='Benign' else 1
                msg=EMB.sub('\n',msg,count=1)
            elif y is None:
                continue
            else:
                msg=c[1]+' '+task+' '+c[ti+4]
                m=EMB.search(msg.split('\n')[0]+'\n')
                if m: msg=EMB.sub('\n',msg,count=1)
            rows.append((src,ts,eid,(c[2]+' '+task+' '+msg) if ti==1 else msg,y,st,act)); kept+=1
    print(f'{src:38s} rows={n0:7d} parsed_events={kept:7d}')
E=pd.DataFrame(rows,columns=['source','ts','event_id','text','y','stage','activity'])
# remove any residual label strings
E['text']=E.text.str.replace(r',(Normal|Valid Accounts|Network Service Discovery|Maintain Access)[^\n]*','',regex=True)
n=len(E); E=E.drop_duplicates(['source','ts','event_id','text']).reset_index(drop=True)
print('events',n,'-> after exact dedup',len(E)); 
# the -ansible / -user / base security logs are exports of the same Windows Security channel: dedup across them
E['chan']=E.source.str.replace(r'windows-securityevents.*','windows-security',regex=True)
E=E.sort_values('y',ascending=False).drop_duplicates(['chan','ts','event_id','text']).sort_values(['chan','ts']).reset_index(drop=True)
print('after cross-export dedup',len(E)); print(pd.crosstab(E.chan,E.y).to_string()); print(E[E.y==1].groupby(['chan','stage','activity']).size().to_string())
print('time span',E.ts.min(),E.ts.max()); print('leak check:',E.text.str.contains('Lateral Movement|,APT|Benign,',regex=True).sum())
E.to_pickle(config.W+'unraveled_events.pkl')
