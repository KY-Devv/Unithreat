"""Step 1: read the raw files once and store compact intermediate files in WORK_DIR."""
import numpy as np, pandas as pd, pickle, config
W = config.W
try:
    import pyarrow.parquet as pqa
    def columns_of(path): return pqa.ParquetFile(path).schema.names
    def read(path, cols=None): return pd.read_parquet(path, columns=cols)
except ImportError:                      # fallback used in the original study (needs snap.so, Linux only)
    import pq
    def columns_of(path): return [e[4].decode() for e in pq.meta(path)[2][1:]]
    def read(path, cols=None): return pq.read(path, columns=None if cols is None else set(cols))[cols] if cols else pq.read(path)

# --- CCCS-CIC-AndMal2020, dynamic
read(config.find('*cicandmal2020-dynamic.parquet')).to_pickle(W + 'andmal_dyn.pkl'); print('dynamic done', flush=True)

# --- APT flows
pd.read_csv(config.find('*flowFeatures.csv'), low_memory=False).to_pickle(W + 'apt_flow.pkl'); print('apt flows done', flush=True)

# --- CCCS-CIC-AndMal2020, static: all rows; keep attributes that are non-zero in at least 0.1% of the apps
f = config.find('*cicandmal2020-static.parquet')
names = columns_of(f); feat = [n for n in names if n not in ('Label', 'F0')]
lab = read(f, ['Label', 'F0'])
kept, kept_names = [], []
for i in range(0, len(feat), 400):
    cols = feat[i:i + 400]
    a = read(f, cols)[cols].to_numpy(dtype=np.float32, na_value=0)
    nz = (a != 0).mean(0); k = nz >= 0.001
    kept.append(a[:, k]); kept_names += [c for c, kk in zip(cols, k) if kk]
    print(f'static columns {i + len(cols)}/{len(feat)}, kept so far {len(kept_names)}', flush=True)
X = np.hstack(kept)
np.save(W + 'andmal_static_X.npy', X)
pickle.dump({'y': lab.Label.values, 'hash': lab.F0.values, 'feat': kept_names, 'd_full': len(feat)}, open(W + 'andmal_static_meta.pkl', 'wb'))
print('static done', X.shape)
