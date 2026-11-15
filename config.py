"""All settings for the rerun. Edit the two folders, then run:  python run_all.py"""
import os
DATA_DIR = r'C:\UniThreat\data'            # folder that holds the two .parquet files, the MalDroid .csv, flowFeatures.csv and the ten cleaned_*.csv files
WORK_DIR = r'C:\UniThreat\work'           # empty folder for intermediate files and results (needs about 5 GB free)
N_JOBS   = 4                              # CPU cores to use (use the number of physical cores of your laptop)

STATIC_MAX_APPS  = None      # None = all 357,805 apps; or a number such as 100000 for a class-stratified sample
APT_FLOWS_PER_IP = 1000      # cap on flows per remote IP (the paper uses 1000); None = all flows (needs about 32 GB RAM)
UNRAVELED_BENIGN = 30000     # benign log events sampled next to all attack events
KNN_MAX_TRAIN    = 100000    # KNN is fitted on at most this many training records (speed); None = no limit
SEED = 42

W = os.path.join(WORK_DIR, '')
os.makedirs(WORK_DIR, exist_ok=True)
def find(pattern):
    import glob
    hits = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not hits: raise FileNotFoundError(f'No file matching {pattern} in {DATA_DIR}')
    return hits[0]
