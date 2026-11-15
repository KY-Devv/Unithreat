# UniThreat

Code and results for the paper *"UniThreat: A Unified, Leakage-Aware Hybrid Stacked-Ensemble Framework for Joint
Detection of Android Malware and Advanced Persistent Threats"*.

The repository contains the modality adapters, Schema-Agnostic Behavioural Profiling (SABP), the validation-gated
stacked ensemble, the entity-grouped evaluation protocol, and the per-fold results behind every table and figure.
The datasets are public but are **not** included; download them from their original sources (see the paper's
reference list).

## Requirements
- Python 3.10 or newer
- `pip install -r requirements.txt`
- About 5 GB of free disk space; 16 GB RAM is enough for the default settings.

## Data: put these files in one folder (names may carry a prefix; they are found by pattern)
| File | Dataset |
|---|---|
| `cicandmal2020-static.parquet` | CCCS-CIC-AndMal2020, static |
| `cicandmal2020-dynamic.parquet` | CCCS-CIC-AndMal2020, dynamic |
| `feature_vectors_syscallsbinders_frequency_5_Cat.csv` | CICMalDroid2020 |
| `flowFeatures.csv` | APT flow dataset |
| ten `cleaned_*.csv` files | Unraveled host logs |

## Settings: edit `config.py` only
- `DATA_DIR` (data folder), `WORK_DIR` (empty output folder), `N_JOBS` (CPU cores).
- `STATIC_MAX_APPS`: `None` uses all 357,805 static apps; the paper used `100000`.
- `APT_FLOWS_PER_IP = 1000` is the setting used in the paper.

## Run
    python run_all.py            # everything
    python run_all.py static     # only the static Android tasks, then the summary

Finished steps are skipped when the command is run again.

| Step | Script | What it does |
|---|---|---|
| 1 | `load_raw.py` | reads the raw files into compact intermediate files |
| 2 | `prep_unraveled.py` | parses the Unraveled logs into one de-duplicated event table |
| 3 | `prep_all.py`, `sabp.py` | adapters A1-A5, SABP features, task files |
| 4 | `exp.py` | six baseline classifiers, five leakage-aware folds |
| 5 | `exp4.py` | base learners on two feature views, out-of-fold probabilities |
| 6 | `meta.py` | validation gate + meta-learner (UniThreat) and the fixed ablation configurations |
| 7 | `exp2.py` | leakage experiment, cross-domain experiments |
| 8 | `final.py` | summary tables, statistics, calibrated decision layer, joint verdict |
| 9 | `figs.py`, `fig_framework.py` | figures |

## Outputs (in `WORK_DIR`)
`summary.csv`, `all_folds.csv`, `stats.json`, `gate.csv`, `res_leakage.csv`, `res_universal.csv`,
`res_unified_layer.csv`, `joint.json`, `info_*.pkl`, and `figures/`.

`results_in_paper/` holds these outputs for the run reported in the paper
(100,164 static apps; at most 1,000 flows per remote IP; random seed 42).

## Notes
- The Unraveled benign sample is drawn at random after the other datasets are prepared, so its numbers can differ
  slightly from the paper when sample-size settings are changed.
- KNN is fitted on at most `KNN_MAX_TRAIN` training records for speed.
- `pq.py` and `snap.c` are a fallback Parquet reader for Linux machines without pyarrow
  (`gcc -O2 -shared -fPIC -o snap.so snap.c`); with pyarrow installed they are not used.
- The pipeline was run on Linux; the pyarrow reading path and Windows have not been tested.
