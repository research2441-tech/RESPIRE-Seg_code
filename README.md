# RESPIRE-Seg

Code accompanying the manuscript on physics-faithful k-space respiratory simulation, motion-state-aware
cardiac MRI segmentation and severity-conditioned conformal contours for free-breathing cine CMR.

Every number in the paper is produced by the scripts in this folder. The folder is flat on purpose: run
everything from inside it.

## Contents

| File | Purpose |
|---|---|
| `breathing.py` | Respiration-belt traces (NeuroKit2 open data) and the four breathing protocols |
| `kspace_sim.py` | Segmented-cine multi-coil k-space corruption engine, severity index, baseline corruptions |
| `prepare_cohorts.py` | Converts ACDC / M&Ms-2 NIfTI folders to compact cohort files |
| `prep_cmrx.py` | Builds the real-motion CMRxMotion evaluation bank (needs the challenge release) |
| `data_prep.py` | Reads cohort files and harmonises labels (1 LV, 2 MYO, 3 RV) |
| `build_bank.py` | Pre-computes every training / calibration / test corruption bank |
| `models.py` | U-Net, nnU-Net-style 2-D, Attention U-Net, Sharp U-Net, CNN–Transformer, multi-task U-Net, RESPIRE-Seg |
| `losses.py` | Dice + CE, signed-distance targets, topology prior, paired consistency (evaluated, off by default) |
| `train.py` / `predict.py` | Training and evaluation (per-slice records and per-volume metrics) |
| `conformal.py` / `analysis.py` | Conformal risk control of contour bands, normalised EF intervals, statistics |
| `cmrx_recal.py` | Recalibration on real CMRxMotion data (subject-level splits, risk per radiologist grade) |
| `final_results.py` | All tables, statistics and result figures |
| `fig_engine.py`, `fig_architecture.py`, `fig_qualitative.py`, `plotstyle.py` | Figures |
| `make_jobs.py`, `run_queue.py` | The exact experiment queue that was run (resumable) |

## Data

* ACDC (Bernard et al., IEEE TMI 2018) and M&Ms-2 (Martín-Isla et al., IEEE JBHI 2023): obtain from the challenge
  organisers, then `python prepare_cohorts.py acdc <root> data/acdc.bin.gz` and the same for `mnms2`.
* CMRxMotion training release (Wang et al., Medical Image Analysis 2026): request access from the organisers,
  unzip, then `python prep_cmrx.py "<path to CMRxMotion Training Dataset>"`.
* Respiration traces: copy `bio_resting_5min_100hz.csv`, `bio_resting_8min_100hz.csv`,
  `bio_eventrelated_100hz.csv` and `rsp_1000hz.txt` from the NeuroKit2 `data/` folder to `data/rsp/`.

Paths default to `/home/claude/data` and `/home/claude/runs`; override with `RESPIRE_DATA` and `RESPIRE_RUNS`.

## Reproduce

```bash
pip install -r requirements.txt
python prep_cmrx.py "<path to CMRxMotion Training Dataset>"
for b in acdc_train_phys acdc_val acdc_cal acdc_test acdc_test_clean acdc_train_clean \
         acdc_train_imgwarp acdc_train_generic acdc_test_stress; do python build_bank.py $b; done
python make_jobs.py && python run_queue.py jobs.txt 2
python final_results.py
python fig_engine.py figs/fig_engine.png && python fig_architecture.py figs/fig_framework.png && python fig_qualitative.py figs/fig_qualitative.png
```

The study was run on a two-core CPU (one thread per job); the full queue takes several hours.

## Licence

MIT for the code. The datasets keep their own licences and are not redistributed here.
