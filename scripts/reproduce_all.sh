#!/usr/bin/env bash
# Every training run behind the paper. Each run trains on the full data for 10 epochs on one GPU and also
# scores the controlled 6/24/96-hour outage of one target. Run sections selectively, e.g.:
#   bash scripts/reproduce_all.sh otarf baselines
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="2021 2022 2023"; DATASETS="beijing_multisite kdd2018_london shanghai_confirmation"
SECTIONS="${*:-otarf baselines ablation inspect extended outages nonorm events24}"
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
done_run() { [ -f "runs/$1/$2/h$3/s$4/result.json" ]; }

if want otarf; then       # OTARF, H = 12/24/48 (natural, masked, tail, native-unit and cost tables)
  for d in $DATASETS; do for h in 12 24 48; do for s in $SEEDS; do
    done_run OTARF-otarf $d $h $s || python scripts/train_otarf.py --dataset $d --horizon $h --seed $s
  done; done; done
fi
if want baselines; then   # SOFTS and TimeFilter: official model code, OTARF training recipe, mask/age variates
  for m in SOFTS TimeFilter; do for d in $DATASETS; do for h in 12 24 48; do for s in $SEEDS; do
    done_run $m-indicator $d $h $s || python scripts/train_baseline.py --model $m --recipe indicator --dataset $d --horizon $h --seed $s
  done; done; done; done
fi
if want ablation; then    # the five variants of the ablation table, H = 24
  for v in no_anchor no_cross mean_pool state_blind state_blind_no_aug; do for d in $DATASETS; do for s in $SEEDS; do
    done_run OTARF-$v $d 24 $s || python scripts/train_otarf.py --dataset $d --horizon 24 --seed $s --variant $v
  done; done; done
fi
if want official; then    # optional, not used in the paper: the baselines with their own official trainers
  for m in SOFTS TimeFilter; do for d in $DATASETS; do for h in 12 24 48; do for s in $SEEDS; do
    done_run $m-official $d $h $s || python scripts/train_baseline.py --model $m --recipe official --dataset $d --horizon $h --seed $s
  done; done; done; done
fi
if want inspect; then     # inference only: attention by source group, knockout, 96 h case study (needs otarf + baselines)
  for d in $DATASETS; do
    [ -f "runs/inspect/$d/summary.json" ] || python scripts/inspect_otarf.py --dataset $d
  done
fi
if want extended; then    # iTransformer, CRIB and OTARF with the peak-weighted loss, H = 12/24/48
  for m in iTransformer CRIB; do for d in $DATASETS; do for h in 12 24 48; do for s in $SEEDS; do
    done_run $m-indicator $d $h $s || python scripts/train_extended_baseline.py --model $m --dataset $d --horizon $h --seed $s
  done; done; done; done
  for d in $DATASETS; do for h in 12 24 48; do for s in $SEEDS; do
    done_run OTARF-peak $d $h $s || python scripts/train_otarf.py --dataset $d --horizon $h --seed $s --peak
  done; done; done
fi
if want outages; then     # inference only: several channels or whole stations out at once, threshold events, naive references
  for d in $DATASETS; do
    for m in persistence seasonal; do for h in 12 24 48; do
      [ -f "runs/naive-$m/$d/h$h/s0/outages/events.json" ] || python scripts/evaluate_outages.py --model $m --dataset $d --horizon $h
    done; done
    for m in OTARF OTARF-peak SOFTS TimeFilter iTransformer CRIB; do for s in $SEEDS; do
      case $m in OTARF) tag=OTARF-otarf;; OTARF-peak) tag=OTARF-peak;; *) tag=$m-indicator;; esac
      [ -f "runs/$tag/$d/h24/s$s/outages/events.json" ] || python scripts/evaluate_outages.py --model $m --dataset $d --seed $s
    done; done
  done
fi
if want nonorm; then      # mechanism test: the four models with window normalization switched off, H = 24
  for m in SOFTS TimeFilter iTransformer CRIB; do for d in $DATASETS; do for s in $SEEDS; do
    done_run $m-nonorm $d 24 $s || python scripts/train_extended_baseline.py --model $m --dataset $d --horizon 24 --seed $s --no-window-norm
  done; done; done
fi
if want events24; then    # inference only: exceedance events on the 24-hour mean, H = 24
  for m in OTARF OTARF-peak SOFTS TimeFilter iTransformer CRIB; do for d in $DATASETS; do for s in $SEEDS; do
    case $m in OTARF) tag=OTARF-otarf;; OTARF-peak) tag=OTARF-peak;; *) tag=$m-indicator;; esac
    [ -f "runs/$tag/$d/h24/s$s/events24/summary.json" ] || python scripts/evaluate_events24.py --model $m --dataset $d --seed $s
  done; done; done
fi
python scripts/summarize.py && python scripts/summarize_extensions.py && python scripts/summarize_v6.py && python scripts/make_figures.py
