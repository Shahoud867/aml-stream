#!/bin/bash
# Queue A: gradient-boosted horizon sweeps on the Small datasets (CPU, about 1-2 hours in total).
cd /c/dev/aml-stream
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
HZ=0,1,60,600,3600,21600,86400,259200,inf
$PY -I scripts/02b_audit_all.py HI-Small LI-Small HI-Medium LI-Medium > results/tail_audit.log 2>&1
$PY -I scripts/03_sweep.py --tag gbt_hi_small_released --horizons $HZ --seeds 0,1,2 --neg-rate 0.25 --save-scores > logs/gbt_hi_small_released.log 2>&1
$PY -I scripts/03_sweep.py --tag gbt_hi_small_documented --max-day 9 --horizons $HZ --seeds 0,1,2 --neg-rate 0.25 --save-scores > logs/gbt_hi_small_documented.log 2>&1
$PY -I scripts/01_prepare.py LI-Small > logs/prepare_li_small.log 2>&1
$PY -I scripts/03_sweep.py --dataset LI-Small --tag gbt_li_small_released --horizons $HZ --seeds 0,1,2 --neg-rate 0.25 --save-scores > logs/gbt_li_small_released.log 2>&1
$PY -I scripts/03_sweep.py --dataset LI-Small --tag gbt_li_small_documented --max-day 9 --horizons $HZ --seeds 0,1,2 --neg-rate 0.25 --save-scores > logs/gbt_li_small_documented.log 2>&1
echo QUEUE_A_DONE
