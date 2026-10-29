#!/bin/bash
# Queue B: experiments derived from Queue A, run on the HI-Small discovery set.
cd /c/dev/aml-stream
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
$PY -I scripts/05_cascade.py --tag gbt_hi_small_released > logs/cascade_hi_released.log 2>&1
$PY -I scripts/05_cascade.py --tag gbt_hi_small_documented --max-day 9 > logs/cascade_hi_documented.log 2>&1
$PY -I scripts/06_attack.py --tag attack_hi_small > logs/attack_hi_small.log 2>&1
$PY -I scripts/07_service.py > logs/service.log 2>&1
for H in 0 1 3600 inf; do
  $PY -I scripts/04_gnn.py --tag gnn_hi_small_released --H $H --seed 0 --epochs 5 > logs/gnn_released_H$H.log 2>&1
done
echo QUEUE_B_DONE
