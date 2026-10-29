#!/bin/bash
# Queue C: feature ablations on the HI-Small discovery set (documented window), 3 seeds.
cd /c/dev/aml-stream
export PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
run() { $PY -I scripts/03_sweep.py --max-day 9 --tag "abl_$1" --horizons "$2" --seeds 0,1,2 --neg-rate 0.25 --drop "$3" > "logs/abl_$1.log" 2>&1; }
run nofwd 3600 '^fwd_'
run nomotif 0,1,3600 '^m_'
run nopair 0,1,3600 '_pair_'
run nohour 0,1,3600 '^hour$'
run nofmt 0,1,3600 '^fmt$'
run nocur 0,1,3600 '^cur_|same_cur'
echo QUEUE_C_DONE
