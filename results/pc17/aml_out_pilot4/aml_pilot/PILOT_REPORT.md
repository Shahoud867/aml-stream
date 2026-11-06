# NB1 pilot: FAIL

- [x] Every pilot job finished and passed verification
- [ ] A killed job resumed and agreed with the uninterrupted run: {"pass": false, "killed_exit_code": 137, "resumed_exit_code": 0, "resumed_done": true, "detail": "", "rule": "correlation at least 0.95", "scores": {"scores_test_eval.parquet": {"rows": 863900, "max_abs_diff": 0.9411141374730505, "mean_abs_diff": 0.0040788276212060815, "correlation": 0.8975452626102749}, "scores_val_eval.parquet": {"rows": 965524, "max_abs_diff": 0.9241355173289776, "mean_abs_diff": 0.0022111956628690487, "correlation": 0.8786393130392652}}}
- [x] A GPU was attached: GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-3ca2c5b7-4cea-8a34-1c55-98ec714a6e43)
- [ ] CHECK BY HAND: Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?: session length 5.61 h
- [x] The formatter port matches the official formatter: {"EdgeID": 0, "from_id": 0, "to_id": 0, "Timestamp": 0, "Amount Sent": 0, "Sent Currency": 0, "Amount Received": 0, "Received Currency": 0, "Payment Format": 0, "Is Laundering": 0}
- [ ] CHECK BY HAND: Check after production: nb1-mgnn-gin-HI-Small-s1 reproduces the published F1 of 64.79% within its spread: official_style_test_f1 in that job's metrics.json

## Measured seconds

- pilot-nb1-mgnn-gin: 10797.8
- pilot-nb1-trainnorm: 230.1
- pilot-nb1-format-check: 10.0
- pilot-nb1-mgnn-gin-replay86400: 10.1
- pilot-nb1-mgnn-gin-killed: 4546.2

## Projected production hours (rough: one short run, scaled by row count)

- P0: 338.0 h over 13 jobs, 0 jobs not projected
- P1: 450.8 h over 11 jobs, 0 jobs not projected
