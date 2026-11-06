# NB2 pilot: FAIL

- [x] Every pilot job finished and passed verification
- [ ] A killed job resumed and agreed with the uninterrupted run: {"pass": false, "killed_exit_code": 0, "resumed_exit_code": 0, "resumed_done": true, "detail": "", "rule": "correlation at least 0.95", "scores": {"scores_test_eval.parquet": {"rows": 863900, "max_abs_diff": 0.8486086949706078, "mean_abs_diff": 0.0013542705032975886, "correlation": 0.9510290141336064}, "scores_val_eval.parquet": {"rows": 965524, "max_abs_diff": 0.7813059985637665, "mean_abs_diff": 0.000625502927124582, "correlation": 0.9622984042267363}}}
- [x] A GPU was attached: GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-3ca2c5b7-4cea-8a34-1c55-98ec714a6e43)
- [ ] CHECK BY HAND: Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?: session length 0.04 h

## Measured seconds

- pilot-nb2-mgnn-pna: 3721.0
- pilot-nb2-mgnn-pna-replay86400: 10.1
- pilot-nb2-mgnn-pna-killed: 2610.8
- pilot-nb2-fraudgt: 120.1

## Projected production hours (rough: one short run, scaled by row count)

- P0: 151.9 h over 12 jobs, 0 jobs not projected
- P1: 107.6 h over 3 jobs, 0 jobs not projected
- P2: 201.8 h over 1 jobs, 0 jobs not projected
