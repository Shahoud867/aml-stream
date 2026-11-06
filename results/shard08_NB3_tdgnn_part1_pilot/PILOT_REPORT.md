# NB3 pilot: PASS

- [x] Every pilot job finished and passed verification
- [x] A killed job resumed and agreed with the uninterrupted run: {"pass": true, "killed_exit_code": 137, "resumed_exit_code": 0, "resumed_done": true, "detail": "", "rule": "correlation at least 0.95", "scores": {"scores_test.parquet": {"rows": 863900, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "correlation": 1.0}, "scores_val.parquet": {"rows": 965524, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "correlation": 0.9999999999999999}}}
- [x] A GPU was attached: GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-3e441960-0855-5624-a0c4-85cbee0e38e8)
- [ ] CHECK BY HAND: Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?: session length 0.04 h

## Measured seconds

- pilot-nb3-tdgnn: 50.6
- pilot-nb3-attack: 5.2
- pilot-nb3-service: 1.0
- pilot-nb3-tdgnn-killed: 39.7

## Projected production hours (rough: one short run, scaled by row count)

- P0: 1.0 h over 41 jobs, 0 jobs not projected
- P1: 1.0 h over 49 jobs, 0 jobs not projected
