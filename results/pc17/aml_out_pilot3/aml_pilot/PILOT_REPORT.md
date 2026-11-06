# NB1 pilot: FAIL

- [ ] Every pilot job finished and passed verification: pilot-nb1-format-check, pilot-nb1-mgnn-gin, pilot-nb1-mgnn-gin-replay86400, pilot-nb1-trainnorm
- [ ] A killed job resumed and agreed with the uninterrupted run: {"pass": false, "note": "pilot-nb1-mgnn-gin did not finish, so there is nothing to compare with"}
- [x] A GPU was attached: GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-3ca2c5b7-4cea-8a34-1c55-98ec714a6e43)
- [ ] CHECK BY HAND: Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?: session length 1.10 h
- [ ] CHECK BY HAND: The formatter port matches the official formatter: {}
- [ ] CHECK BY HAND: Check after production: nb1-mgnn-gin-HI-Small-s1 reproduces the published F1 of 64.79% within its spread: official_style_test_f1 in that job's metrics.json

## Measured seconds


## Projected production hours (rough: one short run, scaled by row count)

- P0: 0.0 h over 0 jobs, 13 jobs not projected
- P1: 0.0 h over 0 jobs, 11 jobs not projected

## Failed or unfinished jobs

### pilot-nb1-format-check (TODO)

RuntimeError('mgnn_format_check failed with exit 1. Its last output:\nTraceback (most recent call last):\n  File "<string>", line 1, in <module>\n  File "D:\\cnet proj\\aml-stream\\src\\amlh\\isolate.py", line 122, in child_main\n    result = fn(ctx, spec["cfg"]) or {}\n             ^^^^^^^^^^^^^^^^^^^^\n  File "D:\\cnet proj\\aml-stream\\src\\amlh\\mgnn.py", line 485, in mgnn_format_check\n    raise ValueError(f"port differs from official formatter: {json.dumps(diffs)}")\nValueError: port differs from official formatter: {"EdgeID": 0, "from_id": 0, "to_id": 0, "Timestamp": 0, "Amount Sent": 56637, "Sent Currency": 0, "Amount Received": 56628, "Received Currency": 0, "Payment Format": 0, "Is Laundering": 0}\n')

```
Traceback (most recent call last):
  File "D:\cnet proj\aml-stream\src\amlh\runner.py", line 299, in run_job
    metrics = self._resolve(spec.fn)(ctx, spec.cfg) or {}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\cnet proj\aml-stream\src\amlh\mgnn.py", line 493, in job
    return run_isolated(ctx, cfg, "amlh.mgnn", name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\cnet proj\aml-stream\src\amlh\isolate.py", line 97, in run_isolated
    raise RuntimeError(f"{name} failed with {explain_exit(code)}. Its last output:\n{err_path.read_text()[-1500:]}")
RuntimeError: mgnn_format_check failed with exit 1. Its last output:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\cnet proj\aml-stream\src\amlh\isolate.py", line 122, in child_main
    result = fn(ctx, spec["cfg"]) or {}
             ^^^^^^^^^^^^^^^^^^^^
  File "D:\cnet proj\aml-stream\src\amlh\mgnn.py", line 485, in mgnn_format_check
    raise ValueError(f"port differs from official formatter: {json.dumps(diffs)}")
ValueError: port differs from official formatter: {"EdgeID": 0, "from_id": 0, "to_id": 0, "Timestamp": 0, "Amount Sent": 56637, "Sent Currency": 0, "Amount Received": 56628, "Received Currency": 0, "Payment Format": 0, "Is Laundering": 0}


```

### pilot-nb1-mgnn-gin (INTERRUPTED)

mgnn_train stopped for lack of session time

### pilot-nb1-mgnn-gin-replay86400 (None)

no DONE marker

### pilot-nb1-trainnorm (None)

no DONE marker

