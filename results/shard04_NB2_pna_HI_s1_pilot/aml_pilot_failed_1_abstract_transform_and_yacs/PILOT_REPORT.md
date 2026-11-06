# NB2 pilot: FAIL

- [ ] Every pilot job finished and passed verification: pilot-nb2-mgnn-pna, pilot-nb2-mgnn-pna-replay86400, pilot-nb2-fraudgt
- [ ] A killed job resumed and agreed with the uninterrupted run: {"pass": false, "note": "pilot-nb2-mgnn-pna did not finish, so there is nothing to compare with"}
- [x] A GPU was attached: GPU 0: NVIDIA GeForce RTX 3080 (UUID: GPU-3e441960-0855-5624-a0c4-85cbee0e38e8)
- [ ] CHECK BY HAND: Check by hand: compare the quota meter with this session's length. Does T4 x2 count double?: session length 0.31 h

## Measured seconds


## Projected production hours (rough: one short run, scaled by row count)

- P0: 0.0 h over 0 jobs, 12 jobs not projected
- P1: 0.0 h over 0 jobs, 3 jobs not projected
- P2: 0.0 h over 0 jobs, 1 jobs not projected

## Failed or unfinished jobs

### pilot-nb2-mgnn-pna (FAILED)

RuntimeError('mgnn_train failed with exit 1. Its last output:\nTraceback (most recent call last):\n  File "<string>", line 1, in <module>\n  File "E:\\aml\\aml-stream\\src\\amlh\\isolate.py", line 122, in child_main\n    result = fn(ctx, spec["cfg"]) or {}\n             ^^^^^^^^^^^^^^^^^^^^\n  File "E:\\aml\\aml-stream\\src\\amlh\\mgnn.py", line 258, in mgnn_train\n    model, optimizer, transform, tr_loader, config = _model(cfg, blob, args, device)\n                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File "E:\\aml\\aml-stream\\src\\amlh\\mgnn.py", line 224, in _model\n    transform = train_util.AddEgoIds()\n                ^^^^^^^^^^^^^^^^^^^^^^\nTypeError: Can\'t instantiate abstract class AddEgoIds with abstract method forward\n')

```
Traceback (most recent call last):
  File "E:\aml\aml-stream\src\amlh\runner.py", line 299, in run_job
    metrics = self._resolve(spec.fn)(ctx, spec.cfg) or {}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml-stream\src\amlh\mgnn.py", line 484, in job
    return run_isolated(ctx, cfg, "amlh.mgnn", name)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml-stream\src\amlh\isolate.py", line 97, in run_isolated
    raise RuntimeError(f"{name} failed with {explain_exit(code)}. Its last output:\n{err_path.read_text()[-1500:]}")
RuntimeError: mgnn_train failed with exit 1. Its last output:
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "E:\aml\aml-stream\src\amlh\isolate.py", line 122, in child_main
    result = fn(ctx, spec["cfg"]) or {}
             ^^^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml-stream\src\amlh\mgnn.py", line 258, in mgnn_train
    model, optimizer, transform, tr_loader, config = _model(cfg, blob, args, device)
                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml-stream\src\amlh\mgnn.py", line 224, in _model
    transform = train_util.AddEgoIds()
                ^^^^^^^^^^^^^^^^^^^^^^
TypeError: Can't instantiate abstract class AddEgoIds with abstract method forward


```

### pilot-nb2-mgnn-pna-replay86400 (DEFERRED)

waiting for ['pilot-nb2-mgnn-pna']

### pilot-nb2-fraudgt (FAILED)

RuntimeError('FraudGT failed with exit 1. Its last output:\nTraceback (most recent call last):\n  File "<frozen runpy>", line 198, in _run_module_as_main\n  File "<frozen runpy>", line 88, in _run_code\n  File "E:\\aml\\aml-stream\\src\\amlh\\fraudgt_driver.py", line 261, in <module>\n    main(sys.argv[1])\n  File "E:\\aml\\aml-stream\\src\\amlh\\fraudgt_driver.py", line 145, in main\n    import fraudGT  # noqa: F401 (registers its modules)\n    ^^^^^^^^^^^^^^\n  File "E:\\aml\\aml_official\\FraudGT\\fraudGT\\__init__.py", line 1, in <module>\n    from .act import * # noqa\n    ^^^^^^^^^^^^^^^^^^\n  File "E:\\aml\\aml_official\\FraudGT\\fraudGT\\act\\example.py", line 5, in <module>\n    from fraudGT.graphgym.config import cfg\n  File "E:\\aml\\aml_official\\FraudGT\\fraudGT\\graphgym\\config.py", line 10, in <module>\n    from yacs.config import CfgNode as CN\nModuleNotFoundError: No module named \'yacs\'\n')

```
Traceback (most recent call last):
  File "E:\aml\aml-stream\src\amlh\runner.py", line 299, in run_job
    metrics = self._resolve(spec.fn)(ctx, spec.cfg) or {}
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml-stream\src\amlh\fraudgt.py", line 87, in fraudgt_train
    raise RuntimeError(f"FraudGT failed with {explain_exit(proc.returncode)}. Its last output:\n{stderr_path.read_text()[-1500:]}")
RuntimeError: FraudGT failed with exit 1. Its last output:
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "E:\aml\aml-stream\src\amlh\fraudgt_driver.py", line 261, in <module>
    main(sys.argv[1])
  File "E:\aml\aml-stream\src\amlh\fraudgt_driver.py", line 145, in main
    import fraudGT  # noqa: F401 (registers its modules)
    ^^^^^^^^^^^^^^
  File "E:\aml\aml_official\FraudGT\fraudGT\__init__.py", line 1, in <module>
    from .act import * # noqa
    ^^^^^^^^^^^^^^^^^^
  File "E:\aml\aml_official\FraudGT\fraudGT\act\example.py", line 5, in <module>
    from fraudGT.graphgym.config import cfg
  File "E:\aml\aml_official\FraudGT\fraudGT\graphgym\config.py", line 10, in <module>
    from yacs.config import CfgNode as CN
ModuleNotFoundError: No module named 'yacs'


```

