# NB1: execution summary

- Code version: `9686ce184d525c0a`
- Session time: 0.26 h
- Complete: **False**
- Job counts: {'FAILED': 2, 'DEFERRED': 2}

| Job | Tier | Status | Note |
|---|---|---|---|
| pilot-nb1-format-check | P0 | FAILED | RuntimeError('mgnn_format_check failed with exit 1. Its last output:\nTraceback (most rece |
| pilot-nb1-mgnn-gin | P0 | FAILED | RuntimeError('mgnn_train failed with exit 1. Its last output:\nTraceback (most recent call |
| pilot-nb1-mgnn-gin-replay86400 | P0 | DEFERRED | waiting for ['pilot-nb1-mgnn-gin'] |
| pilot-nb1-trainnorm | P0 | DEFERRED | waiting for ['pilot-nb1-mgnn-gin'] |
