# Merge report

## Plan status (finished / planned)

| Notebook | P0 | P1 | P2 |
|---|---|---|---|
| NB1 | 11/13 | 2/11 | 0/0 |
| NB2 | 0/12 | 0/3 | 0/1 |
| NB3 | 41/41 | 49/49 | 0/0 |
| NB4 | 0/118 | 0/144 | 0/35 |
| NB5 | 0/108 | 0/68 | 0/0 |

## Verification

- [ ] All notebooks ran the same code: 68ab7cece98e33ad, 8ac92594c2be0f2f, 9686ce184d525c0a, d5ac9b86ae455ee3, d7aa5068018d7d77
- [ ] Every finished job has that code: 68ab7cece98e33ad, 9686ce184d525c0a, d5ac9b86ae455ee3, d7aa5068018d7d77
- [ ] No finished job fails its hash check: nb1-mgnn-gin-HI-Small-cad3600-s1: missing isolated_result.json; nb1-mgnn-gin-HI-Small-s1-replay3600: hash mismatch state.json; nb1-mgnn-gin-HI-Small-s3-replay21600: hash mismatch state.json; nb1-format-check: missing isolated_result.json; nb1-mgnn-gin-HI-Small-s2: missing ckpt/best.bin
- [x] Prepared data agrees across notebooks

## Still sealed

- LI-Small: 0 of 54 sweep jobs finished

## HI-Small (unsealing: not sealed)

**Leakage check: passed.**

### Published models (Gate G1)

| Job | H | Official-style F1 | Published | F1 at val threshold | Core F1 | Tail F1 | Core PR-AUC | Stoppable value at 500/day | Sampler coverage |
|---|---|---|---|---|---|---|---|---|---|
| nb1-mgnn-gin-HI-Small-cad3600-s1 | inf | 0.4587 | 0.6479 | 0.4590 | 0.3609 | 0.6067 | 0.2741 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-cad3600-s1 [eval mode] | inf | 0.5096 |  | 0.5137 | 0.3981 | 0.6945 | 0.3253 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s1 | inf | 0.5081 | 0.6479 | 0.4938 | 0.3299 | 0.8567 | 0.3917 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s1 [eval mode] | inf | 0.5373 |  | 0.5022 | 0.3387 | 0.8617 | 0.4171 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s1-replay21600 | 21600 | 0.5325 | 0.6479 | 0.5219 | 0.3375 | 0.8748 | 0.3337 | 0.5355 | 0.9991 |
| nb1-mgnn-gin-HI-Small-s1-replay86400 | 86400 | 0.5177 | 0.6479 | 0.5095 | 0.3318 | 0.8661 | 0.3443 | 0.4258 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s2 | inf | 0.6417 | 0.6479 | 0.6506 | 0.5111 | 0.8344 | 0.4587 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s2 [eval mode] | inf | 0.6664 |  | 0.6416 | 0.4878 | 0.8731 | 0.4904 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s2-replay21600 | 21600 | 0.6208 | 0.6479 | 0.6192 | 0.4686 | 0.8159 | 0.4171 | 0.5458 | 0.9991 |
| nb1-mgnn-gin-HI-Small-s2-replay3600 | 3600 | 0.6125 | 0.6479 | 0.6131 | 0.4557 | 0.8295 | 0.4109 | 0.5680 | 0.9991 |
| nb1-mgnn-gin-HI-Small-s2-replay86400 | 86400 | 0.6231 | 0.6479 | 0.6211 | 0.4677 | 0.8308 | 0.4197 | 0.4168 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s2-trainnorm | inf | 0.5442 | 0.6479 | 0.5836 | 0.4159 | 0.8615 | 0.4069 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s3 | inf | 0.5544 | 0.6479 | 0.5389 | 0.3672 | 0.8736 | 0.4030 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s3 [eval mode] | inf | 0.5684 |  | 0.5290 | 0.3577 | 0.8818 | 0.4247 | 0.0000 | 0.9990 |
| nb1-mgnn-gin-HI-Small-s3-replay3600 | 3600 | 0.5156 | 0.6479 | 0.5282 | 0.3599 | 0.8484 | 0.3432 | 0.5648 | 0.9991 |
| nb1-mgnn-gin-HI-Small-s3-replay86400 | 86400 | 0.5348 | 0.6479 | 0.5558 | 0.3949 | 0.8351 | 0.3695 | 0.4204 | 0.9990 |

| Hypothesis | Value | CI | Verdict |
|---|---|---|---|
