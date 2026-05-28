# Submit Plan

Baseline/current best local MAPE: `51.178136`.

Decision: no new negative-sampling or two-stage candidate beats the current best. Keep the current best artifact from the prior run.

## Kept Candidate

| file path | local MAPE | delta vs 51.178136 | config | reason |
|---|---:|---:|---|---|
| `outputs/submission_larger_sample_tet_event_inv_y_5group_scale.pkl` | 51.178136 | 0.000000 | larger sample + Tet + Event, `OPT_WEIGHT_MODE=inv_y`, `OPT_BLEND_MODE=raw_only`, `OPT_USE_RAW_ONLY=1`, `OPT_RUN_CATBOOST=0`, refined 5-group scale | Kept current best because no new experiment produced MAPE `< 51.178136`. |

## Rejected Experiments

No submission files were created for these because none beat the baseline.

| experiment | local MAPE | delta vs 51.178136 | config | reason |
|---|---:|---:|---|---|
| negative sampling A refined_5group | 52.946339 | +1.768203 | `OPT_NEGATIVE_SAMPLE_RATIO=0.5`, `OPT_ZERO_WEIGHT=0.3`, Tet + Event, inv_y, raw_only | Rejected: worse than current best. |
| negative sampling B refined_5group | 56.072427 | +4.894291 | `OPT_NEGATIVE_SAMPLE_RATIO=1.0`, `OPT_ZERO_WEIGHT=0.3`, Tet + Event, inv_y, raw_only | Rejected: worse than current best. |
| negative sampling C refined_5group | 63.105339 | +11.927203 | `OPT_NEGATIVE_SAMPLE_RATIO=1.0`, `OPT_ZERO_WEIGHT=0.5`, Tet + Event, inv_y, raw_only | Rejected: worse than current best. |
| two-stage LightGBM | 65.973218 | +14.795082 | LightGBM classifier `sale_flag=y>0` + LightGBM regressor on positive rows; final `classifier_prob * regressor_pred` | Rejected: worse than current best; no submission emitted. |

## Validation Reports

- `reports/train_sampling_diagnostic.md`
- `reports/negative_sampling_validation.md`
- `reports/two_stage_validation.md`

## Submission Checks

- `outputs/submission_larger_sample_tet_event_inv_y_5group_scale.pkl`: `check_submission.py` could not run successfully because the artifact is not present in this workspace.
