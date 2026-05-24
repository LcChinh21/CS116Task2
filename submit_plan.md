# Submit Plan

Submit exactly these 5 files today. Do not submit extra random scale variants.

- Submit 1: outputs/submission_control.csv
  Reason: control artifact with official columns `location,item_id,prediction`.
- Submit 2: outputs/submission_raw_only.csv
  Reason: model hypothesis; raw-only path selected from `pred_raw` validation comparison.
- Submit 3: outputs/submission_scale_best.csv
  Reason: best local validation scale = 1.000. Validation MAPE local=51.548656.
- Submit 4: outputs/submission_scale_best_minus.csv
  Reason: one local step below best scale = 0.975.
- Submit 5: outputs/submission_scale_best_plus.csv
  Reason: one local step above best scale = 1.025.

All five files are generated from local December validation, not leaderboard probing.
