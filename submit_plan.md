# Submit Plan

Submit exactly these 5 files today. Do not submit extra random scale variants.

- Submit 1: outputs/submission_final.pkl
  Reason: portal-format pickle matching the accepted reference schema `location,item_id,quantity`.
- Submit 2: outputs/submission_control.csv
  Reason: control/debug CSV with `location,item_id,prediction`; convert to portal pickle before upload if needed.
- Submit 3: outputs/submission_raw_only.csv
  Reason: model hypothesis; raw-only path selected from `pred_raw` validation comparison.
- Submit 4: outputs/submission_scale_best_minus.csv
  Reason: one local step below best scale = 0.975; convert to portal pickle before upload if needed.
- Submit 5: outputs/submission_scale_best_plus.csv
  Reason: one local step above best scale = 1.025; convert to portal pickle before upload if needed.

All five files are generated from local December validation, not leaderboard probing.
