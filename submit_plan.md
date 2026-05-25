# Submit Plan

Only these 5 candidates are tracked. No random scale variants.

1. `outputs/submission_best_39_0.pkl`
   - role: control_best_39_0
   - public score: 39.0
   - reason: locked control from current best; do not overwrite

2. `outputs/submission_larger_sample_inv_y_raw.pkl`
   - role: larger_sample_inv_y_raw
   - validation MAPE: pending
   - reason: candidate not generated yet
   - metadata: `outputs/candidate_validation_larger_sample.json`

3. `outputs/submission_larger_sample_inv_y_5group_scale.pkl`
   - role: larger_sample_inv_y_raw + 5_group_scale
   - validation MAPE: pending
   - reason: candidate not generated yet
   - metadata: `outputs/candidate_validation_larger_sample.json`

4. `outputs/submission_larger_sample_tet_inv_y_raw.pkl`
   - role: larger_sample_inv_y_raw + Tet
   - validation MAPE: pending
   - reason: candidate not generated yet
   - metadata: `outputs/candidate_validation_larger_sample_tet.json`

5. `outputs/submission_larger_sample_tet_event_inv_y_raw.pkl`
   - role: larger_sample_inv_y_raw + Tet + event_features/group_scale
   - validation MAPE: pending
   - reason: candidate not generated yet
   - metadata: `outputs/candidate_validation_larger_sample_tet_event.json`
