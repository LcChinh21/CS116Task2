# Submit Plan

Only these 5 candidates are tracked. No random scale variants.

1. `outputs/submission_best_39_0.pkl`
   - role: control_best_39_0
   - public score: 39.0
   - reason: locked control from current best; do not overwrite

2. `outputs/submission_larger_sample_inv_y_raw.pkl`
   - role: larger_sample_inv_y_raw
   - validation MAPE: 51.245650
   - scale/postprocess: global scale 1.000
   - reason: new model/feature candidate, not random scale probing
   - metadata: `outputs/candidate_validation_larger_sample.json`

3. `outputs/submission_larger_sample_inv_y_5group_scale.pkl`
   - role: larger_sample_inv_y_raw + 5_group_scale
   - validation MAPE: 51.214840
   - selected group scale: True
   - scales: `{"high": 0.95, "low": 1.0, "mid": 1.0, "very_high": 1.15, "very_low": 1.0}`
   - reason: 5-group scale kept only if validation beats raw
   - metadata: `outputs/candidate_validation_larger_sample.json`

4. `outputs/submission_larger_sample_tet_inv_y_raw.pkl`
   - role: larger_sample_inv_y_raw + Tet
   - validation MAPE: 51.232501
   - scale/postprocess: global scale 1.000
   - reason: new model/feature candidate, not random scale probing
   - metadata: `outputs/candidate_validation_larger_sample_tet.json`

5. `outputs/submission_larger_sample_tet_event_inv_y_5group_scale.pkl`
   - role: larger_sample_inv_y_raw + Tet + event_features/group_scale
   - validation MAPE: 51.178136
   - selected group scale: True
   - scales: `{"high": 0.95, "low": 1.0, "mid": 1.0, "very_high": 1.125, "very_low": 1.0}`
   - reason: 5-group scale kept only if validation beats raw
   - metadata: `outputs/candidate_validation_larger_sample_tet_event.json`
