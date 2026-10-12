# DegradSense curriculum revision results

Linear Regression and Decision Tree replace the earlier Random Forest and constant mean baseline. Five model families and two Decision Tree ablations are evaluated on five paired development splits for each dataset. The single-split demonstration is separate from this repeated comparison.

## Repeated development results

| Dataset | Model | Mean RMSE | Split standard deviation |
|---|---|---:|---:|
| FD001 | Decision Tree | 25.07 | 6.23 |
| FD001 | Decision Tree latest readings | 26.62 | 9.71 |
| FD001 | Decision Tree single sensor | 31.01 | 5.11 |
| FD001 | GRU | 16.99 | 2.88 |
| FD001 | Linear Regression | 21.68 | 1.08 |
| FD001 | LSTM | 15.14 | 3.32 |
| FD001 | XGBoost | 17.46 | 5.03 |
| FD003 | Decision Tree | 17.45 | 10.59 |
| FD003 | Decision Tree latest readings | 21.80 | 7.21 |
| FD003 | Decision Tree single sensor | 27.49 | 3.73 |
| FD003 | GRU | 13.16 | 5.73 |
| FD003 | Linear Regression | 15.02 | 2.31 |
| FD003 | LSTM | 14.46 | 4.32 |
| FD003 | XGBoost | 11.20 | 2.87 |

Units are operating cycles, with targets/predictions capped at 125. Overlapping development splits make the standard deviation descriptive rather than an independent confidence interval or significance test.

## Release evaluation

| Dataset | Selected model | Official test RMSE | MAE | Endpoint interval coverage | Mean interval width |
|---|---|---:|---:|---:|---:|
| FD001 | LSTM | 16.28 | 11.18 | 94% | 53.01 |
| FD003 | XGBoost | 14.94 | 10.79 | 91% | 44.12 |

The selected model families remain LSTM on FD001 and XGBoost on FD003. The added simple models did not outperform those selections on mean development RMSE.

## College demonstration: one FD001 split

| Model | Validation endpoint RMSE | MAE |
|---|---:|---:|
| Linear Regression | 13.63 | 11.26 |
| Decision Tree | 17.89 | 13.55 |
| XGBoost | 14.33 | 10.16 |
| LSTM | 11.47 | 8.03 |
| GRU | 10.47 | 7.58 |

This run uses 80 fitting and 20 validation engines, seed 42. Its randomly truncated endpoint mechanism is the current engine-specific sampler; scores are not directly comparable with the original version-one sampler or five-split averages. No official test evaluation was run for this single-split demonstration.

## Provenance and limitations

Both datasets and official outcomes were inspected before this revision. These release results are retrospective, not a fresh untouched benchmark. Development fits for unchanged XGBoost/LSTM/GRU candidates were reused only after checking matching settings, engine cohorts, training-data hashes, endpoint sampling and preprocessing; newly added linear/tree configurations were fitted. Reuse is recorded in development metric rows. There are 70 candidate/split evaluations, not a claim of 70 newly executed fits.

The research study retains 60 development / 20 calibration / 20 assessment engines, model selection using development endpoints only, a fixed warning rule, nominal 90% endpoint conformal intervals, and 2,000-resample engine-bootstrap RMSE intervals. Intervals can be broad and do not guarantee whole-trajectory coverage or express failure probabilities.

The prepared inference bundle uses this revision. Historical version-two documentation and evidence remain archived for traceability. No hosted app has been launched. Simulation performance does not establish real-equipment reliability or a novel research method.
