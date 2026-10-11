# EngineScope research results

Version two evaluates sensor fusion and historical context using a prespecified study on NASA C-MAPSS FD001 and FD003. It completed 70 development model fits across five engine splits and then refitted one frozen, validation-selected model per dataset. The model and calibration choices were fixed before final assessment and official test evaluation. Results are in operating cycles.

## Repeated development experiments

Each development split uses 45 fitting engines and 15 validation engines from a fixed pool of 60. Values below are mean RMSE plus or minus standard deviation over seeds 7, 21, 42, 84 and 123. Splits overlap, so this standard deviation measures variation across splits rather than independent uncertainty or significance.

| Candidate | FD001 validation RMSE | FD003 validation RMSE |
|---|---:|---:|
| Mean predictor | 43.00 ± 3.90 | 39.28 ± 5.39 |
| Random Forest with window statistics | 18.43 ± 3.05 | 12.96 ± 4.08 |
| XGBoost with window statistics | 17.46 ± 5.03 | 11.20 ± 2.87 |
| LSTM | 15.14 ± 3.32 | 14.46 ± 4.32 |
| GRU | 16.99 ± 2.88 | 13.16 ± 5.73 |
| Random Forest with latest readings only | 19.77 ± 1.93 | 17.79 ± 3.55 |
| Random Forest with one selected sensor | 24.97 ± 3.98 | 19.86 ± 2.81 |

The single sensor was selected using training-only absolute correlation with uncapped RUL. Its selection is repeated inside each split. The all-sensor Random Forest has lower mean error than the single-sensor reference on both datasets. Historical window statistics also improve its mean error relative to latest readings only. These are descriptive results for the tested settings; they do not prove a causal fusion mechanism or isolate temporal ordering within a recurrent model.

The first exploratory run ranked XGBoost ahead of LSTM on one FD001 split. The repeated study selects LSTM on FD001 and XGBoost on FD003. This supports comparing model families rather than claiming that deep learning universally wins.

## Frozen official test evaluation

The selected model is refitted using all 60 development engines. Twenty separate calibration-engine endpoints establish nominal 90% absolute-residual prediction intervals. Each official test set contains 100 engines, with one prediction at its final available cycle. Targets and predictions are capped at 125 cycles for the primary evaluation.

| Dataset | Frozen model | Test RMSE | Test MAE | 95% RMSE bootstrap interval | Observed interval coverage | Mean interval width |
|---|---|---:|---:|---|---:|---:|
| FD001 | LSTM, 14 refit epochs | 16.28 | 11.18 | 13.10 to 19.42 | 94% | 53.01 |
| FD003 | XGBoost, 200 trees | 14.94 | 10.79 | 12.20 to 17.54 | 91% | 44.12 |

RMSE intervals use 2,000 bootstrap resamples of engine endpoints. Calibrated intervals concern capped RUL at one endpoint, not the probability of a particular failure time. Exchangeable calibration and future endpoints are an assumption. Their widths are substantial, so coverage alone should not be treated as a precise forecast. Coverage on the smaller assessment endpoint sets was 85% for FD001 and 95% for FD003; nominal 90% does not mean exactly 90% in every finite sample or subgroup.

Uncapped-target metrics, raw endpoint predictions, interval bounds, source hashes, and frozen settings are saved alongside the primary metrics. Both datasets are modeled separately; this is replication under another fault configuration rather than transfer learning.

## Warning assessment

Twenty reserved assessment engines per dataset are replayed through their full lifetimes using only observations available through each cycle. The warning rule is fixed before assessment: predicted RUL at most 30 cycles for three consecutive cycles. The comparison alarm uses sensor 2 and a fixed three-sigma deviation from nominal statistics fitted on development engines.

| Dataset and rule | Missed engines | False-positive cycle rate outside final 30 cycles | Mean confirmed lead time among detected engines |
|---|---:|---:|---:|
| FD001 LSTM warning | 0 of 20 | 4.05% | 25.85 |
| FD001 sensor-2 threshold | 9 of 20 | 1.15% | 6.18 |
| FD003 XGBoost warning | 0 of 20 | 1.51% | 22.90 |
| FD003 sensor-2 threshold | 9 of 20 | 2.02% | 15.36 |

Lead time counts the first persistent warning inside the final 30-cycle horizon. Premature alerts count toward the false-positive cycle rate. The threshold's average lead time uses its 11 detected engines, while each learned model's average uses all 20. The FD001 result therefore involves a clear tradeoff: fewer misses and earlier warnings accompany more false-positive cycles. FD003 improves the reported measures relative to this particular threshold on this assessment sample. Neither establishes dominance over a tuned industrial alarm system.

FD001 assessment is retrospective because the dataset was used in version-one exploration. It is excluded from version-two fitting and calibration, but it is not a globally pristine dataset. FD003 was first modeled under the prespecified version-two protocol.

## Failure patterns and interpretation

| Dataset | Late-life RMSE, RUL 0 to 30 | Middle-life RMSE, RUL 31 to 80 | Early-life RMSE, RUL above 80 |
|---|---:|---:|---:|
| FD001 | 4.16, 25 engines | 22.57, 20 engines | 16.99, 55 engines |
| FD003 | 6.63, 20 engines | 18.59, 33 engines | 14.62, 47 engines |

Both final models show their largest stage-level error in the middle-life group. Mean signed error there is +10.16 cycles on FD001 and +6.07 on FD003, indicating life overestimation. That pattern warrants further investigation; it does not establish a physical fault explanation. Middle-life interval coverage is also lower than overall coverage: 85% and 87.88%, respectively. Global coverage is not a guarantee for every life stage.

The dashboard exposes the ten worst engine errors, life-stage tables, and assessment trajectories so these limits are inspectable. Version-one SHAP and sensor-removal results remain available as exploratory explanations. They describe model dependence under correlated sensors and do not establish causation.

## Reproduction and reporting

Run `python -m rul study --output artifacts/research_v2 --epochs 16 --trees 100` from the locked project environment. The saved protocol contains exact engine assignments, candidate families, settings, hashes, and sampling seeds. Completed results are preserved. Since official tests have now been evaluated for this release, future tuning must not describe them as untouched validation data.

An accurate resume statement is:

> Built a reproducible predictive-maintenance study comparing seven configurations across five engine splits on NASA C-MAPSS FD001 and FD003; implemented engine-disjoint evaluation, sensor/history ablations, calibrated prediction intervals, and an interactive inference dashboard. Validation-selected models achieved official-test RMSE of 16.28 and 14.94 cycles under a capped-RUL protocol.

Local verification passed 21 tests, including data isolation, causal windows, exact calibration quantiles, label-independent inference, model gradients, and dashboard dataset/replay controls. A GitHub Actions workflow is configured; it has not been run remotely because the project has not been published.
