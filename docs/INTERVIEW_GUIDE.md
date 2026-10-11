# Explaining EngineScope in an interview

Describe the project as a controlled study of sensor fusion and temporal context for remaining useful life estimation. Lead with the question, experimental controls, and measured results. Do not claim that LSTM always beats classical models or that the dataset represents real flight telemetry.

## What to demonstrate

1. Explain one sensor window and its RUL target, then locate those operations in `rul/data.py`.
2. Explain why engines, rather than overlapping windows, are split between fitting and evaluation.
3. Show the five-seed development table and distinguish variation across overlapping splits from independent statistical evidence.
4. Show the frozen model selection and explain why official test results did not select the model.
5. Show the latest-reading and single-sensor ablations. Explain what they control and which conclusions they cannot establish.
6. Replay an assessment engine. Explain why an always-on alarm should not be rewarded as a successful early warning.
7. Show a large prediction error and the life-stage table. Distinguish an observed failure pattern from a verified physical explanation.
8. Upload a sensor-history file and explain the interval target, calibration assumptions, and padding behavior.

## Questions to prepare for

**Why cap RUL?** The cap encodes an early-life plateau assumption and prevents very long healthy histories from dominating the target range. It is a chosen modeling assumption, not measured degradation onset. Capped and uncapped evaluation conventions must be stated explicitly.

**Why not randomly split windows?** Adjacent windows share most readings. Random splitting can place near-duplicates and the same engine's characteristic patterns into both fitting and evaluation, making performance look better than its ability to predict another engine.

**What is sensor fusion here?** The model receives several channels jointly. The single-sensor reference and sensor-removal experiments investigate the benefit of their joint information. They do not prove every channel is weak, establish causal fault mechanisms, or demonstrate a novel fusion architecture.

**Why compare window statistics and recurrent networks?** Tree models learn from summaries of the same observation period; recurrent models learn directly from the ordered sequence. The latest-reading Random Forest controls whether historical context helps a tree model. This does not isolate the effect of temporal order within an LSTM; a sequence-order ablation would be a further experiment.

**What is the uncertainty interval?** It uses the finite-sample order statistic of absolute errors on separate calibration-engine endpoints. The nominal target is 90% coverage for capped RUL under exchangeable endpoints. Report observed official test coverage and width. Do not present it as a 90% probability that a particular engine will fail in that interval or as guaranteed coverage across its whole lifetime.

**Why do models change ranking?** Different fitting and validation engines can have different degradation patterns, and model capacity and random optimization matter. The study measures ranking stability. It does not establish a physical explanation for every ranking change.

**What makes FD003 useful?** It provides a second fault configuration with a single operating condition. We fit it separately under the same study design. That is replication; it is not a claim that a model trained on FD001 transfers to FD003.

**What remains limited?** Simulated data, small development/calibration/assessment cohorts, fixed hyperparameters, overlapping development splits, and retrospective exposure to FD001. The official test set has now been evaluated for this release. Future changes must treat it as a used benchmark and should reserve fresh data for new claims.

## Before putting the project on your resume

Reproduce one experiment yourself, trace an input through preprocessing and prediction, and explain why each evaluation group exists. Cite exact saved metrics and label them as validation or official test results. Describe AI assistance honestly if asked; your demonstrated understanding and ownership matter more than the amount of generated code.
