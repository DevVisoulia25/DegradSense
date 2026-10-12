# DegradSense model card

## Purpose and intended use

DegradSense is an educational research implementation of remaining useful life estimation from simulated turbofan sensor histories. Use it to reproduce controlled predictive-maintenance experiments and inspect model behavior. Its final model for FD001 is an LSTM; its final model for FD003 is XGBoost. The datasets are modeled independently.

The trained models support the selected dataset's sensor definitions, ordering, and units. They are not validated for real equipment or new operating regimes. FD002 and FD004 require further operating-condition handling and are not supported by this research inference interface.

## Input and output

Input consists of engine ID, consecutive operating cycles starting at 1, three operating settings, and 21 sensor measurements in the canonical C-MAPSS schema. The model features use informative sensor channels from the most recent 30 cycles. IDs, cycle indices, labels, and operating settings are not model predictors in this single-regime implementation. Histories shorter than 30 cycles repeat their first available observation for left padding.

Output is estimated capped RUL in cycles, between 0 and 125, and an endpoint interval calibrated to a nominal 90% coverage target. An estimate at the cap does not distinguish an engine with 125 cycles remaining from one with a much longer life. The interval is not a failure probability and does not establish simultaneous coverage over every cycle of an engine's history.

## Data and evaluation

NASA C-MAPSS is simulated run-to-failure data. Each dataset has 60 development engines, 20 calibration engines, and 20 assessment engines. Candidate comparison uses five splits within development; final models fit the full development pool. Official evaluation uses 100 separate test engines per dataset after model choices and calibration are frozen.

FD001 had prior exploratory exposure in version one. FD003 was first modeled under the version-two protocol. Exact assignments and source hashes are in `artifacts/research_v2/protocol.json`. Dataset attribution: A. Saxena and K. Goebel (2008), Turbofan Engine Degradation Simulation Data Set, NASA Ames Prognostics Data Repository.

See [research results](RESULTS.md) for test errors, bootstrap intervals, calibration width and coverage, warning outcomes, and life-stage failure analysis. In particular, FD001 warning sensitivity comes with a higher false-positive cycle rate than the simple threshold baseline, and middle-life errors are larger on both datasets.

## Limitations and maintenance

This release uses small calibration and assessment groups, fixed model settings, overlapping development splits, and synthetic equipment trajectories. No experiment establishes a novel fusion architecture, causal sensor-fault relationships, or operational maintenance savings. Calibration assumptions may fail when endpoint distributions, sensor noise, operating regimes, or equipment change.

Model artifacts are local files produced by this project. Inference loads those frozen artifacts; the dashboard does not accept uploaded model files. The upload flow validates the sensor schema and returns predictions without using supplied labels.

The official test sets have been evaluated for this release. Any subsequent tuning should treat these benchmarks as used data and reserve fresh evidence for new generalization claims. Record dataset convention, model selection, calibration, and actual evaluation coverage when reporting a new version.
