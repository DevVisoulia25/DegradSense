# DegradSense model card

The current model comparison includes Linear Regression, Decision Tree, XGBoost, LSTM, and GRU; Random Forest and the mean baseline are removed from active experiments. Decision Tree latest-reading and single-sensor variants study historical context and fusion.

The system estimates capped remaining useful life (maximum 125 cycles) from 30-cycle NASA C-MAPSS sensor histories. FD001 and FD003 are modeled independently. FD002/FD004 and real equipment with different sensor meanings or operating conditions are outside the validated scope.

Linear Regression uses standardized window statistics and ordinary least squares. Decision Tree uses maximum depth 14 and minimum leaf size 3. All fitting-time normalization/sensor selection uses fitting engines only. Engine identifiers, cycle indices, and labels are excluded from predictors.

The revised study preserves 60 development / 20 calibration / 20 assessment cohorts and five development splits. Identical unchanged development candidates can reuse checked version-two fits; new linear and tree candidates are fitted explicitly. Both datasets had earlier exposure, so revision-three assessment and official-test results are retrospective.

Calibration uses 20 engine endpoints and the exact finite-sample split-conformal residual quantile for nominal 90% endpoint intervals. Exchangeability is an assumption; empirical coverage and interval width must be reported. Endpoint calibration does not guarantee simultaneous coverage over an entire replay.

Warnings use predicted RUL at most 30 for three consecutive cycles; missed engines, premature-alert cycle fraction, and confirmed lead time are reported together. An illustrative fixed sensor-2 three-sigma alarm is retained as a rule comparison, not a fitted mean baseline.

Short histories repeat their first observation to form a 30-cycle window. Uploaded labels are discarded from model inputs. The app processes uploaded sensor histories in memory and loads only repository-controlled model checkpoints.

See [results](RESULTS.md) for measured performance. No hosted deployment is active. This educational simulation project is not a certified maintenance decision system.
