# Explaining DegradSense in college

Start with the simpler models, then introduce the advanced comparisons.

**What does the project do?** It estimates remaining engine operating cycles from sensor histories and evaluates whether a persistent low-RUL prediction provides useful early warning. NASA C-MAPSS is simulated turbofan data.

**What is Linear Regression here?** We summarize each sensor's last 30 readings using six statistics, standardize those features using fitting engines, and fit an ordinary least-squares equation: predicted RUL = intercept + sum(coefficient × standardized feature). Predictions are clipped to the target range [0,125]. The coefficient plot uses standardized features; correlated features complicate interpretation.

**What is the Decision Tree?** A single tree repeatedly splits feature values to reduce squared prediction error. A leaf predicts the average training target of samples reaching that leaf. Maximum depth 14 and minimum leaf size 3 control complexity. This is one tree, not Random Forest.

**Why compare with advanced models?** XGBoost combines boosted trees; LSTM and GRU learn from ordered sensor sequences. They remain comparison models, while Linear Regression and Decision Tree provide the curriculum-friendly starting point. Do not claim that these advanced methods were taught in class if they were self-study.

**Why does mean still appear in feature descriptions?** A window's sensor mean is an input feature. The removed mean model was a constant baseline predicting the same average RUL for every engine. They are different uses of an average.

**Why split by engine?** Randomly mixing overlapping windows from one engine into train and validation can leak near-duplicate histories. Engine-disjoint splits measure performance on other engines.

**How are models compared fairly?** Use the same engine groups, truncated endpoints, target convention, and preprocessing. Development results select models; calibration and assessment have separate roles. Do not compare single-split scores directly with five-split averages.

**Is this a new untouched test?** No. Both official test outcomes were previously inspected. The current curriculum revision is retrospective and preserves that limitation.

**What should I demonstrate?** In Single-split experiment select Linear Regression (`linear`) and Decision Tree (`dt`), show their metrics and replay, and explain coefficients/tree attribution. Then show the five-seed Research evidence comparison and uncertainty/failure analysis.

**What should I avoid claiming?** Novel architecture, causal sensor explanations, real-engine validation, guaranteed failure time, or that the simpler models outperform every advanced model. Explain assistance honestly and demonstrate that you understand the code.
