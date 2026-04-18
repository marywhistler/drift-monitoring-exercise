## Bike Sharing Drift Monitoring with Evidently

### Commands for executing:

`python3.11 -m venv .venvsource .venv/bin/activate`

`python3.11 --versionpip install -r requirements.txt`

`python run_bike_sharing_monitoring.py`\
`evidently ui --workspace ./datascientest-workspace`

------------------------------------------------------------------------

### Exam Questions:

#### 1. What changed over weeks 1, 2 and 3?

The model performance worsened over time: - week 1 had worse results than the January baseline (RMSE = 18.02; MAE = 10.89; R² = .878) - week 2 was again slightly worse (RMSE = 21.83; MAE = 14.64; R² = .865) - week 3 was the worst performance-wise (RMSE = 37.63; MAE = 24.67; R² = .729)

I.e., the drift became stronger over time!

------------------------------------------------------------------------

#### 2. What seems to be the root cause of the drift?

The change in the target behavior.

Mean Bike count: 60.34 (week 1) -\> 70.08 (week 2) -\> 84.16 (week 3)\
Mean Prediction: 55.10 (week 1) -\> 60.87 (week 2) -\> 61.51 (week 3), i.e. flatter increase\
Mean error became more negative over time...

------------------------------------------------------------------------

#### 3. Which strategy is to be applied?

Probably, to retrain the model with more recent data or update it more regularly, since in February the model from January already was outdated.
