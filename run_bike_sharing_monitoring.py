import os

import evidently
import datetime
import pandas as pd
import numpy as np
import requests
import zipfile
import io
import json

from sklearn import datasets, ensemble, model_selection
from scipy.stats import anderson_ksamp

from evidently.metrics import (
    RegressionQualityMetric,
    RegressionErrorPlot,
    RegressionErrorDistribution,
)
from evidently.metric_preset import DataDriftPreset, RegressionPreset, TargetDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.ui.workspace import Workspace

# ignore warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

WORKSPACE_NAME = "datascientest-workspace"
PROJECT_NAME = "bike_sharing_monitoring"
PROJECT_DESCRIPTION = "Bike Sharing monitoring exam"

WEEK_PERIODS = {
    "week_1": ("2011-01-29 00:00:00", "2011-02-07 23:00:00"),
    "week_2": ("2011-02-07 00:00:00", "2011-02-14 23:00:00"),
    "week_3": ("2011-02-15 00:00:00", "2011-02-21 23:00:00"),
}


def _fetch_data() -> pd.DataFrame:
    content = requests.get(
        "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
        verify=False,
        timeout=60,
    ).content

    with zipfile.ZipFile(io.BytesIO(content)) as arc:
        raw_data = pd.read_csv(
            arc.open("hour.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
        )
    return raw_data


def _process_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    raw_data = raw_data.copy()
    raw_data.index = raw_data.apply(
        lambda row: datetime.datetime.combine(
            row.dteday.date(),
            datetime.time(int(row.hr)),
        ),
        axis=1,
    )
    return raw_data.sort_index()


def add_report_to_workspace(
    workspace: Workspace,
    project_name: str,
    project_description: str,
    report: Report,
) -> None:
    project = None

    for existing_project in workspace.list_projects():
        if existing_project.name == project_name:
            project = existing_project
            break

    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    workspace.add_report(project.id, report)


def build_scored_dataset(
    dataframe: pd.DataFrame,
    model,
    features: list[str],
    target: str,
    prediction: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe[prediction] = model.predict(dataframe[features])
    return dataframe.sort_index()


def rmse_score(
    dataframe: pd.DataFrame,
    target: str,
    prediction: str,
) -> float:
    return float(np.sqrt(np.mean((dataframe[target] - dataframe[prediction]) ** 2)))


def main() -> None:
    os.makedirs(WORKSPACE_NAME, exist_ok=True)

    # load data
    raw_data = _process_data(_fetch_data())

    # Feature selection
    target = 'cnt'
    prediction = 'prediction'
    numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
    categorical_features = ['season', 'holiday', 'workingday']
    features = numerical_features + categorical_features

    # Reference and current data split
    reference_jan11 = raw_data.loc['2011-01-01 00:00:00':'2011-01-28 23:00:00'].copy()
    current_feb11 = raw_data.loc['2011-01-29 00:00:00':'2011-02-28 23:00:00'].copy()

    # Train test split ONLY on reference_jan11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        reference_jan11[numerical_features + categorical_features],
        reference_jan11[target],
        test_size=0.3,
        random_state=42,
    )

    # Model training
    regressor = ensemble.RandomForestRegressor(random_state=0, n_estimators=50)
    regressor.fit(X_train, y_train)

    # Predictions
    preds_train = regressor.predict(X_train)
    preds_test = regressor.predict(X_test)

    # Add actual target and prediction columns to the training data for later
    # performance analysis
    X_train['target'] = y_train
    X_train['prediction'] = preds_train

    # Add actual target and prediction columns to the test data for later
    # performance analysis
    X_test['target'] = y_test
    X_test['prediction'] = preds_test

    # Initialize the column mapping object, which is evidently used to know how the
    # data is structured.
    column_mapping = ColumnMapping()

    # Map the actual target and prediction column names in the dataset for evidently
    column_mapping.target = 'target'
    column_mapping.prediction = 'prediction'

    # Specify which features are numerical and which are categorical for the
    # evidently report
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    workspace = Workspace.create(WORKSPACE_NAME)

    # Initialize the regression performance report with the default regression
    # metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the training data as reference and
    # test data as current
    # The data is sorted by index to ensure consistent ordering for the comparison
    regression_performance_report.run(reference_data=X_train.sort_index(),
                                      current_data=X_test.sort_index(),
                                      column_mapping=column_mapping)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        PROJECT_DESCRIPTION,
        regression_performance_report,
    )

    # Minimal exam addition:
    # the production model must be built on the whole January dataset.
    reference_jan_full = raw_data.loc['2011-01-01 00:00:00':'2011-01-31 23:00:00'].copy()

    # Train the production model
    regressor.fit(reference_jan_full[numerical_features + categorical_features],
                  reference_jan_full[target])

    # Perform column mapping
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.numerical_features = numerical_features
    column_mapping.categorical_features = categorical_features

    # Generate predictions for the reference data
    ref_prediction = regressor.predict(
        reference_jan_full[numerical_features + categorical_features]
    )
    reference_jan_full['prediction'] = ref_prediction

    # Initialize the regression performance report with the default regression
    # metrics preset
    regression_performance_report = Report(metrics=[
        RegressionPreset(),
    ])

    # Run the regression performance report using the reference data
    regression_performance_report.run(reference_data=None,
                                      current_data=reference_jan_full.sort_index(),
                                      column_mapping=column_mapping)
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        PROJECT_DESCRIPTION,
        regression_performance_report,
    )

    weekly_data = {}
    weekly_rmse = {}

    for week_name, (start, end) in WEEK_PERIODS.items():
        current_week = raw_data.loc[start:end].copy()
        current_week = build_scored_dataset(
            current_week,
            regressor,
            features,
            target,
            prediction,
        )

        weekly_data[week_name] = current_week
        weekly_rmse[week_name] = rmse_score(current_week, target, prediction)

        week_report = Report(metrics=[
            RegressionPreset(),
        ])
        week_report.run(
            reference_data=reference_jan_full.sort_index(),
            current_data=current_week.sort_index(),
            column_mapping=column_mapping,
        )
        add_report_to_workspace(
            workspace,
            PROJECT_NAME,
            PROJECT_DESCRIPTION,
            week_report,
        )

    worst_week = max(weekly_rmse, key=weekly_rmse.get)
    worst_week_data = weekly_data[worst_week]

    column_mapping_target = ColumnMapping()
    column_mapping_target.target = target
    column_mapping_target.prediction = prediction
    column_mapping_target.numerical_features = numerical_features
    column_mapping_target.categorical_features = []

    target_drift_report = Report(metrics=[
        TargetDriftPreset(),
    ])
    target_drift_report.run(
        reference_data=reference_jan_full.sort_index(),
        current_data=worst_week_data.sort_index(),
        column_mapping=column_mapping_target,
    )
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        PROJECT_DESCRIPTION,
        target_drift_report,
    )

    last_week_data = weekly_data["week_3"][numerical_features].copy()
    reference_numeric = reference_jan_full[numerical_features].copy()

    column_mapping_drift = ColumnMapping()
    column_mapping_drift.numerical_features = numerical_features
    column_mapping_drift.categorical_features = []

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    data_drift_report.run(
        reference_data=reference_numeric.sort_index(),
        current_data=last_week_data.sort_index(),
        column_mapping=column_mapping_drift,
    )
    add_report_to_workspace(
        workspace,
        PROJECT_NAME,
        PROJECT_DESCRIPTION,
        data_drift_report,
    )

    print("Monitoring pipeline completed.")
    print("Weekly RMSE:")
    print(json.dumps(weekly_rmse, indent=4))
    print(f"Worst week: {worst_week}")
    print(f"Open the UI with: evidently ui --workspace ./{WORKSPACE_NAME}")


if __name__ == "__main__":
    main()
