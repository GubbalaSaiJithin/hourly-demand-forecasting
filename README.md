# Hourly Demand Forecasting

A Python pipeline for rolling one-hour-ahead bike-rental forecasts. It compares gradient boosting with daily and weekly seasonal baselines, evaluates predictions chronologically, and produces SQL summaries and an HTML report.

## Quick start

From the project directory, using Python 3.10:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pipeline.py
.\.venv\Scripts\python.exe -m unittest -v
Start-Process .\artifacts\report.html
```

On macOS/Linux, substitute `.venv/bin/python` for `.\.venv\Scripts\python.exe`. The Python executable is called directly, so environment activation is optional.

The hourly dataset is included. To use another compatible CSV:

```powershell
.\.venv\Scripts\python.exe pipeline.py --data path/to/hour.csv --output artifacts
```

Required columns are `dteday` (date), `hr` (integer hour from 0 to 23), and `cnt` (non-negative rental count).

## Pipeline

1. Validate timestamps and counts, sort observations, and restore the complete hourly time grid. Missing observations remain missing.
2. Create calendar features, 1/24/168-hour demand lags, and past-only 24/168-hour rolling averages.
3. Keep rows with observed targets and the seasonal lags needed for a consistent comparison.
4. Split rows chronologically into 70% training, 15% validation and 15% test.
5. Compare two histogram gradient-boosting configurations using validation MAE, then refit the selected configuration on training plus validation.
6. Evaluate the final model and both seasonal baselines on the same test rows.
7. Save the model, predictions, metrics, SQLite database, daily SQL aggregates and HTML report.

The feature set excludes target components (`casual` and `registered`) and unknown future weather. Rolling averages are shifted before calculation so that the current target cannot enter its own features.

## Results

The full pipeline was rerun on 26 September 2026. The source contains 17,379 observations and 165 missing hours on the complete time grid. After eligibility filtering, 16,950 observations remain. The final test contains 2,543 hours from 12 September through 31 December 2012.

| Method | Test MAE, rentals/hour | Test RMSE |
|---|---:|---:|
| Previous day, same hour | 77.57 | 129.42 |
| Previous week, same hour | 66.43 | 112.61 |
| Histogram gradient boosting | 29.14 | 46.75 |

The model's MAE is 56.1% lower than the better seasonal baseline on this split. This is an offline comparison, not a measured business saving. `artifacts/metrics.json` records exact split dates, candidate scores, dependency versions and the dataset checksum.

## Outputs

| File | Contents |
|---|---|
| `artifacts/model.joblib` | Fitted model and feature names; generated locally |
| `artifacts/predictions.csv` | Actual values, predictions and seasonal baselines |
| `artifacts/metrics.json` | Evaluation, configuration and reproducibility metadata |
| `artifacts/demand.sqlite` | Prediction table for SQL analysis; generated locally |
| `artifacts/daily_summary.csv` | Daily totals and hourly MAE calculated with SQL |
| `artifacts/report.html` | Results table and recent daily summaries |

## Validation and limitations

Four regression tests cover future-target isolation, rolling-window alignment, missing hours and invalid input. The full pipeline and tests passed locally on Windows/Python 3.10 and in the configured GitHub workflow.

The model remains fixed during testing, but each prediction can use actual demand observed before that hour. This evaluates rolling one-hour-ahead forecasts; a fixed day-ahead forecast would require a different setup. Missing baseline lags cause rows to be excluded. Source timestamps use the dataset's local hour labels; production use would require explicit timezone and daylight-saving handling.

The dataset describes Washington DC in 2011-2012. Results do not establish performance in Berlin, on current data, or in production. There is no live data connection or prediction API.

## Further work

- Compare expanding-window validation with the current development split.
- Analyse errors by hour and weekday, including the impact of excluded observations.
- Add a simpler learned baseline and compare runtime as well as error.
- Add a prediction API with explicit feature-availability checks.

## References and licensing

- Fanaee-T, H. (2013). [Bike Sharing dataset, UCI](https://doi.org/10.24432/C5W894), CC BY 4.0. The data concern Capital Bikeshare in Washington DC, 2011-2012. `data/hour.csv` is included unchanged; `data/Readme.txt` contains the original dataset notes.
- [Original dataset download](https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip).
- [Scikit-learn forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) and its [source](https://github.com/scikit-learn/scikit-learn/blob/main/examples/applications/plot_time_series_lagged_features.py) are methodology references. Scikit-learn uses the BSD-3-Clause license.

The project code is provided under the included [MIT license](LICENSE). The dataset and dependencies retain their own licenses.
