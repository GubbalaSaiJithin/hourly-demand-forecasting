# Hourly Demand Forecasting

A CPU-friendly learning project for data science and analytics internships. Estimate bike rentals for the next hour from past demand and calendar features. This is a small historical forecasting application, not a big-data platform or production deployment.

## Run on Windows PowerShell

Open a terminal in this project folder. Use Python 3.10 (tested) or a compatible newer version.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe pipeline.py
.\.venv\Scripts\python.exe -m unittest -v
Start-Process .\artifacts\report.html
```

The Python executable is called directly, so PowerShell activation-policy changes are unnecessary. On macOS/Linux, use `.venv/bin/python` in place of `.\.venv\Scripts\python.exe`.

## What is implemented

- Reads the included hourly dataset and checks duplicate timestamps and target validity.
- Restores the hourly time grid before creating lags, leaving missing observations missing.
- Uses calendar fields, 1/24/168-hour lags, and past-only rolling averages.
- Splits eligible rows chronologically: 70% training, 15% validation, 15% test.
- Compares two gradient-boosting configurations on validation MAE, refits the selected configuration on train + validation, and evaluates once on the final test period.
- Compares against previous-day and previous-week baselines on identical rows.
- Saves a fitted model, predictions, metrics, a SQLite database, SQL daily aggregates and an HTML report.

The SQL reporting step uses GROUP BY and aggregate functions. It does not claim an ETL orchestration system, distributed processing, or a production data warehouse.

## Actual local run: 24 September 2026

17,379 source observations; 165 unobserved hours on the complete time grid. After requiring observed targets and seasonal baseline lags, 16,950 rows are eligible. Test: 2,543 rows, 12 September–31 December 2012.

| Method | Test MAE, rentals/hour | Test RMSE |
|---|---:|---:|
| Previous day, same hour | 77.57 | 129.42 |
| Previous week, same hour | 66.43 | 112.61 |
| Histogram gradient boosting | 29.14 | 46.75 |

MAE is 56.1% lower than the better of these two seasonal baselines on this split. This is an offline result, not a measured business saving. The full split dates, dependency versions, dataset checksum and validation results are in `artifacts/metrics.json`.

The model stays fixed during test evaluation, but its lag inputs use actual demand observed before each test hour. This is rolling one-hour-ahead forecasting. It is not a fixed-origin forecast of an entire future day. Unknown future weather and target components (`casual`/`registered`) are excluded. Source timestamps are treated as the dataset's local hour labels; production use would need explicit timezone/DST handling.

## Make it your own

1. Add expanding-window validation on the development period. Keep the final test period untouched while choosing changes.
2. Analyse MAE by hour and weekday using SQL. Explain where the model fails and whether missing-hour filtering changes the evaluation population.
3. Add a simpler learned baseline and compare its accuracy and runtime. Do not assume more complexity wins.
4. Add a prediction API with input validation and integration tests. Only then list model serving on your CV.
5. Optional: publish a small dashboard with screenshots. Do not claim demand in Berlin: these data are from Washington DC.

Estimated learning and extension effort: about 20–30 focused hours if Python/pandas are familiar; longer if time-series evaluation is new. This is a planning estimate.

## Interview questions to answer

- Why is a random train/test split unsuitable here?
- How can missing timestamps break `shift(24)`?
- Why are `casual` and `registered` excluded?
- Which values are available when predicting hour t?
- Why use MAE alongside RMSE, and what does MAE 29.14 mean?
- Is this suitable for day-ahead staffing? What would need to change?
- Which part did you personally extend, and did it improve validation results?

## CV wording after you reproduce and extend it

“Developed a rolling hourly demand-forecasting pipeline using Python, pandas, scikit-learn and SQLite; compared seasonal baselines with gradient boosting using chronological validation.”

Add your own measured held-out MAE and baseline comparison after reproducing the experiment. Credit this AI-assisted starter in your README and explain your contribution. Do not copy the result into the CV before you can explain the experiment.

## Open-source references and data attribution

- [Official scikit-learn forecasting example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) and [example source on GitHub](https://github.com/scikit-learn/scikit-learn/blob/main/examples/applications/plot_time_series_lagged_features.py). Scikit-learn is BSD-3-Clause licensed. These are learning references; this starter is a separately authored implementation, not a fork of the example.
- Fanaee-T, H. (2013). [Bike Sharing dataset, UCI](https://doi.org/10.24432/C5W894), CC BY 4.0. Original data concern Capital Bikeshare in Washington DC, 2011–2012. `data/hour.csv` is included unchanged; transformations happen in the pipeline. `data/Readme.txt` contains the original dataset notes.
- Dataset download: https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip

The included MIT license applies to the new starter code, not to the dataset or dependencies. The full pipeline and four regression tests passed locally on 26 September 2026, including past-only features, missing hours and invalid inputs. Fresh-environment installation, deployment and production performance are not verified.
