"""Reproducible, rolling one-hour-ahead bike-demand forecasting."""
from pathlib import Path
import argparse
import hashlib
import json
import sqlite3
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent
FEATURES = ['hour', 'weekday', 'month', 'is_weekend', 'lag_1', 'lag_24', 'lag_168', 'mean_24', 'mean_168']

def load_data(path):
    raw = pd.read_csv(path)
    needed = {'dteday', 'hr', 'cnt'}
    if not needed.issubset(raw.columns):
        raise ValueError(f'CSV must include {sorted(needed)}')
    raw['hr'] = pd.to_numeric(raw['hr'], errors='coerce')
    raw['cnt'] = pd.to_numeric(raw['cnt'], errors='coerce')
    if raw.hr.isna().any() or not raw.hr.between(0, 23).all() or not (raw.hr % 1 == 0).all():
        raise ValueError('Hour must be an integer from 0 to 23.')
    if raw.cnt.isna().any() or not np.isfinite(raw.cnt).all():
        raise ValueError('Demand counts must be finite numbers.')
    raw['timestamp'] = pd.to_datetime(raw['dteday']) + pd.to_timedelta(raw['hr'], unit='h')
    if raw.timestamp.isna().any() or raw.timestamp.duplicated().any() or (raw.cnt < 0).any() or raw.cnt.isna().any():
        raise ValueError('Duplicate timestamps, negative counts or missing targets found.')
    # Reindex before shifting: 24 rows in the source are not always 24 hours.
    series = raw.set_index('timestamp').sort_index()['cnt'].asfreq('h')
    return series

def make_features(series):
    frame = pd.DataFrame(index=series.index)
    frame['hour'] = frame.index.hour
    frame['weekday'] = frame.index.dayofweek
    frame['month'] = frame.index.month
    frame['is_weekend'] = (frame.index.dayofweek >= 5).astype(int)
    for lag in [1, 24, 168]:
        frame[f'lag_{lag}'] = series.shift(lag)
    for window in [24, 168]:
        frame[f'mean_{window}'] = series.shift(1).rolling(window, min_periods=window // 2).mean()
    return frame[FEATURES]

def score(actual, predicted):
    return {'mae': float(mean_absolute_error(actual, predicted)),
            'rmse': float(np.sqrt(mean_squared_error(actual, predicted)))}

def run(path=ROOT / 'data/hour.csv', output=ROOT / 'artifacts'):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    series = load_data(path)
    frame = make_features(series)
    frame['target'] = series
    # All candidates and baselines are compared on the same rows.
    frame = frame.dropna(subset=['target', 'lag_24', 'lag_168'])
    first, second = int(len(frame) * .70), int(len(frame) * .85)
    train, valid, test = frame.iloc[:first], frame.iloc[first:second], frame.iloc[second:]
    if min(len(train), len(valid), len(test)) < 100:
        raise ValueError('At least 100 observations are required in each time split.')
    candidates = []
    for leaves in [15, 31]:
        model = HistGradientBoostingRegressor(max_leaf_nodes=leaves, max_iter=150,
                    learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=42)
        model.fit(train[FEATURES], train.target)
        candidates.append((score(valid.target, np.maximum(0, model.predict(valid[FEATURES])))['mae'], leaves))
    best_mae, best_leaves = min(candidates)
    model = HistGradientBoostingRegressor(max_leaf_nodes=best_leaves, max_iter=150,
                    learning_rate=.08, l2_regularization=1., early_stopping=False, random_state=42)
    development = frame.iloc[:second]
    model.fit(development[FEATURES], development.target)
    prediction = np.maximum(0, model.predict(test[FEATURES]))
    metrics = {
        'task': 'Rolling one-hour-ahead forecast; previous observed hours available at every origin.',
        'dataset': 'UCI Bike Sharing (Washington DC, 2011-2012), NOT Berlin data.',
        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__},
        'source_observations': int(series.notna().sum()), 'missing_hours': int(series.isna().sum()),
        'eligible_observations': len(frame),
        'splits': {label: {'rows': len(part), 'start': str(part.index.min()), 'end': str(part.index.max())}
                   for label, part in [('train', train), ('validation', valid), ('test', test)]},
        'selection': {'validation_mae': best_mae, 'max_leaf_nodes': best_leaves,
                      'candidates': [{'max_leaf_nodes': leaves, 'validation_mae': mae} for mae, leaves in candidates]},
        'test': {'previous_day': score(test.target, test.lag_24),
                 'previous_week': score(test.target, test.lag_168),
                 'gradient_boosting': score(test.target, prediction)}
    }
    best_baseline = min(metrics['test']['previous_day']['mae'], metrics['test']['previous_week']['mae'])
    metrics['mae_reduction_vs_best_seasonal_baseline_percent'] = 100 * (best_baseline - metrics['test']['gradient_boosting']['mae']) / best_baseline
    results = pd.DataFrame({'actual': test.target, 'predicted': prediction,
                           'previous_day': test.lag_24, 'previous_week': test.lag_168})
    results.to_csv(output / 'predictions.csv', index_label='timestamp')
    with sqlite3.connect(output / 'demand.sqlite') as connection:
        results.reset_index().to_sql('predictions', connection, if_exists='replace', index=False)
        daily = pd.read_sql_query('''SELECT date(timestamp) AS day,
            SUM(actual) AS actual_rentals, SUM(predicted) AS predicted_rentals,
            AVG(ABS(actual - predicted)) AS hourly_mae
            FROM predictions GROUP BY date(timestamp) ORDER BY day''', connection)
        daily.to_csv(output / 'daily_summary.csv', index=False)
    joblib.dump({'model': model, 'features': FEATURES}, output / 'model.joblib')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    rows = ''.join(f'<tr><td>{name}</td><td>{value["mae"]:.2f}</td><td>{value["rmse"]:.2f}</td></tr>' for name, value in metrics['test'].items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Demand forecasting results</title>
    <style>body{{font:17px system-ui;max-width:900px;margin:50px auto;padding:20px;color:#173047}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}small{{color:#555}}</style>
    <h1>One-hour-ahead demand forecasting</h1><p>UCI Bike Sharing · Washington DC · Historical evaluation</p>
    <p>Chronological train/validation/test split. Previous observed hours are available for each prediction.
    This does not evaluate a full day forecast made at midnight.</p><table><tr><th>Method</th><th>MAE (rentals/hour)</th><th>RMSE</th></tr>{rows}</table>
    <p>Test period: {metrics['splits']['test']['start']} to {metrics['splits']['test']['end']}.</p>
    <h2>Daily totals on evaluated hours</h2>{daily.tail(14).round(2).to_html(index=False)}
    <p><small>Missing source hours are not treated as zero. Rows missing required seasonal baseline lags are excluded.
    Historical offline results do not establish production impact or performance in Berlin.</small></p></html>'''
    (output / 'report.html').write_text(html, encoding='utf-8')
    print(json.dumps(metrics, indent=2))
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT / 'data/hour.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts')
    args = parser.parse_args()
    run(args.data, args.output)
