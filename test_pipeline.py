import unittest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from pipeline import FEATURES, make_features, load_data

class ForecastTests(unittest.TestCase):
    def test_invalid_hours_and_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv'
            for hour,count in [(24,1),(1.5,1),(1,float('inf')),(1,-1)]:
                pd.DataFrame({'dteday':['2024-01-01'],'hr':[hour],'cnt':[count]}).to_csv(path,index=False)
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_missing_hour_remains_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.csv'
            pd.DataFrame({'dteday':['2024-01-01']*2,'hr':[0,2],'cnt':[1,3]}).to_csv(path,index=False)
            series=load_data(path)
            self.assertEqual(len(series),3)
            self.assertTrue(pd.isna(series.iloc[1]))

    def test_features_do_not_read_present_or_future_targets(self):
        index = pd.date_range('2024-01-01', periods=400, freq='h')
        original = pd.Series(np.arange(400, dtype=float), index=index)
        changed = original.copy()
        changed.iloc[250:] = 999999
        pd.testing.assert_frame_equal(make_features(original).iloc[:251], make_features(changed).iloc[:251])

    def test_shift_excludes_current_hour(self):
        series = pd.Series(np.arange(300, dtype=float), index=pd.date_range('2024-01-01', periods=300, freq='h'))
        frame = make_features(series)
        self.assertEqual(frame.iloc[200].lag_24, 176)
        self.assertEqual(frame.iloc[200].mean_24, np.arange(176, 200).mean())
        self.assertTrue({'cnt', 'casual', 'registered', 'target'}.isdisjoint(FEATURES))

if __name__ == '__main__':
    unittest.main()
