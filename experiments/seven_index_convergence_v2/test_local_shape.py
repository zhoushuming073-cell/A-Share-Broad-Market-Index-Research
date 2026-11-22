import unittest

import numpy as np
import pandas as pd

from local_shape import OUT, metrics


class LocalShapeTests(unittest.TestCase):
    def test_parallel_and_drifting_paths(self):
        common = np.sin(np.arange(300) / 7)
        paths = pd.DataFrame({"a": common + .2, "b": common})
        parallel = metrics(paths, paths.diff())
        self.assertAlmostEqual(parallel["level"], .2)
        self.assertAlmostEqual(parallel["centered"], 0)
        self.assertAlmostEqual(parallel["rolling_120"], 0)
        paths["a"] += np.arange(300) * .001
        drift = metrics(paths, paths.diff())
        self.assertGreater(drift["centered"], .08)
        paths["a"] -= .5
        shifted = metrics(paths, paths.diff())
        for key in ["centered", "rolling_60", "rolling_120", "rolling_252"]:
            self.assertAlmostEqual(shifted[key], drift[key])

    def test_second_half_and_archive(self):
        close = pd.read_csv(OUT/"input_closes_main.csv", index_col=0)
        result = pd.read_csv(OUT/"local_shape_pairs.csv")
        pair = result.query("period=='late' and method=='original'")
        self.assertEqual(len(pair), 21)
        for row in pair.itertuples():
            a, b = row.pair.split("_")
            ratio = close[a]/close[b]
            gap = np.log(ratio / ratio.iloc[0]).loc[row.start:row.end]
            self.assertEqual(len(gap), row.days)
            self.assertAlmostEqual(row.centered, np.linalg.norm(gap-gap.mean()) / np.sqrt(len(gap)))
            windows = np.lib.stride_tricks.sliding_window_view(gap, 120)
            self.assertAlmostEqual(row.rolling_120, windows.std(axis=1).mean())
        ref = pd.read_csv(OUT/"independent_pair_deletion.csv").query("sample=='main' and K==10")
        actual = result.query("period=='full' and method=='pair_K10'").set_index("pair")
        np.testing.assert_allclose(actual.loc[ref.pair, "level_gain"], ref.improvement)
        for _, g in result.groupby(["method", "pair"]):
            g = g.set_index("period")
            for field in ["days", "removed"]:
                self.assertEqual(g.at["full",field], g.at["early",field]+g.at["late",field])


if __name__ == "__main__":
    unittest.main()
