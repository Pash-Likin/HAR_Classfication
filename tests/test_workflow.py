"""在隔离临时目录运行小规模合成夹具；检查接口衔接，不评价HAR效果。"""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import tempfile
import unittest
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from test_core import fixture, small_config, ROOT
sys.path.insert(0, str(ROOT / "src"))
from har_study.study import (grouped_folds, search_candidates, select_configs,
                            fit_selected, evaluate_test, paired_subject_bootstrap, benchmark)


class WorkflowTest(unittest.TestCase):
    def test_synthetic_module_integration(self):
        with tempfile.TemporaryDirectory(prefix="har_software_test_") as td, threadpool_limits(limits=1), redirect_stdout(io.StringIO()):
            out = Path(td)
            (out / "tables").mkdir(); (out / "models").mkdir()
            train, test, c = fixture(range(1, 13)), fixture(range(13, 16), seed=23), small_config()
            c['models']=['RBF_SVM','RandomForest','LogisticRegression','KNN','FisherLDA']
            c['lr_grid']={'model__C':[1.0]}
            c['knn_grid']={'model__n_neighbors':[3], 'model__weights':['uniform']}
            folds, _ = grouped_folds(train, c)
            candidate = search_candidates(train, c, folds, out, workers=1, cache=out / ".cache")
            self.assertEqual(len(candidate), 18)
            chosen, curve = select_configs(candidate)
            self.assertEqual(len(chosen), 10)
            self.assertEqual(len(curve), 15)
            fitted = fit_selected(train, chosen, c, out)
            stages = pd.read_csv(out / "tables/final_fit_and_size.csv")
            self.assertEqual(len(stages), 10)
            self.assertTrue({"scale_fit_seconds", "reduce_fit_seconds",
                             "classifier_fit_seconds", "final_fit_seconds"} <= set(stages.columns))
            self.assertTrue((stages.final_fit_seconds > 0).all())
            evaluate_test(test, fitted, chosen, out)
            paired_subject_bootstrap(out, c)
            benchmark(fitted, test.X, c, out)
            preds = pd.read_csv(out / "tables/test_predictions.csv")
            self.assertEqual(len(preds), len(test.y) * 10)
            scores = pd.read_csv(out / "tables/test_metrics.csv")
            self.assertEqual(len(scores), 10)
            self.assertTrue(scores.accuracy.between(0, 1).all())
            self.assertEqual(len(pd.read_csv(out / "tables/timing_raw.csv")), 40)
            self.assertEqual(set(candidate.method), {"full", "pca"})
            self.assertEqual(len(pd.read_csv(out / "tables/subject_bootstrap.csv")), 5)
            self.assertFalse((out / "tables/selected_features.csv").exists())
            for sid, pipe in fitted.items():
                reloaded = joblib.load(out / "models" / f"{sid}.joblib")
                np.testing.assert_array_equal(pipe.predict(test.X), reloaded.predict(test.X))
            # 所有夹具文件在离开TemporaryDirectory时删除，不放进正式outputs/。


if __name__ == "__main__": unittest.main()
