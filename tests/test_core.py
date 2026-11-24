"""软件正确性测试。夹具是小型合成数据，不提供HAR实验性能证据。"""
from __future__ import annotations
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from har_study.data import (ACTIVITIES, Split, audit_splits, extract_official_archive,
                            find_dataset_root, load_split, prepare_dataset)
from har_study.study import (grouped_folds, make_pipeline, metrics, select_configs, validate_config)
from har_study.verify import verify_run


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def fixture(subject_ids, seed=11, dimensions=24):
    rng = np.random.default_rng(seed)
    X, y, groups = [], [], []
    for person in subject_ids:
        offset = rng.normal(0, .12, dimensions)
        for cls in range(1, 7):
            for _ in range(3):
                vector = rng.normal(0, .5, dimensions) + offset
                vector[(cls - 1) * 2:(cls - 1) * 2 + 2] += 3
                X.append(vector); y.append(cls); groups.append(person)
    return Split(np.array(X), np.array(y), np.array(groups), [f"f{i:03d}" for i in range(dimensions)])


def small_config():
    c = config()
    c.update(models=["RBF_SVM", "RandomForest"], cv_folds=3, dimensions=[4, 8], rf_trees=12, timing_repeats=2,
             timing_warmups=1, latency_sample_count=8, bootstrap_repeats=200)
    c["svm_grid"] = {"model__C": [1.0], "model__gamma": ["scale"]}
    c["rf_grid"] = {"model__max_depth": [None], "model__min_samples_leaf": [1]}
    c["lda_grid"] = {"model__shrinkage": [None, "auto"]}  # 测试夹具仅用两组；正式配置包含五组。
    return c


def tiny_files():
    X = np.arange(72).reshape(12, 6) / 72
    def txt(a):
        b = io.StringIO(); np.savetxt(b, a); return b.getvalue().encode()
    result = {"features.txt": "\n".join(f"{i} f{i}" for i in range(1, 7)).encode(),
              "activity_labels.txt": "\n".join(f"{i} {name}" for i, name in ACTIVITIES.items()).encode()}
    for name, person in (("train", 1), ("test", 2)):
        result[f"{name}/X_{name}.txt"] = txt(X)
        result[f"{name}/y_{name}.txt"] = ("\n".join(map(str, [1, 2, 3, 4, 5, 6] * 2))).encode()
        result[f"{name}/subject_{name}.txt"] = ("\n".join([str(person)] * 12)).encode()
    return result


class CoreTests(unittest.TestCase):
    def test_default_config(self): validate_config(config())

    def test_repeated_dimensions_rejected(self):
        c = config(); c["dimensions"] = [20, 20]
        with self.assertRaises(ValueError): validate_config(c)

    def test_feature_dimension_bound(self):
        c = config(); c["dimensions"] = [561]
        with self.assertRaises(ValueError): validate_config(c)

    def test_pca_selection_requires_highest_cv_score(self):
        c = config(); c["pca_selection"] = "smallest_eligible_k"
        with self.assertRaises(ValueError): validate_config(c)

    def test_legacy_tolerance_config_rejected(self):
        c = config(); c["allowed_macro_f1_drop"] = .01
        with self.assertRaises(ValueError): validate_config(c)

    def test_group_isolation_and_coverage(self):
        d = fixture(range(1, 13)); c = small_config()
        folds, rows = grouped_folds(d, c)
        self.assertEqual(len(rows), len(d.y))
        for tr, va in folds:
            self.assertFalse(set(d.subjects[tr]) & set(d.subjects[va]))
        self.assertEqual(sorted(r["row_id"] for r in rows), list(range(len(d.y))))

    def test_group_folds_repeatable(self):
        d = fixture(range(1, 13)); c = small_config()
        a, _ = grouped_folds(d, c); b, _ = grouped_folds(d, c)
        for (at, av), (bt, bv) in zip(a, b):
            np.testing.assert_array_equal(at, bt); np.testing.assert_array_equal(av, bv)

    def test_subject_not_in_input(self):
        d = fixture(range(1, 4))
        p = make_pipeline("RBF_SVM", "full", 24, small_config())
        p.fit(d.X, d.y)
        self.assertEqual(p.n_features_in_, 24)

    def test_scaler_only_fits_supplied_training_rows(self):
        d = fixture(range(1, 7)); tr = np.arange(len(d.y) // 2)
        p = make_pipeline("RBF_SVM", "pca", 4, small_config())
        p.fit(d.X[tr], d.y[tr])
        np.testing.assert_allclose(p.named_steps["scale"].mean_, d.X[tr].mean(axis=0))
        before = p.named_steps["scale"].mean_.copy()
        p.predict(d.X[tr] + 1e4)
        np.testing.assert_array_equal(before, p.named_steps["scale"].mean_)

    def test_pca_dimension(self):
        d = fixture(range(1, 4)); p = make_pipeline("RBF_SVM", "pca", 4, small_config()).fit(d.X, d.y)
        self.assertEqual(p[:-1].transform(d.X).shape, (len(d.y), 4))

    def test_feature_selection_route_removed(self):
        with self.assertRaises(ValueError):
            make_pipeline("RandomForest", "anova", 8, small_config())

    def test_clone_and_repeat_predictions(self):
        d = fixture(range(1, 4)); c = small_config()
        p = make_pipeline("RandomForest", "pca", 8, c)
        a = clone(p).fit(d.X, d.y).predict(d.X)
        b = clone(p).fit(d.X, d.y).predict(d.X)
        np.testing.assert_array_equal(a, b)

    def test_highest_pca_score_even_when_all_pca_scores_below_full(self):
        rows = []
        for model in ("RBF_SVM", "RandomForest"):
            pca_scores = [(32, .94), (64, .955), (96, .97), (128, .98)] if model == "RBF_SVM" else [
                (32, .90), (64, .93), (96, .92), (128, .91)]
            for method, vals in (("full", [(561, .96)]), ("pca", pca_scores)):
                for k, score in vals:
                    rows.append({"candidate_id": f"{model}_{method}_{k}", "model": model, "method": method,
                                 "k": k, "cv_macro_f1": score, "params_json": "{}"})
        chosen, _ = select_configs(pd.DataFrame(rows))
        pca = chosen[chosen.method == "pca"].set_index("model")
        self.assertEqual(pca.loc["RBF_SVM", "k"], 128)
        self.assertEqual(pca.loc["RandomForest", "k"], 64)
        self.assertAlmostEqual(pca.loc["RandomForest", "cv_delta_from_full_pp"], -3.0)
        self.assertTrue((pca.selection_rule == "highest_cv_macro_f1").all())
        self.assertTrue((chosen.selection_metric == "cv_macro_f1").all())
        self.assertNotIn("meets_cv_tolerance", chosen)
        self.assertNotIn("cv_threshold", chosen)

    def test_dimension_selection_uses_tuned_parameters_at_each_k(self):
        # 固定参数会偏向128维；64维调参后分数最高，应联合选中64维及其参数。
        values = [("full", 561, "a", .96), ("full", 561, "b", .94),
                  ("pca", 32, "a", .93), ("pca", 32, "b", .94),
                  ("pca", 64, "a", .91), ("pca", 64, "b", .985),
                  ("pca", 96, "a", .97), ("pca", 96, "b", .965),
                  ("pca", 128, "a", .98), ("pca", 128, "b", .98)]
        rows = [{"candidate_id": f"RBF_SVM_{method}_{k}_{parameter}", "model": "RBF_SVM",
                 "method": method, "k": k, "cv_macro_f1": score,
                 "params_json": json.dumps({"parameter": parameter})}
                for method, k, parameter, score in values]
        chosen, curve = select_configs(pd.DataFrame(rows))
        selected = chosen[chosen.method == "pca"].iloc[0]
        self.assertEqual(selected.k, 64)
        self.assertEqual(json.loads(selected.params_json)["parameter"], "b")
        self.assertEqual(json.loads(chosen[chosen.method == "full"].iloc[0].params_json)["parameter"], "a")
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[curve.k == 128].iloc[0].candidate_id, "RBF_SVM_pca_128_a")

    def test_exact_score_tie_prefers_smaller_dimension_then_stable_candidate(self):
        values = [("full", 561, "full", .96), ("pca", 32, "z_32", .97),
                  ("pca", 32, "b_32", .97), ("pca", 64, "a_64", .97),
                  ("pca", 96, "a_96", .96), ("pca", 128, "a_128", .95)]
        rows = [{"candidate_id": cid, "model": "RBF_SVM", "method": method,
                 "k": k, "cv_macro_f1": score, "params_json": "{}"}
                for method, k, cid, score in values]
        for records in (rows, list(reversed(rows))):
            chosen, _ = select_configs(pd.DataFrame(records))
            selected = chosen[chosen.method == "pca"].iloc[0]
            self.assertEqual(selected.k, 32)
            self.assertEqual(selected.candidate_id, "b_32")

    def test_selection_rejects_legacy_feature_selection_candidates(self):
        with self.assertRaises(ValueError):
            select_configs(pd.DataFrame([{"method": "anova"}]))

    def test_serialized_scores_preserve_near_tie_selection(self):
        from har_study.study import save_table
        rows = pd.DataFrame([
            {"candidate_id": "full", "model": "KNN", "method": "full", "k": 561,
             "cv_macro_f1": .94, "params_json": "{}"},
            {"candidate_id": "pca32", "model": "KNN", "method": "pca", "k": 32,
             "cv_macro_f1": .95, "params_json": "{}"},
            {"candidate_id": "pca128", "model": "KNN", "method": "pca", "k": 128,
             "cv_macro_f1": np.nextafter(.95, 1.0), "params_json": "{}"},
        ])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cv_candidates.csv"
            save_table(rows, path)
            loaded = pd.read_csv(path, float_precision="round_trip")
            np.testing.assert_array_equal(loaded.cv_macro_f1, rows.cv_macro_f1)
            chosen, _ = select_configs(loaded)
            self.assertEqual(chosen[chosen.method == "pca"].iloc[0].k, 128)

    def test_identical_prediction_metrics(self):
        values = np.array([1, 2, 3, 4, 5, 6])
        m = metrics(values, values)
        self.assertEqual(m["accuracy"], 1); self.assertEqual(m["macro_f1"], 1)

    def test_audit_rejects_people_overlap(self):
        d = fixture(range(1, 4))
        with self.assertRaises(ValueError): audit_splits(d, d)

    def test_audit_disjoint_people(self):
        a = fixture(range(1, 4)); b = fixture(range(4, 7), seed=19)
        self.assertEqual(audit_splits(a, b)["subject_overlap"], [])

    def test_safe_nested_zip_and_loading(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td); inner = io.BytesIO()
            with zipfile.ZipFile(inner, "w") as z:
                for name, content in tiny_files().items(): z.writestr("UCI HAR Dataset/" + name, content)
                z.writestr("../../must_not_escape.txt", "not extracted")
            outer = td / "official_wrapper.zip"
            with zipfile.ZipFile(outer, "w") as z:
                z.writestr("README.txt", "outer optional readme")
                z.writestr("UCI HAR Dataset.zip", inner.getvalue())
            root = extract_official_archive(outer, td / "dataset")
            self.assertFalse((td / "must_not_escape.txt").exists())
            self.assertEqual(find_dataset_root(root), root)
            d = load_split(root, "train", strict=False)
            self.assertEqual(d.X.shape, (12, 6))
            with self.assertRaises(ValueError): load_split(root, "train")

    def test_missing_subject_file_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                for name, content in tiny_files().items():
                    if name != "train/subject_train.txt": z.writestr(name, content)
            with self.assertRaises(ValueError): extract_official_archive(archive, Path(td) / "out")

    def test_no_synthetic_fallback_when_download_fails(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("urllib.request.urlopen", side_effect=OSError("offline test fixture")):
                with self.assertRaises(RuntimeError): prepare_dataset(None, Path(td))
            self.assertFalse(any(Path(td).rglob("X_train.txt")))

    def test_verifier_rejects_incomplete_and_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for m in ({"status": "running", "synthetic": False}, {"status": "completed", "synthetic": True}):
                (td / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises(ValueError): verify_run(td)

    def test_verifier_rejects_legacy_selection_protocol_before_reading_results(self):
        # 仅构造协议检查所需的临时元数据，不生成或宣称真实HAR成绩。
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "manifest.json").write_text('{"status":"completed","synthetic":false}')
            audit = {"train": {"subject_ids": [1], "samples": 7352},
                     "test": {"subject_ids": [2], "samples": 2947}}
            (td / "data_audit.json").write_text(json.dumps(audit))
            wrong_rule = config(); wrong_rule["pca_selection"] = "smallest_eligible_k"
            legacy = config(); legacy.pop("pca_selection"); legacy["allowed_macro_f1_drop"] = .01
            for frozen in (wrong_rule, legacy):
                (td / "config_frozen.json").write_text(json.dumps(frozen))
                with self.assertRaises(ValueError): verify_run(td)

    def test_plot_rejects_synthetic(self):
        from har_study.plots import draw_all
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "manifest.json").write_text('{"status":"completed","synthetic":true}')
            with self.assertRaises(ValueError): draw_all(td)


if __name__ == "__main__": unittest.main()
