"""PCA联合网格、维数比较、Fisher及断点恢复的测试。"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from test_core import fixture, config, small_config, ROOT
sys.path.insert(0,str(ROOT))
import run_experiment as runner
from har_study.study import grouped_folds,make_pipeline,validate_config,select_configs

class CompactTests(unittest.TestCase):
    def test_plan_counts(self):
        c=runner.build_config();p=runner.build_search_plan(c)
        self.assertEqual(p['parameter_counts'],dict(RBF_SVM=8,RandomForest=6,LogisticRegression=5,KNN=8,FisherLDA=5))
        self.assertEqual(p['n_candidates'],160);self.assertEqual(p['n_cv_fits'],800)
        self.assertEqual(p['n_final_models'],10)
        self.assertNotIn('n_protocol_fits',p)  # 本版仅按人员分组CV，不运行旧协议对照
        self.assertEqual(p['pca_dimensions'],[32,64,96,128])
        self.assertEqual(p['methods'],['full','pca'])
        self.assertEqual(p['pca_selection'],'highest_cv_macro_f1')
        self.assertNotIn('allowed_macro_f1_drop',c)

    def test_all_dimensions_share_the_same_model_grid(self):
        c=runner.build_config();p=runner.build_search_plan(c)
        specs=pd.DataFrame(runner.candidate_specs(c,p))
        self.assertEqual(set(specs.method),{'full','pca'})
        self.assertEqual(set(specs[specs.method=='pca'].k),{32,64,96,128})
        self.assertEqual(set(specs[specs.method=='full'].k),{561})
        for model,part in specs.groupby('model'):
            expected={runner.canonical(x) for x in p['parameter_sets'][model]}
            for _,group in part.groupby(['method','k']):
                self.assertEqual(set(group.params_json),expected)
    def test_old_grid_subset(self):
        p=runner.build_search_plan(runner.build_config())
        for m,g in runner.previous_grids().items():
            self.assertLessEqual({runner.canonical(x) for x in ParameterGrid(g)},
                                 {runner.canonical(x) for x in p['parameter_sets'][m]})
    def test_seed_plan_repeatable(self):
        c=runner.build_config()
        self.assertEqual(runner.build_search_plan(c),runner.build_search_plan(c))
    def test_fisher_is_classifier_not_transformer(self):
        c=config();d=fixture(range(1,7))
        for shrinkage in c['lda_grid']['model__shrinkage']:
            pipe=make_pipeline('FisherLDA','full',24,c)
            pipe.set_params(model__shrinkage=shrinkage).fit(d.X,d.y)
            clf=pipe.named_steps['model']
            self.assertEqual(clf.solver,'lsqr')
            self.assertEqual(clf.coef_.shape,(6,24))
            self.assertEqual(set(clf.classes_),set(range(1,7)))
    def test_invalid_shrinkage_rejected(self):
        c=config();c['lda_grid']['model__shrinkage']=[2]
        with self.assertRaises(ValueError):validate_config(c)
    def test_fisher_grid_is_five_distinct_configs(self):
        c = runner.build_config()
        self.assertEqual(c['lda_grid']['model__shrinkage'], [None, 'auto', 0.05, 0.2, 0.5])
        fisher = runner.build_search_plan(c)['parameter_sets']['FisherLDA']
        self.assertEqual(len(fisher), 5)
        self.assertEqual(len({runner.canonical(p) for p in fisher}), 5)
        for params in fisher:
            self.assertEqual(set(params), {'model__shrinkage'})

    def test_duplicate_shrinkage_rejected(self):
        c=config()
        c['lda_grid']['model__shrinkage']=[None, 'auto', 0.2, 0.2]
        with self.assertRaises(ValueError): validate_config(c)

    def test_fisher_fingerprint_changes(self):
        c=config();p=runner.build_search_plan(c)
        token=runner.fingerprint({'config':c,'plan':p})
        c['lda_grid']['model__shrinkage']=[None,'auto',.1]
        self.assertNotEqual(token,runner.fingerprint({'config':c,'plan':p}))
    def test_completed_checkpoints_reused(self):
        c=small_config();c['models']=['FisherLDA'];d=fixture(range(1,13))
        folds,_=grouped_folds(d,c);plan=runner.build_search_plan(c)
        with tempfile.TemporaryDirectory() as td,threadpool_limits(limits=1),redirect_stdout(io.StringIO()):
            out=Path(td);(out/'tables').mkdir()
            original=runner.run_search(d,c,plan,folds,out,1,'test-only-context')
            self.assertEqual(len(list((out/'checkpoints').glob('*.json'))),18)
            with patch.object(runner,'score_fold',side_effect=AssertionError('completed task reran')):
                repeated=runner.run_search(d,c,plan,folds,out,1,'test-only-context')
            pd.testing.assert_frame_equal(original,repeated)
            first=next((out/'checkpoints').glob('*.json'))
            content=runner.read_json(first);content['context_sha256']='wrong';runner.atomic_json(first,content)
            with self.assertRaises(ValueError):runner.run_search(d,c,plan,folds,out,1,'test-only-context')
    def test_selection_lock_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'tables').mkdir()
            (out/'tables/cv_candidates.csv').write_text('a\n1\n')
            runner.atomic_json(out/'config_frozen.json',{})
            runner.atomic_json(out/'search_plan.json',{})
            one=pd.DataFrame([{'selection_id':'FisherLDA__full','candidate_id':'first'}])
            runner.lock_selection(out,one)
            self.assertEqual(runner.lock_selection(out,one)['selected'][0]['candidate_id'],'first')
            one.loc[0,'candidate_id']='changed'
            with self.assertRaises(ValueError):runner.lock_selection(out,one)
    def test_new_model_has_no_fake_old_grid_result(self):
        c=small_config();c['models']=['FisherLDA']
        specs=runner.candidate_specs(c,runner.build_search_plan(c),24)
        rows=[dict(row,cv_macro_f1=.9,cv_macro_f1_std=.01) for row in specs]
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'tables').mkdir()
            runner.comparison_tables(pd.DataFrame(rows),c,out)
            old=(out/'tables/old_vs_extended_cv.csv').read_text(encoding='utf-8-sig')
            self.assertNotIn('FisherLDA',old)
            self.assertEqual(len(pd.read_csv(out/'tables/fixed_parameter_comparison.csv')),3)

    def test_four_dimension_comparison_reuses_cv_and_marks_highest_score(self):
        c=runner.build_config();c['models']=['FisherLDA']
        specs=runner.candidate_specs(c,runner.build_search_plan(c))
        # Synthetic scores exercise selection logic; they are not HAR results.
        scores={561:.96,32:.94,64:.96,96:.97,128:.98}
        rows=[]
        for spec in specs:
            bonus=.005 if json.loads(spec['params_json'])['model__shrinkage']=='auto' else 0
            rows.append(dict(spec,cv_macro_f1=scores[spec['k']]+bonus,cv_macro_f1_std=.01))
        candidates=pd.DataFrame(rows)
        selected,_=select_configs(candidates)
        with tempfile.TemporaryDirectory() as td:
            out=Path(td);(out/'tables').mkdir()
            runner.comparison_tables(candidates,c,out)
            runner.dimension_comparison_table(out,selected)
            table=pd.read_csv(out/'tables/pca_dimension_comparison.csv')
            self.assertEqual(list(table.k),[32,64,96,128])
            self.assertEqual(list(table.loc[table.is_selected_dimension,'k']),[128])
            self.assertEqual(list(table.cv_rank),[4,3,2,1])
            self.assertNotIn('meets_cv_tolerance',table)
            self.assertNotIn('cv_threshold',table)
            self.assertTrue((table.selection_rule=='highest_cv_macro_f1').all())
            np.testing.assert_allclose(table.cv_delta_from_full_pp,
                                       100*(table.tuned_cv_macro_f1-table.full_cv_macro_f1))
            self.assertTrue((table.tuned_cv_macro_f1>=table.fixed_cv_macro_f1).all())
            for row in table.itertuples():
                source=candidates[candidates.candidate_id==row.tuned_candidate_id].iloc[0]
                self.assertEqual(row.tuned_params_json,source.params_json)
                self.assertAlmostEqual(row.tuned_cv_macro_f1,source.cv_macro_f1)
            self.assertTrue(table.evidence_scope.str.contains('training_grouped_cv_only').all())
            self.assertFalse((out/'tables/test_metrics.csv').exists())

if __name__=='__main__':unittest.main()
