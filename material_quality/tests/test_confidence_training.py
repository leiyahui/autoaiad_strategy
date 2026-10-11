"""验证曝光权重、时间窗口隔离和账户内评估边界。"""

import datetime
import unittest

import numpy as np

from material_quality.training.train_confidence_classifier import (
    training_weights,validation_folds,within_account_metrics,paired_account_interval,
)
from material_quality.training.train_serving_model import _fit_serving_towers


class ConfidenceTrainingTest(unittest.TestCase):
    """避免曝光进入预测输入或权重、切分发生静默错位。"""

    def test_low_exposure_is_downweighted_without_label_input(self):
        exposure=np.array([0,5,125,500,5000],dtype=float)
        weights=training_weights(exposure,np.array(['甲']*5),'exposure')
        self.assertTrue(np.all(np.diff(weights)>=0))
        self.assertEqual(weights[-1],weights[-2])
        self.assertAlmostEqual(weights[0]/weights[-1],0.1)
        self.assertAlmostEqual(weights.mean(),1)
        np.testing.assert_array_equal(training_weights(exposure,np.array(['甲']*5),'uniform'),np.ones(5))

    def test_invalid_exposure_is_rejected(self):
        for values in [[-1,1],[np.nan,1],[np.inf,1]]:
            with self.assertRaises(ValueError):
                training_weights(np.array(values),np.array(['甲','乙']),'exposure')

    def test_industry_balance_limits_large_group_dominance(self):
        groups=np.array(['甲']*16+['乙']*4)
        weights=training_weights(np.ones(20)*500,groups,'exposure_industry')
        self.assertAlmostEqual(weights[-1]/weights[0],2.0)

    def test_validation_labels_close_before_test_boundary(self):
        start=datetime.date(2026,9,15)
        dates=np.arange(start.toordinal()-80,start.toordinal()+10)
        folds=validation_folds(dates,start)
        self.assertEqual(len(folds),3)
        for fold in folds:
            boundary=datetime.date.fromisoformat(fold['start']).toordinal()
            self.assertTrue(np.all(dates[fold['train']]+7<=boundary))
            self.assertTrue(np.all(dates[fold['validation']]+7<=start.toordinal()))
            self.assertFalse(np.any(fold['train']&fold['validation']))

    def test_within_account_does_not_compare_different_accounts(self):
        labels=np.array([0,0,1,1,0,0,1,1])
        predictions=np.array([.01,.02,.03,.04,.8,.81,.82,.83])
        groups=np.array(['甲']*4+['乙']*4)
        result=within_account_metrics(labels,predictions,groups)
        self.assertEqual(result['macro_auc'],1)
        self.assertEqual(result['covered_samples'],8)
        self.assertEqual(paired_account_interval(labels,predictions,predictions,groups,30)['interval'],[0,0])

    def test_training_rejects_misaligned_weights_before_fitting(self):
        with self.assertRaises(ValueError):
            _fit_serving_towers({},np.array([0,1]),sample_weight=np.array([1.0]))
        with self.assertRaises(ValueError):
            _fit_serving_towers({},np.array([0,1]),sample_weight=np.array([1.0,0]))
