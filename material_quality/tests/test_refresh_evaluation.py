"""验证时间隔离和新旧模型配对比较的边界。"""

import datetime
import unittest

import numpy as np

from material_quality.training.evaluate_refresh_model import purged_masks, paired_auc_intervals


class RefreshEvaluationTest(unittest.TestCase):
    """防止七日标签穿越训练与测试边界。"""

    def test_label_window_must_close_before_test(self):
        start = datetime.date(2026,9,15)
        dates = np.array([start.toordinal()+v for v in [-8,-7,-6,-1,0,6,7]])
        train,test = purged_masks(dates,start,start+datetime.timedelta(days=7))
        np.testing.assert_array_equal(train,[True,True,False,False,False,False,False])
        np.testing.assert_array_equal(test,[False,False,False,False,True,True,False])
        self.assertFalse(np.any(train & test))

    def test_identical_predictions_have_zero_paired_difference(self):
        labels = np.array([0,1,0,1,0,1,0,1])
        predictions = np.array([.1,.8,.4,.5,.2,.7,.3,.6])
        result = paired_auc_intervals(labels,predictions,predictions,iterations=30)
        self.assertEqual(result['auc_delta_95_interval'],[0.0,0.0])
