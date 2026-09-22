"""Regression checks for merged inference, routing, missing geometry and saved heads."""
import json
import unittest

import numpy as np

from .. import config as C
from ..solution.geometry import features, spine_axis
from ..solution.model import Blend, GeometryBlend, Model, probability, HEADS
from .train_geometry import fit_geometry, geometry_probability


class GeometryCheck(unittest.TestCase):
    def test_axis_and_small_images(self):
        y, x = np.mgrid[:315, :300]
        img = (np.abs(x - (150 + np.tan(np.radians(6)) * (y - 157))) < 30).astype(np.uint8) * 170
        self.assertTrue(3.5 < spine_axis(img)[0] < 8.5)
        self.assertTrue(np.isnan(features(np.arange(16, dtype=np.uint8).reshape(4, 4), False)[:2]).all())
        self.assertTrue(np.isnan(features(np.zeros((4, 4), dtype=np.uint8), True)[2:]).all())

    def test_numeric_export_matches_sklearn(self):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        x = np.array([[0., np.nan], [1., 5.], [2., 8.], [3., np.nan], [4., 12.], [5., 15.]])
        y = np.array([0, 0, 1, 0, 1, 1])
        m = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(C=1., class_weight='balanced', solver='liblinear', random_state=42, max_iter=2000)).fit(x, y)
        h = json.loads(json.dumps(fit_geometry(x, y), allow_nan=False))
        np.testing.assert_allclose(geometry_probability(x, h), m.predict_proba(x)[:, 1], atol=1e-12)

    def test_routing_batch_independence_and_invalid_heads(self):
        members = []
        for _ in range(2):
            m = Model.__new__(Model)
            m.metadata = {'dimensions': 1, 'preprocess': {}}
            m.heads = {k:{'coef':[1.], 'intercept':0., 'threshold':.5} for k in HEADS}
            members.append(m)
        h = {'spine_axis':{'coef':[.2,.1],'intercept':-.3,'impute':[0.,0.]},
             'hip_roi':{'coef':[.1]*5,'intercept':-.2,'impute':[0.]*5}}
        model = GeometryBlend(members, {k:{'threshold':.5} for k in HEADS}, h)
        x = np.array([[-2., -2., 4., 5., 0., 0., 0., 0., 0.], [2., 2., 0., 0., 200., 4., 0., np.nan, 1.]])
        frame, scores = model.classify(x)
        self.assertEqual(frame.anatomical_region.tolist(), [C.REGION_SPINE, C.REGION_FEMUR])
        for i in range(2):
            single, _ = model.classify(x[i:i+1])
            self.assertEqual(single.iloc[0].to_dict(), frame.iloc[i].to_dict())
        self.assertTrue(np.isfinite(frame.quality_prob).all())
        self.assertEqual(frame.quality_class.tolist(), frame.violation_type.astype(bool).astype(int).tolist())
        for key, types in [('spine_quality', C.TARGETS[:3]), ('hip_quality', C.TARGETS[3:])]:
            direct = probability(x[:, :1], members[0].heads[key])
            np.testing.assert_allclose(scores[key], .5*direct+.5*(1-np.prod([1-scores[t] for t in types],axis=0)))
        with self.assertRaises(ValueError):
            GeometryBlend(members, model.heads, {})
        bad = json.loads(json.dumps(h)); bad['spine_axis']['coef'][0] = float('nan')
        with self.assertRaises(ValueError):
            GeometryBlend(members, model.heads, bad)


if __name__ == '__main__':
    unittest.main()
