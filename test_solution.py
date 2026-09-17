import tempfile
import unittest
from pathlib import Path
import numpy as np
from model import Model, features, normalize, DIM
from solution import read_data, validate_data, stress_text

class SolutionTests(unittest.TestCase):
    def test_split_integrity(self):
        validate_data({s:read_data(s) for s in ('train','validation','test')})
    def test_emoji_preserved(self):
        self.assertEqual(normalize('  BOHOT  badhiya 😒?! '),'bohot badhiya 😒?!')
        self.assertFalse(np.array_equal(features('badhiya service')[0],features('badhiya service 😒')[0]))
    def test_symbol_ablation(self):
        a,x=features('badhiya service','strip_symbols')
        b,y=features('badhiya service 😒','strip_symbols')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)
    def test_negation_preserved(self):
        self.assertIn('nahi',normalize('accha nahi hai'))
        self.assertFalse(np.array_equal(features('accha hai')[0],features('accha nahi hai')[0]))
    def test_case_and_whitespace(self):
        a,x=features('ORDER   cancel');b,y=features('order cancel')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)
    def test_unicode_and_zwj(self):
        self.assertEqual(normalize('👩🏽‍💻'), '👩🏽‍💻')
        for t in ['தமிழ் super','বাংলা bhalo','हिंदी accha','🥲?!']:
            i,v=features(t);self.assertGreater(len(i),0);self.assertAlmostEqual(float(v@v),1,places=5)
    def test_input_limits(self):
        for text in ['', '   ', 'x'*513]:
            with self.assertRaises(ValueError):features(text)
        features('x'*512)
        with self.assertRaises(TypeError):features(None)
    def test_parameter_limit(self):
        m=Model();self.assertLess(m.w.size+m.b.size,500_000_000)
    def test_save_reload(self):
        m=Model.load(Path(__file__).parent/'artifacts/full.npz')
        text='mera parcel mila hi nhi 😒'
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d)/'m.npz');n=Model.load(Path(d)/'m.npz')
            self.assertEqual(m.predict(text),n.predict(text))
    def test_probabilities(self):
        m=Model.load(Path(__file__).parent/'artifacts/full.npz')
        for p in m.probabilities('refund kab milega bhai'):
            self.assertTrue(np.all(np.isfinite(p)));self.assertAlmostEqual(float(p.sum()),1,places=5)
    def test_stress_preserves_semantics_cues(self):
        text=stress_text('order nahi mila 😒!!!')
        self.assertIn('nahii',text);self.assertIn('😒!!!',text)

if __name__=='__main__': unittest.main()
