import tempfile
import unittest
from pathlib import Path
import numpy as np
from model import Model, Attention, features, normalize, tokens, DIM
from solution import read_data, validate_data, stress_text

class SolutionTests(unittest.TestCase):
    # Train/validation/test must not share groups or texts, and labels must be valid.
    def test_split_integrity(self):
        validate_data({s:read_data(s) for s in ('train','validation','test')})
    def test_emoji_preserved(self):
        self.assertEqual(normalize('  BOHOT  badhiya 😒?! '),'bohot badhiya 😒?!')
        self.assertFalse(np.array_equal(features('badhiya service')[0],features('badhiya service 😒')[0]))
    # In 'strip_symbols' mode the emoji must have NO effect on the features.
    def test_symbol_ablation(self):
        a,x=features('badhiya service','strip_symbols')
        b,y=features('badhiya service 😒','strip_symbols')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)
    # "nahi" must survive cleaning and change the features ("accha" vs "accha nahi").
    def test_negation_preserved(self):
        self.assertIn('nahi',normalize('accha nahi hai'))
        self.assertFalse(np.array_equal(features('accha hai')[0],features('accha nahi hai')[0]))
    # Uppercase and extra spaces must not matter.
    def test_case_and_whitespace(self):
        a,x=features('ORDER   cancel');b,y=features('order cancel')
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)
    # Complex emojis and other scripts (Tamil, Bengali, Devanagari) must work,
    # and every feature vector must have length 1.
    def test_unicode_and_zwj(self):
        self.assertEqual(normalize('👩🏽‍💻'), '👩🏽‍💻')
        for t in ['தமிழ் super','বাংলা bhalo','हिंदी accha','🥲?!']:
            i,v=features(t);self.assertGreater(len(i),0);self.assertAlmostEqual(float(v@v),1,places=5)
    # Empty, blank, too-long (513 chars) and non-string inputs must be rejected.
    def test_input_limits(self):
        for text in ['', '   ', 'x'*513]:
            with self.assertRaises(ValueError):features(text)
        features('x'*512)
        with self.assertRaises(TypeError):features(None)
    # Model must stay under the 500 million parameter limit.
    def test_parameter_limit(self):
        m=Model();self.assertLess(m.parameter_count(),500_000_000)
    # Saving and re-loading a model must give identical predictions.
    def test_save_reload(self):
        m=Model.load(Path(__file__).parent/'artifacts/full.npz')
        text='mera parcel mila hi nhi 😒'
        with tempfile.TemporaryDirectory() as d:
            m.save(Path(d)/'m.npz');n=Model.load(Path(d)/'m.npz')
            self.assertEqual(m.predict(text),n.predict(text))
    # Output probabilities must be finite numbers that sum to 1.
    def test_probabilities(self):
        m=Model.load(Path(__file__).parent/'artifacts/full.npz')
        for p in m.probabilities('refund kab milega bhai'):
            self.assertTrue(np.all(np.isfinite(p)));self.assertAlmostEqual(float(p.sum()),1,places=5)
    # The stress-test misspeller must keep negation and emoji/punctuation cues.
    def test_stress_preserves_semantics_cues(self):
        text=stress_text('order nahi mila 😒!!!')
        self.assertIn('nahii',text);self.assertIn('😒!!!',text)

    # Hand-written attention gradients must match finite differences.
    def test_attention_gradients(self):
        att=Attention(seed=1)
        rng=np.random.default_rng(0)
        att.params={k:v.astype(np.float64) for k,v in att.params.items()}
        att.params['P']+=rng.normal(0,.1,att.params['P'].shape); att.params['Wout']*=50
        _,flat,lens=tokens('accha nahi laga ordr cancel kr do 😒')
        dz=rng.normal(size=9)
        loss=lambda: float(att.forward(flat,lens)[0]@dz)
        g=att.backward(att.forward(flat,lens)[1],dz)
        rows,gE=g['E']
        checks=[(k,(1,2) if att.params[k].ndim==2 else (3,),g[k][(1,2) if att.params[k].ndim==2 else (3,)]) for k in Attention.DENSE]
        checks.append(('E',(rows[0],5),gE[0,5]))
        for k,idx,analytic in checks:
            old=att.params[k][idx]
            att.params[k][idx]=old+1e-6; a=loss(); att.params[k][idx]=old-1e-6; b=loss(); att.params[k][idx]=old
            self.assertAlmostEqual((a-b)/2e-6,float(analytic),places=5,msg=k)
    # Context matters: word order changes the attention branch, and models saved
    # before the attention branch existed still load as linear-only models.
    def test_attention_order_and_legacy_load(self):
        m=Model.load(Path(__file__).parent/'artifacts/full.npz')
        self.assertIsNotNone(m.att)
        self.assertFalse(np.allclose(m.logits('accha nahi hai'),m.logits('nahi accha hai')))
        self.assertAlmostEqual(sum(a for _,a in m.attention_weights('refund kab milega')),1,places=5)
        with tempfile.TemporaryDirectory() as d:
            Model(attention=False).save(Path(d)/'old.npz')
            self.assertIsNone(Model.load(Path(d)/'old.npz').att)

if __name__=='__main__': unittest.main()
