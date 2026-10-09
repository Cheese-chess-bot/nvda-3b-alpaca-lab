import json, tempfile, unittest
from pathlib import Path
import numpy as np
from core import *


class CoreTests(unittest.TestCase):
    def test_cash(self):
        env=MarketEnv(np.zeros((3,2)),[.1,-.2,.3]); infos=[]
        for _ in range(3): infos.append(env.step(0)[3])
        self.assertEqual(metrics(infos)['total_return'],0.)

    def test_exact_roundtrip_cost(self):
        env=MarketEnv(np.zeros((1,2)),[0]); _,_,done,info=env.step(4)
        expected=10000/(1+.001*.1)*(1-.001*.1)
        self.assertAlmostEqual(info['equity'],expected,places=9); self.assertTrue(done)
        self.assertAlmostEqual(env.weight,0)

    def test_weight_drift_and_rebalance(self):
        env=MarketEnv(np.zeros((2,1)),[.10,0]); _,_,_,info=env.step(4)
        self.assertAlmostEqual(env.weight,.11/1.01)
        old=env.equity; oldw=env.weight; target=min(.1,1000/old)
        fee=.001*(oldw-target)/(1-.001*target)
        _,_,_,info=env.step(4)
        self.assertAlmostEqual(info['equity'],old*(1-fee)*(1-.001*target),places=8)

    def test_notional_cap(self):
        env=MarketEnv(np.zeros((2,1)),[0,0]); env.equity=20000; env.peak=20000
        info=env.step(4)[3]; self.assertLessEqual(info['exposure']*20000,1000)

    def test_loss_halt_latched(self):
        env=MarketEnv(np.zeros((3,1)),[-.4,.8,.2]); info=env.step(4)[3]
        self.assertTrue(info['halted']); self.assertFalse(env.i>=env.end)
        info=env.step(4)[3]; self.assertEqual(info['exposure'],0); self.assertTrue(info['halted'])
        self.assertEqual(env.step(4)[3]['exposure'],0)

    def test_terminal_exit_cost(self):
        self.assertLess(run_backtest([[0]],[.01],lambda o,i:4)['total_return'],.001)

    def test_invalid_action(self):
        env=MarketEnv([[0]],[0])
        for a in [-1,5,1.5,True]:
            with self.assertRaises(ValueError): env.step(a)

    def test_future_return_not_observation(self):
        a=MarketEnv([[1,2],[2,3]],[.1,.2]); b=MarketEnv([[1,2],[2,3]],[-.5,.9])
        np.testing.assert_array_equal(a.obs(),b.obs())

    def test_scaler_train_only(self):
        train=np.array([[1,2],[3,4]]); scaler=Scaler.fit(train); before=scaler.export()
        scaler.transform([[999999,-999999]])
        self.assertEqual(before,scaler.export()); self.assertEqual(scaler.mean.tolist(),[2,3])
        self.assertTrue(np.isfinite(Scaler.fit([[0,0],[0,0]]).transform([[1,1]])).all())

    def test_gae_terminal(self):
        advantage,target=gae([1,2],[.5,.7],[False,True],999,gamma=1,lam=1)
        np.testing.assert_allclose(target,[3,2],atol=1e-6)
        np.testing.assert_allclose(advantage,[2.5,1.3],atol=1e-6)

    def test_gae_truncation(self):
        _,target=gae([1],[.5],[False],2,gamma=.9,lam=1)
        self.assertAlmostEqual(float(target[0]),2.8,places=6)

    def test_news_revision_cutoff(self):
        raw=dict(id='1',source='s',headline='x',created_at='2025-01-01T00:00:00Z')
        a=normalize_article(raw); b=normalize_article(dict(raw,headline='revised',updated_at='2025-01-03T00:00:00Z'))
        chosen=select_articles([a,b],'2025-01-02T00:00:00Z')
        self.assertEqual([x['headline'] for x in chosen],['x'])
        self.assertEqual(select_articles([a,b],'2025-01-03T00:00:00Z')[0]['headline'],'revised')

    def test_news_requires_timezone(self):
        with self.assertRaises(ValueError): normalize_article(dict(id='1',source='s',headline='x',available_at='2025-01-01'))

    def test_news_strict_output(self):
        for raw in ['buy NVDA now', '{"sentiment": 10}', '{"sentiment":NaN,"relevance":1,"uncertainty":0,"event":"other"}']:
            self.assertIsNone(parse_signal(raw))
        raw='{"sentiment":0.5,"relevance":1,"uncertainty":0.2,"event":"earnings"}'
        self.assertEqual(parse_signal(raw)['sentiment'],.5)

    def test_missing_news_explicit(self):
        a=normalize_article(dict(id='1',source='s',headline='x',available_at='2025-01-01T00:00:00Z'))
        f=news_features([a],{a['key']:None},'2025-01-02T00:00:00Z')
        self.assertEqual(f['news_missing'],1); self.assertEqual(f['news_valid'],0); self.assertEqual(f['news_count'],1)

    def test_no_future_bars(self):
        bars={s:[dict(t=(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i)).isoformat(),c=10+i*.1,h=11+i*.1,l=9+i*.1,v=100) for i in range(61)] for s in SYMBOLS}
        row=dict(date='2025-03-02',features={s:dict(r20=.1) for s in SYMBOLS})
        before=math_features(row,bars)
        for s in SYMBOLS: bars[s].append(dict(t='2026-01-01T00:00:00Z',c=999,h=999,l=999,v=999999))
        self.assertEqual(before,math_features(row,bars))

    def test_holdout_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ledger.json'; claim_holdout(p,'2025-01-01','2025-02-01','x')
            for start,end in [('2025-01-02','2025-01-03'),('2024-12-01','2025-01-01')]:
                with self.assertRaises(RuntimeError): claim_holdout(p,start,end,'y')
            claim_holdout(p,'2025-02-02','2025-03-01','z'); self.assertEqual(len(read(p)),2)

    def test_independent_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'halt.json'
            ok=guard_decision(4,20000,0,0,1,0,p); self.assertEqual(ok['target_notional'],1000)
            bad=guard_decision(4,20000,-.03,0,1,0,p); self.assertFalse(bad['allowed'])
            self.assertFalse(guard_decision(4,20000,0,0,1,0,p)['allowed'])

    def test_stale_and_pending_orders_halt(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(guard_decision(4,10000,0,0,31,0,Path(d)/'a')['allowed'])
            self.assertFalse(guard_decision(4,10000,0,0,1,0,Path(d)/'b',open_orders=True)['allowed'])
            self.assertFalse(guard_decision(4,10000,0,0,1,.5,Path(d)/'c')['allowed'])

    def test_promotion_and_rollback(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); a=d/'a'; a.write_text('first'); b=d/'b'; b.write_text('second'); reg=d/'champion.json'
            self.assertFalse(promote(reg,a,dict(checks=dict(gate=False))))
            self.assertFalse(reg.exists())
            self.assertTrue(promote(reg,a,dict(checks=dict(gate=True))))
            self.assertTrue(promote(reg,b,dict(checks=dict(gate=True))))
            b.write_text('corrupt')
            with self.assertRaises(RuntimeError): verify_champion(reg)
            self.assertTrue(rollback(reg,d/'halt','integrity'))
            self.assertEqual(verify_champion(reg)['sha256'],filehash(a)); self.assertTrue((d/'halt').exists())


if __name__=='__main__': unittest.main()
