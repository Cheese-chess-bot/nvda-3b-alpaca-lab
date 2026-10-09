"""Offline failure-path tests; all paper APIs are fakes. No credentials/network."""
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from core import POLICY, digest, dump, filehash, read, claim_holdout
from ledger import Ledger, now
from recovery import controller_lock, health, run
from research import PREDICTION, propose
from risk import check_order
from validation import development, folds, deflated_sharpe


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); self.db=Ledger(self.root)
        self.h=self.db.hypothesis('testable hypothesis','choppy')
    def tearDown(self): self.db.close(); self.tmp.cleanup()
    def test_prediction_immutable_and_resume_does_not_count_twice(self):
        self.db.reserve('a',self.h,'c',{'lr':1},PREDICTION)
        self.db.reserve('a',self.h,'c',{'lr':1},PREDICTION)
        self.assertEqual(self.db.db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0],1)
        with self.assertRaises(RuntimeError): self.db.reserve('a',self.h,'c',{'lr':2},PREDICTION)
    def test_cap_survives_new_campaign(self):
        for i in range(3): self.db.reserve(str(i),self.h,'c',{'n':i},PREDICTION)
        with self.assertRaises(RuntimeError): self.db.reserve('new',self.h,'other',{},PREDICTION)
    def test_failed_ideas_and_lessons_are_searchable(self):
        self.db.reserve('a',self.h,'c',{},PREDICTION)
        self.db.finish('a',dict(passed=False,checks={'stress_positive':False}))
        self.assertEqual(self.db.search(status='failed')[0]['lessons'],'["stress_positive"]')
    def test_holdout_crash_reservation_is_not_reset(self):
        self.db.claim_holdout('2030-01-01','2030-03-01','a')
        other=Ledger(self.root)
        try:
            with self.assertRaises(RuntimeError): other.claim_holdout('2030-02-01','2030-04-01','b')
        finally: other.close()
    def test_legacy_holdout_and_sqlite_share_ranges(self):
        dump(self.root/'holdout_ledger.json',[dict(start='2025-01-01',end='2025-06-01',run_id='v2')])
        with self.assertRaises(RuntimeError): claim_holdout(self.root/'holdout_ledger.json','2025-02-01','2025-08-01','v3')
    def test_trial_count_and_campaign_immutable(self):
        self.db.reserve_trials('a',40); self.db.reserve_trials('a',40)
        self.assertEqual(self.db.trial_count(),40)
        self.db.campaign('c',{'x':1})
        with self.assertRaises(RuntimeError): self.db.campaign('c',{'x':2})
    def test_changed_regime_does_not_reset_same_hypothesis(self):
        self.assertEqual(self.h,self.db.hypothesis('testable hypothesis','high_volatility'))


class ValidationTests(unittest.TestCase):
    def test_walkforward_purges_and_train_only_scales(self):
        from datetime import date,timedelta
        dates=[(date(2020,1,1)+timedelta(days=i)).isoformat() for i in range(402)]
        part=dict(raw=[[float(i)] for i in range(400)],returns=[.01]*400,dates=dates[:400],label_ends=dates[2:],as_of=dates[:400])
        for packs,oos,scaler in folds(part):
            self.assertLess(max(packs['train']['label_ends']),packs['validation']['dates'][0])
            self.assertLess(max(packs['validation']['label_ends']),oos['dates'][0])
            self.assertAlmostEqual(scaler['mean'][0],np.mean(packs['train']['raw']))
    def test_test_partition_never_requested(self):
        class Guard(dict):
            def __getitem__(self,key):
                if key=='test': raise AssertionError('test accessed')
                return super().__getitem__(key)
        part={k:[1] for k in ('raw','returns','dates','label_ends','as_of')}
        self.assertEqual(len(development({'partitions':Guard(train=part,validation=part)})['returns']),2)
    def test_dsr_more_trials_lowers_score(self):
        returns=np.random.default_rng(4).normal(.001,.01,400)
        self.assertGreater(deflated_sharpe(returns,1)['probability'],deflated_sharpe(returns,500)['probability'])
    def test_zero_variance_does_not_pass(self):
        self.assertEqual(deflated_sharpe([.01]*100,1)['probability'],0)
    def test_proposals_react_to_failure_and_never_repeat(self):
        a=propose(set(),[]); b=propose({digest(a)},[{'stress_positive':False}])
        self.assertEqual(b['learning_rate'],1e-4)
        c=propose({digest(a),digest(b)},[])
        self.assertIsNone(propose({digest(a),digest(b),digest(c)},[]))


class RecoveryTests(unittest.TestCase):
    def test_corrupt_checkpoint_restores_verified_predecessor(self):
        from ppo import save_torch,load_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            save_torch(path,dict(fingerprint='same',update=1))
            save_torch(path,dict(fingerprint='same',update=2))
            path.write_bytes(b'corrupt')
            self.assertEqual(load_checkpoint(path,'same')['update'],1)
            with self.assertRaises(RuntimeError): load_checkpoint(path,'different')
    def test_live_lock_not_stolen_and_released_after_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            with controller_lock(path):
                with self.assertRaises(RuntimeError):
                    with controller_lock(path): pass
            with controller_lock(path): pass
    def test_deadline_kills_quiet_child(self):
        env=dict(os.environ,NVDA_DEADLINE=str(time.time()+.4))
        start=time.monotonic()
        with self.assertRaises(TimeoutError): run([sys.executable,'-c','import time; time.sleep(30)'],env=env)
        self.assertLess(time.monotonic()-start,12)
    def test_corrupt_artifact_rolls_back_but_halt_stays(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); good=root/'good.pt'; good.write_bytes(b'verified')
            prior=dict(bundle=str(good),sha256=filehash(good),policy_hash=digest(POLICY))
            dump(root/'paper_candidate.json',dict(bundle=str(root/'missing'),sha256='bad',policy_hash=digest(POLICY),previous=prior))
            result=health(root)
            self.assertTrue(result['halted']); self.assertTrue(result['registries']['paper_candidate.json']['rollback'])
            self.assertEqual(read(root/'paper_candidate.json')['sha256'],prior['sha256'])


class RiskTests(unittest.TestCase):
    def base(self):
        return dict(side='buy',qty=1,limit_price=100.,held_qty=0.,equity=10000.,cash=10000.,day_return=0.,drawdown=0.,quote_age=1.,open_orders=False,halted=False)
    def test_valid_order(self): self.assertTrue(check_order(**self.base())[0])
    def test_limits_nan_and_position_value_are_enforced(self):
        for changes in ({'qty':float('nan')},{'halted':True},{'open_orders':True},{'quote_age':31.},
                        {'day_return':-.02},{'drawdown':.08},{'held_qty':10.},{'cash':1.},
                        {'side':'sell','qty':2.,'held_qty':1.}):
            self.assertFalse(check_order(**dict(self.base(),**changes))[0],str(changes))


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        artifact=self.root/'paper.pt'; artifact.write_bytes(b'fake test artifact')
        dump(self.root/'paper_candidate.json',dict(bundle=str(artifact),sha256=filehash(artifact),policy_hash=digest(POLICY)))
        class Fake:
            def __init__(api): api.calls=0; api.fail=False
            def snapshot(api): return dict(market_open=True,equity=10000.,cash=10000.,last_equity=10000.,held_qty=0.,
                bid=99.99,ask=100.,quote_at=now(),observed_at=now(),session='2026-10-09',account_key='test',
                foreign_positions=False,open_orders=False,blocked=False)
            def submit(api,request):
                api.calls+=1
                if api.fail: raise TimeoutError('uncertain')
                return 'fake-order'
        self.api=Fake(); self.features=dict(raw_features={},as_of=now())
        self.proposer=lambda *a,**kw:dict(allowed=True,target_notional=1000.)
    def tearDown(self): self.tmp.cleanup()
    def test_dry_run_submits_nothing(self):
        from paper import tick
        result=tick(self.root,self.features,self.api,proposer=self.proposer)
        self.assertIn('dry_run',result); self.assertEqual(self.api.calls,0)
    def test_same_session_is_idempotent(self):
        from paper import tick
        self.assertTrue(tick(self.root,self.features,self.api,True,self.proposer)['submitted'])
        result=tick(self.root,self.features,self.api,True,self.proposer)
        self.assertEqual(result['reason'],'session_order_already_recorded'); self.assertEqual(self.api.calls,1)
    def test_timeout_never_resubmits(self):
        from paper import tick
        self.api.fail=True
        tick(self.root,self.features,self.api,True,self.proposer)
        tick(self.root,self.features,self.api,True,self.proposer)
        self.assertEqual(self.api.calls,1); self.assertTrue((self.root/'risk_halt.json').exists())
    def test_staging_failed_research_is_blocked(self):
        from paper import stage
        dump(self.root/'evolution'/'summary.json',dict(passed=False,winner=dict(bundle='paper.pt',sha256=filehash(self.root/'paper.pt'))))
        dump(self.root/'holdout_report.json',dict(passed=True,research_passed=False,checks={'ok':True}))
        with self.assertRaises(RuntimeError): stage(self.root,self.root)


if __name__=='__main__': unittest.main()
