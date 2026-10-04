import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvda_lab as lab


class TestData(unittest.TestCase):
    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):
            lab.stamp('2025-01-01')

    def test_news_revision_cannot_leak_backwards(self):
        cutoff = lab.stamp('2025-01-02T00:00:00Z')
        original = dict(id='1', kind='news', symbol='NVDA', source='test', available_at='2025-01-01T12:00:00Z', text='original')
        revision = dict(original, available_at='2025-01-03T12:00:00Z', text='future')
        result = lab.event_context([original, revision], cutoff)
        self.assertEqual(result['news'][0]['text'], 'original')

    def test_old_event_expires(self):
        e = dict(id='1', kind='macro', symbol='*', source='test', available_at='2020-01-01T00:00:00Z', text='old')
        self.assertEqual(lab.event_context([e], lab.stamp('2025-01-01T00:00:00Z'))['macro'], [])

    def test_modalities_have_independent_budget(self):
        events = [dict(id=str(i), kind='news', symbol='NVDA', source='test', available_at='2025-01-01T00:00:00Z', text='x') for i in range(30)]
        events.append(dict(events[0], kind='filing'))
        result = lab.event_context(events, lab.stamp('2025-01-02T00:00:00Z'))
        self.assertEqual(len(result['news']), 2)
        self.assertEqual(len(result['filing']), 1)

    def test_missing_session_rejected(self):
        raw = lab.read(lab.DATA/'bars.json')
        raw['bars']['QQQ'].pop()
        with self.assertRaises(ValueError):
            lab.normalized_bars(raw)

    def test_future_bars_do_not_change_features(self):
        raw = lab.read(lab.DATA/'bars.json')
        dates, bars = lab.normalized_bars(raw)
        before = lab.market_features(dates, bars, 100)
        for s in lab.SYMBOLS:
            bars[s][dates[101]]['c'] *= 100
        self.assertEqual(before, lab.market_features(dates, bars, 100))

    def test_split_purge_and_label_alignment(self):
        ds = lab.read(lab.DATA/'dataset.json')
        self.assertLess(max(x['label_end'] for x in ds['train']), ds['validation'][0]['date'])
        self.assertLess(max(x['label_end'] for x in ds['validation']), ds['test'][0]['date'])
        dates, bars = lab.normalized_bars(lab.read(lab.DATA/'bars.json'))
        for row in (ds['train'][0], ds['validation'][0], ds['test'][-1]):
            i = dates.index(row['date'])
            expected = bars['NVDA'][dates[i+2]]['o']/bars['NVDA'][dates[i+1]]['o']-1
            self.assertAlmostEqual(expected, row['return'])
            self.assertEqual(dates[i+2], row['label_end'])


class TestRisk(unittest.TestCase):
    def test_budget_does_not_scale_with_large_account(self):
        self.assertEqual(lab.sizing(100000, 100000, 200, 0, True), 5)
        self.assertEqual(lab.sizing(1000, 1000, 200, 0, True), 0)

    def test_cash_and_no_short(self):
        self.assertEqual(lab.sizing(100000, 150, 200, 0, True), 0)
        self.assertEqual(lab.sizing(10000, 5000, 200, 4, False), -4)
        with self.assertRaises(ValueError):
            lab.sizing(10000, 5000, 200, -1, False)

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            lab.sizing(float('nan'), 1000, 200, 0, True)

    def test_quote_stale_crossed_and_spread_rejected(self):
        current = lab.stamp('2026-10-02T14:00:00Z')
        for q in ({'bp': 100, 'ap': 101, 't': current.isoformat()},
                  {'bp': 101, 'ap': 100, 't': current.isoformat()},
                  {'bp': 100, 'ap': 100.01, 't': (current-timedelta(minutes=2)).isoformat()}):
            with self.assertRaises(ValueError):
                lab.check_quote(q, current)
        self.assertEqual(lab.check_quote({'bp': 100, 'ap': 100.01, 't': current.isoformat()}, current), (100, 100.01))

    def test_uncertainty_and_volatility_abstention(self):
        row = {'features': {'NVDA': {'vol20': 0.02}}}
        self.assertEqual(lab.ai_weights([row], [[0, 0, 0]], 1, 0.5), [0])
        row['features']['NVDA']['vol20'] = 0.1
        self.assertEqual(lab.ai_weights([row], [[0, 0, 10]], 1, 0.5), [0])

    def test_costs_include_entry_and_exit(self):
        m = lab.metrics([0], [0.1], 10)
        self.assertAlmostEqual(m['total_return'], -0.0002)

    def test_cash_has_zero_performance(self):
        m = lab.metrics([0.5, -0.5], [0, 0])
        self.assertEqual(m['total_return'], 0)
        self.assertEqual(m['sharpe_zero_rf'], 0)


class FakeAPI:
    def __init__(self, fail=False, missing=False):
        self.posts = 0
        self.fail = fail
        self.missing = missing
    def call(self, path, params=None, body=None, method='GET', **kw):
        if method == 'POST':
            self.posts += 1
            if self.fail:
                raise TimeoutError('response lost after broker accepted')
            return {'status': 'accepted'}
        if self.missing:
            raise lab.APIError(404, 'not found')
        return {'status': 'filled'}


class TestOrders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = patch.object(lab, 'ROOT', Path(self.tmp.name))
        self.patch.start()
        self.db = lab.connect_db()
        self.payload = {'client_order_id': 'nvda-lab-20261002', 'symbol': 'NVDA', 'qty': '1'}
    def tearDown(self):
        self.db.close()
        self.patch.stop()
        self.tmp.cleanup()

    def test_duplicate_submission_reconciles(self):
        api = FakeAPI()
        lab.submit_once(api, self.db, '2026-10-02', self.payload)
        lab.submit_once(api, self.db, '2026-10-02', self.payload)
        self.assertEqual(api.posts, 1)
        self.assertEqual(self.db.execute('SELECT status FROM intent').fetchone()[0], 'filled')

    def test_ambiguous_timeout_is_persisted_and_not_retried(self):
        api = FakeAPI(fail=True)
        with self.assertRaises(TimeoutError):
            lab.submit_once(api, self.db, '2026-10-02', self.payload)
        self.assertIsNotNone(lab.state_get(self.db, 'halt'))
        self.assertEqual(self.db.execute('SELECT status FROM intent').fetchone()[0], 'pending')
        lab.submit_once(api, self.db, '2026-10-02', self.payload)
        self.assertEqual(api.posts, 1)

    def test_missing_ambiguous_order_not_resubmitted(self):
        api = FakeAPI(fail=True, missing=True)
        with self.assertRaises(TimeoutError):
            lab.submit_once(api, self.db, '2026-10-02', self.payload)
        with self.assertRaises(RuntimeError):
            lab.submit_once(api, self.db, '2026-10-02', self.payload)
        self.assertEqual(api.posts, 1)

    def test_hash_detects_model_tampering(self):
        path = Path(self.tmp.name)/'model'
        path.mkdir()
        (path/'adapter.safetensors').write_bytes(b'first')
        before = lab.model_hash(path)
        (path/'report.json').write_text('{}')
        self.assertEqual(before, lab.model_hash(path))
        (path/'adapter.safetensors').write_bytes(b'changed')
        self.assertNotEqual(before, lab.model_hash(path))


class PaperAPI:
    def __init__(self):
        self.posts = []
        self.equity = '10000'
        self.held = []
        self.open_orders = []
    def call(self, path, params=None, body=None, method='GET', **kw):
        if method == 'POST':
            self.posts.append(body)
            return {'status': 'filled'}
        if path == '/v2/account':
            return {'id': 'mock-account', 'status': 'ACTIVE', 'equity': self.equity, 'last_equity': '10000', 'cash': '10000'}
        if path == '/v2/clock':
            return {'timestamp': '2026-10-02T13:36:00Z', 'is_open': True}
        if path == '/v2/calendar':
            return [{'date': '2026-10-01', 'open': '09:30', 'close': '16:00'},
                    {'date': '2026-10-02', 'open': '09:30', 'close': '16:00'}]
        if path == '/v2/positions':
            return self.held
        if path == '/v2/orders':
            return self.open_orders
        if path == '/v2/orders:by_client_order_id':
            return {'status': 'filled'}
        if path == '/v2/stocks/NVDA/quotes/latest':
            return {'quote': {'bp': 200, 'ap': 200.01, 't': '2026-10-02T13:36:00Z'}}
        raise AssertionError('Unexpected request: ' + path)


class TestPaperIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(lab, 'ROOT', self.root), patch.object(lab, 'RUNS', self.root/'runs')]
        for p in self.patches:
            p.start()
        self.model_path = self.root/'runs'/'mock'
        self.model_path.mkdir(parents=True)
        (self.model_path/'adapter.safetensors').write_bytes(b'mock weights, not a real model')
        lab.dump(self.root/'champion.json', {'run': 'mock', 'passed': True, 'model_hash': lab.model_hash(self.model_path),
                 'feed': 'iex', 'temperature': 1, 'threshold': 0.5})
        self.api = PaperAPI()
        self.predictor = SimpleNamespace(logits=lambda rows: [[0, 0, 10]]*len(rows))
    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def test_dry_run_never_posts(self):
        result = lab.paper_tick(SimpleNamespace(execute=False), self.api, self.predictor)
        self.assertEqual(result['mode'], 'dry_run')
        self.assertEqual(self.api.posts, [])
        self.assertLessEqual(float(result['order']['qty'])*float(result['order']['limit_price']), 1000)

    def test_execute_then_restart_reconciles_without_duplicate(self):
        args = SimpleNamespace(execute=True)
        result = lab.paper_tick(args, self.api, self.predictor)
        self.assertEqual(result['result']['submitted'], 'filled')
        second = lab.paper_tick(args, self.api, self.predictor)
        self.assertEqual(second['reconciled'], 'filled')
        self.assertEqual(len(self.api.posts), 1)

    def test_loss_limit_halts_without_order(self):
        self.api.equity = '9700'
        with self.assertRaises(RuntimeError):
            lab.paper_tick(SimpleNamespace(execute=True), self.api, self.predictor)
        self.assertEqual(self.api.posts, [])
        with lab.connect_db() as db:
            self.assertIn('Equity', lab.state_get(db, 'halt'))

    def test_halt_file_blocks(self):
        (self.root/'HALT').touch()
        with self.assertRaises(RuntimeError):
            lab.paper_tick(SimpleNamespace(execute=True), self.api, self.predictor)
        self.assertEqual(self.api.posts, [])

    def test_other_positions_block(self):
        self.api.held = [{'symbol': 'AAPL', 'qty': '1'}]
        with self.assertRaises(RuntimeError):
            lab.paper_tick(SimpleNamespace(execute=True), self.api, self.predictor)
        self.assertEqual(self.api.posts, [])

    def test_tampered_model_blocks(self):
        (self.model_path/'adapter.safetensors').write_bytes(b'tampered')
        with self.assertRaises(RuntimeError):
            lab.paper_tick(SimpleNamespace(execute=True), self.api, self.predictor)
        self.assertEqual(self.api.posts, [])


if __name__ == '__main__':
    unittest.main()
