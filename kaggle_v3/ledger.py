"""Durable experiment memory. Deleting/restoring older state invalidates the protocol."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from core import digest, read


def now():
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, root):
        self.root = Path(root)
        path = self.root / 'memory' / 'ledger.db'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS hypotheses (
          id TEXT PRIMARY KEY, idea TEXT NOT NULL, market TEXT NOT NULL,
          source TEXT NOT NULL, regime TEXT NOT NULL, created TEXT NOT NULL,
          status TEXT NOT NULL, lessons TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE IF NOT EXISTS campaigns (
          id TEXT PRIMARY KEY, spec TEXT NOT NULL, spec_hash TEXT NOT NULL,
          created TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS attempts (
          id TEXT PRIMARY KEY, hypothesis TEXT NOT NULL, campaign TEXT NOT NULL,
          parameters TEXT NOT NULL, prediction TEXT NOT NULL, created TEXT NOT NULL,
          status TEXT NOT NULL, result TEXT);
        CREATE TABLE IF NOT EXISTS trials (
          id TEXT PRIMARY KEY, attempts INTEGER NOT NULL, sharpe REAL);
        CREATE TABLE IF NOT EXISTS holdouts (
          id TEXT PRIMARY KEY, start TEXT NOT NULL, end TEXT NOT NULL,
          created TEXT NOT NULL, run_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, created TEXT NOT NULL, kind TEXT NOT NULL,
          subject TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS orders (
          id TEXT PRIMARY KEY, created TEXT NOT NULL, request TEXT NOT NULL,
          status TEXT NOT NULL, broker_id TEXT);
        CREATE TABLE IF NOT EXISTS observations (
          id TEXT PRIMARY KEY, created TEXT NOT NULL, artifact TEXT NOT NULL,
          payload TEXT NOT NULL);
        ''')
        self.db.execute('INSERT OR IGNORE INTO settings VALUES (?,?)',
                        ('fresh_after', max('2026-10-02', now()[:10])))
        if self.db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('Research memory integrity check failed; restore a verified backup')

    def close(self):
        self.db.close()

    def event(self, kind, subject, payload):
        self.db.execute('INSERT INTO events(created,kind,subject,payload) VALUES (?,?,?,?)',
                        (now(), kind, subject, json.dumps(payload, allow_nan=False)))

    def campaign(self, name, spec):
        encoded = json.dumps(spec, sort_keys=True, allow_nan=False)
        self.db.execute('INSERT OR IGNORE INTO campaigns VALUES (?,?,?,?,?)',
                        (name, encoded, digest(spec), now(), 'running'))
        row = self.db.execute('SELECT * FROM campaigns WHERE id=?', (name,)).fetchone()
        if row['spec_hash'] != digest(spec):
            raise RuntimeError('Campaign specification changed; retain history and use a new cycle')
        return dict(row)

    def hypothesis(self, idea, regime, source='bounded_research', market='NVDA'):
        identity = digest(dict(idea=idea.strip().lower(), market=market))
        self.db.execute('INSERT OR IGNORE INTO hypotheses VALUES (?,?,?,?,?,?,?,?)',
                        (identity, idea, market, source, regime, now(), 'untested', '[]'))
        return identity

    def search(self, market='NVDA', regime=None, status=None):
        query = 'SELECT * FROM hypotheses WHERE market=?'; params = [market]
        for key, value in [('regime', regime), ('status', status)]:
            if value is not None:
                query += ' AND '+key+'=?'; params.append(value)
        return [dict(r) for r in self.db.execute(query, params)]

    def reserve(self, identity, hypothesis, campaign, parameters, prediction, cap=3):
        """Prediction and attempt count commit BEFORE training; retries keep the same ID."""
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.db.execute('SELECT * FROM attempts WHERE id=?', (identity,)).fetchone()
            encoded = json.dumps(parameters, sort_keys=True, allow_nan=False)
            predicted = json.dumps(prediction, sort_keys=True, allow_nan=False)
            if old:
                if (old['hypothesis'], old['campaign'], old['parameters'], old['prediction']) != (
                        hypothesis, campaign, encoded, predicted):
                    raise RuntimeError('An immutable attempt/prediction changed')
            else:
                count = self.db.execute('SELECT COUNT(*) FROM attempts WHERE hypothesis=?', (hypothesis,)).fetchone()[0]
                if count >= cap:
                    raise RuntimeError('Hypothesis revision budget exhausted; failures stay in memory')
                self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,NULL)',
                                (identity, hypothesis, campaign, encoded, predicted, now(), 'running'))
            self.db.execute('COMMIT')
            return dict(old) if old else None
        except BaseException:
            self.db.execute('ROLLBACK'); raise

    def finish(self, identity, result):
        row = self.db.execute('SELECT * FROM attempts WHERE id=?', (identity,)).fetchone()
        encoded = json.dumps(result, sort_keys=True, allow_nan=False)
        if row['result'] is not None and row['result'] != encoded:
            raise RuntimeError('Attempt result is immutable')
        status = 'passed_backtest_only' if result['passed'] else 'failed'
        lessons = [k for k, value in result['checks'].items() if not value]
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.db.execute('UPDATE attempts SET status=?,result=? WHERE id=?', (status, encoded, identity))
            prior = json.loads(self.db.execute('SELECT lessons FROM hypotheses WHERE id=?', (row['hypothesis'],)).fetchone()[0])
            self.db.execute('UPDATE hypotheses SET status=?,lessons=? WHERE id=?',
                            (status, json.dumps(sorted(set(prior+lessons))), row['hypothesis']))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); raise

    def reserve_trials(self, identity, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('Invalid trial count')
        self.db.execute('INSERT OR IGNORE INTO trials VALUES (?,?,NULL)', (identity, count))
        if self.db.execute('SELECT attempts FROM trials WHERE id=?', (identity,)).fetchone()[0] != count:
            raise RuntimeError('Trial budget changed')

    def trial_count(self):
        return self.db.execute('SELECT COALESCE(SUM(attempts),0) FROM trials').fetchone()[0]

    def import_history(self):
        for path in sorted((self.root / 'cycles').glob('*/candidate_*_history.json')):
            cfg = read(path.parent/'config.json') if (path.parent/'config.json').exists() else {}
            if cfg.get('evolve'): continue
            prepared = path.parent/'prepared.json'
            if prepared.exists() and self.db.execute('SELECT 1 FROM trials WHERE id=?',
                    ('ppo:'+read(prepared)['fingerprint'],)).fetchone(): continue
            self.reserve_trials('legacy:'+str(path.relative_to(self.root)), max(1, cfg.get('updates',len(read(path)))))
        legacy = self.root / 'holdout_ledger.json'
        for row in read(legacy) if legacy.exists() else []:
            key = 'legacy:'+digest(row)
            self.db.execute('INSERT OR IGNORE INTO holdouts VALUES (?,?,?,?,?)',
                            (key, row['start'], row['end'], now(), row.get('run_id', 'legacy')))

    def claim_holdout(self, start, end, run_id):
        # Dates are canonical ISO; reject malformed ranges before comparison.
        from datetime import date
        if date.fromisoformat(start).isoformat()!=start or date.fromisoformat(end).isoformat()!=end or start>end:
            raise ValueError('Invalid holdout dates')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT 1 FROM holdouts WHERE NOT(end < ? OR start > ?)', (start, end)).fetchone():
                raise RuntimeError('Holdout overlaps an already consumed interval; collect fresh dates')
            self.db.execute('INSERT INTO holdouts VALUES (?,?,?,?,?)',
                            (digest([start, end, run_id]), start, end, now(), run_id))
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK'); raise

    @property
    def fresh_after(self):
        return self.db.execute("SELECT value FROM settings WHERE key='fresh_after'").fetchone()[0]
