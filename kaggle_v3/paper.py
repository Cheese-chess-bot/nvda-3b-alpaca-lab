"""Opt-in Alpaca PAPER adapter. Training never imports or runs this module.
An upstream point-in-time feature snapshot is required for every decision.
"""
import argparse
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from core import POLICY, digest, dump, filehash, read, utc, verify_champion
from ledger import Ledger, now
from recovery import controller_lock, health
from risk import check_order


def stage(work, root):
    """Stage one validated artifact for paper observation; no broker requests."""
    work,root=Path(work),Path(root)
    summary=read(work/'evolution'/'summary.json'); holdout=read(work/'holdout_report.json')
    winner=summary['winner']; bundle=work/winner['bundle']
    if not summary['passed'] or not holdout.get('research_passed') or not holdout['passed'] or not all(holdout['checks'].values()):
        raise RuntimeError('Development AND fresh holdout gates must pass before paper observation')
    if holdout['candidate_sha256']!=winner['sha256'] or filehash(bundle)!=winner['sha256']:
        raise RuntimeError('Candidate/holdout integrity mismatch')
    target=root/'artifacts'/(winner['sha256']+'.pt'); target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and filehash(target)!=winner['sha256']: raise RuntimeError('Stored artifact corrupted')
    if not target.exists():
        temporary=target.with_suffix('.tmp'); shutil.copyfile(bundle,temporary); os.replace(temporary,target)
    entry=dict(bundle=str(target.resolve()),sha256=winner['sha256'],policy_hash=digest(POLICY),
               report=holdout,hypothesis=summary['hypothesis'],staged_at=now(),stage='paper_observation')
    registry=root/'paper_candidate.json'
    if registry.exists():
        previous=verify_champion(registry)
        if previous['sha256']==entry['sha256']: return previous
        entry['previous']=previous
    dump(registry,entry)
    return entry


class AlpacaPaper:
    """Fixed SDK paper environment; no live or arbitrary URL option."""
    def __init__(self):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        key=os.environ.get('APCA_API_KEY_ID'); secret=os.environ.get('APCA_API_SECRET_KEY')
        if not key or not secret: raise RuntimeError('Both Alpaca paper credentials are required in the environment')
        self.trading=TradingClient(key,secret,paper=True)
        self.data=StockHistoricalDataClient(key,secret)

    def snapshot(self):
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        account=self.trading.get_account(); clock=self.trading.get_clock()
        if not clock.is_open: return dict(market_open=False)
        quote=self.data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols='NVDA',feed=DataFeed.IEX))['NVDA']
        positions=self.trading.get_all_positions()
        orders=self.trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN,limit=500))
        foreign=any(p.symbol!='NVDA' for p in positions)
        held=sum(float(p.qty) for p in positions if p.symbol=='NVDA')
        return dict(market_open=True,equity=float(account.equity),cash=float(account.cash),
            last_equity=float(account.last_equity),held_qty=held,foreign_positions=foreign,
            account_key=digest(str(account.id)),open_orders=bool(orders),
            blocked=bool(account.trading_blocked or account.account_blocked),
            bid=float(quote.bid_price),ask=float(quote.ask_price),quote_at=quote.timestamp.isoformat(),
            observed_at=now(),session=clock.timestamp.astimezone(__import__('zoneinfo').ZoneInfo('America/New_York')).date().isoformat())

    def submit(self, request):
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        order=self.trading.submit_order(LimitOrderRequest(symbol='NVDA',qty=request['qty'],
            side=OrderSide.BUY if request['side']=='buy' else OrderSide.SELL,
            limit_price=request['limit_price'],time_in_force=TimeInForce.DAY,
            client_order_id=request['client_order_id']))
        return str(order.id)

    def lookup(self, client_id):
        return self.trading.get_order_by_client_id(client_id)


def tick(root, features, api, submit=False, proposer=None):
    from runtime import propose
    root=Path(root); ledger=Ledger(root); halt=root/'risk_halt.json'
    def blocked(reason):
        dump(halt,dict(halted=True,reason=reason)); ledger.event('blocked','paper',dict(reason=reason))
        return dict(allowed=False,reason=reason)
    try:
        entry=verify_champion(root/'paper_candidate.json')
        if halt.exists(): return dict(allowed=False,reason='persistent_halt')
        if ledger.db.execute("SELECT 1 FROM orders WHERE status IN ('reserved','unknown')").fetchone():
            return blocked('unresolved_submission; reconcile by client_order_id')
        snap=api.snapshot()
        if not snap['market_open']: return dict(allowed=False,reason='market_closed')
        numbers=[snap[k] for k in ('equity','cash','last_equity','held_qty','bid','ask')]
        if not all(math.isfinite(x) for x in numbers) or min(snap['equity'],snap['last_equity'],snap['bid'])<=0 or snap['ask']<snap['bid']:
            return blocked('invalid_broker_state')
        if snap['blocked'] or snap['foreign_positions'] or snap['held_qty']<0 or snap['cash']<0:
            return blocked('dedicated_unleveraged_NVDA_paper_account_required')
        account=ledger.db.execute("SELECT value FROM settings WHERE key='paper_account'").fetchone()
        if account and account[0]!=snap['account_key']: return blocked('paper_account_changed')
        ledger.db.execute("INSERT OR IGNORE INTO settings VALUES ('paper_account',?)",(snap['account_key'],))
        history=[json.loads(r[0]) for r in ledger.db.execute('SELECT payload FROM observations')]
        peak=max([snap['equity'],snap['last_equity']]+[v['equity'] for v in history])
        age=(utc(now())-utc(snap['quote_at'])).total_seconds()
        observed_age=(utc(now())-utc(snap['observed_at'])).total_seconds()
        if not 0<=observed_age<=30: return blocked('stale_account_snapshot')
        if (snap['ask']-snap['bid'])/snap['ask']>.005: return blocked('wide_spread')
        fn=proposer or propose
        proposal=fn(root,features['raw_features'],equity=snap['equity'],
            position_weight=snap['held_qty']*snap['ask']/snap['equity'],peak_equity=peak,
            day_return=snap['equity']/snap['last_equity']-1,quote_age_seconds=age,
            features_as_of=features['as_of'],decision_as_of=now(),open_orders=snap['open_orders'],
            registry_name='paper_candidate.json')
        ledger.event('decision',entry['sha256'],dict(snapshot=snap,proposal=proposal,feature_hash=digest(features),submit_enabled=submit))
        if not proposal['allowed']: return proposal
        ledger.db.execute('INSERT OR IGNORE INTO observations VALUES (?,?,?,?)',
            (entry['sha256']+':'+snap['session']+('paper' if submit else 'shadow'),now(),entry['sha256'],json.dumps(dict(snap,mode='paper' if submit else 'shadow'))))
        target=proposal['target_notional']; held=snap['held_qty']; delta=target-held*snap['ask']
        side='buy' if delta>0 else 'sell'
        price=round(snap['ask']*1.001,2) if side=='buy' else max(.01,round(snap['bid']*.999,2))
        qty=math.floor(max(0,min(target,snap['equity']*.1,1000.))/price-held) if side=='buy' else math.floor(min(held,-delta/snap['ask']))
        if qty<=0: return dict(allowed=True,submitted=False,reason='no_whole_share_rebalance')
        ok,reason=check_order(side=side,qty=qty,limit_price=price,held_qty=held,equity=snap['equity'],
            cash=snap['cash'],day_return=snap['equity']/snap['last_equity']-1,drawdown=1-snap['equity']/peak,
            quote_age=age,open_orders=snap['open_orders'],halted=halt.exists())
        if not ok: return blocked(reason)
        # One order identity per session across all candidate versions prevents replacement double-orders.
        client_id='nv3-'+digest(snap['account_key']+snap['session'])[:32]
        request=dict(client_order_id=client_id,side=side,qty=qty,limit_price=price)
        if not submit: return dict(allowed=True,submitted=False,dry_run=request)
        if ledger.db.execute('SELECT 1 FROM orders WHERE id=?',(client_id,)).fetchone():
            return dict(allowed=False,reason='session_order_already_recorded')
        latest=api.snapshot()
        if not latest['market_open'] or latest['session']!=snap['session'] or latest['held_qty']!=held or latest['account_key']!=snap['account_key']:
            return blocked('broker_state_changed_before_submission')
        if latest['blocked'] or latest['foreign_positions']: return blocked('broker_reconciliation_required')
        if not 0<latest['last_equity'] or latest['ask']<latest['bid'] or latest['bid']<=0:
            return blocked('invalid_final_snapshot')
        if not 0<=(utc(now())-utc(latest['observed_at'])).total_seconds()<=30:
            return blocked('stale_final_snapshot')
        ok,reason=check_order(side=side,qty=qty,limit_price=max(price,latest['ask']) if side=='buy' else price,
            held_qty=held,equity=latest['equity'],cash=latest['cash'],
            day_return=latest['equity']/latest['last_equity']-1,
            drawdown=1-latest['equity']/max(peak,latest['equity']),
            quote_age=(utc(now())-utc(latest['quote_at'])).total_seconds(),
            open_orders=latest['open_orders'],halted=halt.exists())
        if not ok: return blocked(reason)
        ledger.db.execute('INSERT INTO orders VALUES (?,?,?,?,NULL)',(client_id,now(),json.dumps(request),'reserved'))
        try: broker_id=api.submit(request)
        except Exception:
            ledger.db.execute("UPDATE orders SET status='unknown' WHERE id=?",(client_id,))
            return blocked('submission_uncertain; never retry blindly')
        ledger.db.execute("UPDATE orders SET status='submitted',broker_id=? WHERE id=?",(broker_id,client_id))
        ledger.event('submitted',entry['sha256'],dict(client_order_id=client_id,broker_id=broker_id))
        return dict(allowed=True,submitted=True,client_order_id=client_id)
    except Exception as exc:
        health(root)
        return blocked('paper_runtime_'+type(exc).__name__)
    finally: ledger.close()


def review(root, artifact=None):
    """Weekly/on-demand critic. Account observations are diagnostics, not verified alpha."""
    root=Path(root); ledger=Ledger(root)
    try:
        if artifact is None: artifact=verify_champion(root/'paper_candidate.json')['sha256']
        rows=[json.loads(r[0]) for r in ledger.db.execute('SELECT payload FROM observations WHERE artifact=? ORDER BY created',(artifact,))]
        shadow_sessions=sum(r.get('mode')!='paper' for r in rows)
        rows=[r for r in rows if r.get('mode')=='paper']
        from datetime import timedelta
        cutoff=(datetime.now(timezone.utc)-timedelta(days=7)).isoformat()
        recent=[dict(r) for r in ledger.db.execute('SELECT kind,created,payload FROM events WHERE created>=? AND kind IN (\'blocked\',\'decision\',\'submitted\')',(cutoff,))]
        result=dict(artifact=artifact,sessions=len(rows),shadow_sessions=shadow_sessions,recent_events=len(recent),status='insufficient_paper_history',
                    automatic_promotion=False,limitations=['Dedicated account equity; transfers/cash flows are not attribution-adjusted.',
                     'First observation each session, not official closing NAV.', 'Broker fills and cash flows need review before promotion.'])
        if len(rows)>=2:
            equities=[r['equity'] for r in rows]; peak=equities[0]; dd=0.
            for value in equities: peak=max(peak,value); dd=max(dd,1-value/peak)
            result.update(observed_return=equities[-1]/equities[0]-1,max_drawdown=dd)
            result['status']='paused' if dd>=POLICY['max_drawdown'] else 'ready_for_human_review' if len(rows)>=20 else 'observing'
            if result['status']=='paused': dump(root/'risk_halt.json',dict(halted=True,reason='paper_review_drawdown'))
        ledger.event('critic_report',artifact,result)
        entry=read(root/'paper_candidate.json')
        hypothesis=entry.get('hypothesis')
        if hypothesis:
            row=ledger.db.execute('SELECT lessons FROM hypotheses WHERE id=?',(hypothesis,)).fetchone()
            if row:
                lessons=json.loads(row[0]); lesson='paper:'+result['status']
                ledger.db.execute('UPDATE hypotheses SET lessons=? WHERE id=?',(json.dumps(sorted(set(lessons+[lesson]))),hypothesis))
        dump(root/'journal'/('review-'+now()[:10]+'.json'),result)
        return result
    finally: ledger.close()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--state',required=True)
    p.add_argument('--stage'); p.add_argument('--features'); p.add_argument('--submit-paper',action='store_true')
    p.add_argument('--review',action='store_true'); p.add_argument('--reconcile')
    a=p.parse_args(); root=Path(a.state)
    with controller_lock(root/'controller.lock'):
        if a.stage: result=stage(a.stage,root)
        elif a.review: result=review(root)
        elif a.reconcile:
            ledger=Ledger(root)
            try:
                if not ledger.db.execute('SELECT 1 FROM orders WHERE id=?',(a.reconcile,)).fetchone(): raise ValueError('Unknown local order ID')
                order=AlpacaPaper().lookup(a.reconcile)
                ledger.db.execute('UPDATE orders SET status=?,broker_id=? WHERE id=?',(str(order.status),str(order.id),a.reconcile))
                result=dict(reconciled=True,halt_cleared=False)
            finally: ledger.close()
        elif a.features: result=tick(root,read(a.features),AlpacaPaper(),a.submit_paper)
        else: result=health(root)
        print(json.dumps(result,indent=2))


if __name__=='__main__': main()
