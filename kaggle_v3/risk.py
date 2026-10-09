"""Independent paper-order limits. All percentages are FRACTIONS, e.g. .02 = 2%."""
import math
from core import POLICY


def check_order(*, side, qty, limit_price, held_qty, equity, cash, day_return,
                drawdown, quote_age, open_orders, halted):
    values=[qty,limit_price,held_qty,equity,cash,day_return,drawdown,quote_age]
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in values):
        return False,'invalid_numeric_state'
    if halted: return False,'persistent_halt'
    if open_orders: return False,'unreconciled_orders'
    if equity<=0 or cash<0 or held_qty<0 or qty<=0 or int(qty)!=qty or limit_price<=0:
        return False,'invalid_account_or_quantity'
    if not 0<=quote_age<=30 or not 0<=drawdown<=1 or day_return<=-1:
        return False,'stale_or_invalid_state'
    if day_return<=-POLICY['day_loss'] or drawdown>=POLICY['max_drawdown']:
        return False,'loss_limit'
    if side not in ('buy','sell'): return False,'invalid_side'
    if side=='sell' and qty>held_qty: return False,'short_sale_blocked'
    if side=='buy':
        if qty*limit_price>cash: return False,'cash_limit'
        if (held_qty+qty)*limit_price>min(equity*POLICY['max_weight'],POLICY['max_notional'])+1e-8:
            return False,'position_limit'
    return True,'ok'
