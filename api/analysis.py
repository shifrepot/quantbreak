from http.server import BaseHTTPRequestHandler
import json, yfinance as yf, numpy as np
from scipy.stats import norm
from urllib.parse import urlparse, parse_qs

TICKER_MAP = {
    "TQQQ": "TQQQ", "SOXL": "SOXL", "SQQQ": "SQQQ",
    "WTI": "USO",   "NG": "UNG",    "BOIL": "BOIL",
}
LEVERAGES = [0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]

def get_closes(sym, period="65d"):
    tk   = yf.Ticker(sym)
    hist = tk.history(period=period)
    if hasattr(hist.columns, "levels"):
        hist.columns = hist.columns.get_level_values(0)
    closes = hist["Close"].dropna()
    if len(closes) < 10:
        raise ValueError(f"Not enough data: {len(closes)} rows")
    return closes

def cvar(returns, alpha=0.05):
    var  = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) > 0 else float(var)

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        qs    = parse_qs(urlparse(self.path).query)
        asset = qs.get("asset", ["TQQQ"])[0].upper()
        sym   = TICKER_MAP.get(asset, asset)

        try:
            closes  = get_closes(sym, "65d")
            returns = closes.pct_change().dropna().values

            sigma = float(np.std(returns) * np.sqrt(252))
            S     = float(closes.iloc[-1])
            K, T, r = S, 0.5, 0.05

            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            prob_profit    = float(norm.cdf(d2))
            bs_expected_ret = float((np.exp(r*T) - 1) * 100)
            monthly_sigma  = sigma / np.sqrt(12)
            bs_tail_prob   = float(norm.cdf(-0.30 / monthly_sigma) * 100)

            # Fat Tail
            n = len(returns)
            monthly_rets = [
                float(np.prod(1 + returns[i:i+21]) - 1)
                for i in range(0, n-21, 5)
            ]
            crash_count   = sum(1 for r_ in monthly_rets if r_ < -0.30)
            actual_tail   = crash_count / max(len(monthly_rets), 1) * 100
            fat_tail_ratio = round(actual_tail / bs_tail_prob, 1) if bs_tail_prob > 0 else 8.3

            # Vol Decay
            lev3 = float(np.prod(1 + returns*3) - 1) * 100
            lev1 = float(np.prod(1 + returns)   - 1) * 100
            vol_decay = round(lev3 - lev1*3, 1)

            # CVaR by leverage
            cvar_by_lev  = []
            mean_by_lev  = []
            for lev in LEVERAGES:
                lr = returns * lev
                cvar_by_lev.append(round(cvar(lr) * 100, 2))
                mean_by_lev.append(float(np.mean(lr)))

            feasible = [(i, cvar_by_lev[i]) for i in range(len(LEVERAGES)) if mean_by_lev[i] > 0]
            opt_idx  = max(feasible, key=lambda x: -x[1])[0] if feasible else 4

            max_abs  = max(abs(v) for v in cvar_by_lev) or 1
            energies = [round(abs(c)/max_abs, 4) for c in cvar_by_lev]

            body = json.dumps({
                "sigma":            round(sigma*100, 1),
                "bs_prob_profit":   round(prob_profit*100, 1),
                "bs_expected_ret":  round(bs_expected_ret, 1),
                "bs_tail_prob":     round(bs_tail_prob, 3),
                "fat_tail_ratio":   fat_tail_ratio,
                "vol_decay":        vol_decay,
                "cvar_5pct":        round(cvar(returns)*100, 2),
                "cvar_by_leverage": cvar_by_lev,
                "optimal_leverage": LEVERAGES[opt_idx],
                "optimal_idx":      opt_idx,
                "energies":         energies,
            })
        except Exception as e:
            body = json.dumps({"error": str(e)})

        self._respond(body)

    def _respond(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass
